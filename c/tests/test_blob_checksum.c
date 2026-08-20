/* test_blob_checksum.c -- verify-on-first-touch against __metadata__["colibri.sha256"] (#13).
 *
 * The engine-level half of the checksum feature: st_verify_once's contract.
 * The primitive itself is pinned separately against NIST vectors in
 * test_sha256.c -- this covers when it runs, when it does not, and what
 * happens on a mismatch.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include "../st.h"

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

/* One shard, one tensor "w" of `n` bytes, optionally with a checksum map. */
static void write_fixture(const char *dir, int with_sums, const char *forced_digest){
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    enum { N = 4096 };
    static unsigned char q[N];
    for(int i=0;i<N;i++) q[i]=(unsigned char)((i*37+11)&0xff);
    char digest[65];
    { Sha256 s; sha256_init(&s); sha256_update(&s,q,N);
      unsigned char d[32]; sha256_final(&s,d); sha256_hex(d,digest); }
    const char *use = forced_digest ? forced_digest : digest;

    char hdr[1024]; int hl;
    if(with_sums)
        hl=snprintf(hdr,sizeof hdr,
            "{\"__metadata__\":{\"colibri.sha256\":\"{\\\"w\\\":\\\"%s\\\"}\"},"
            "\"w\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]}}", use, N, N);
    else
        hl=snprintf(hdr,sizeof hdr,
            "{\"w\":{\"dtype\":\"U8\",\"shape\":[%d],\"data_offsets\":[0,%d]}}", N, N);
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/)\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,(size_t)hl,f); fwrite(q,1,N,f); fclose(f);
}

static void rm_fixture(const char *dir){
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

int main(void){
    unsigned char buf[4096];

    /* 1. no checksum map: verification is a no-op, and the container loads. */
    { const char *dir="tests/tmp_sum_none";
      write_fixture(dir,0,NULL);
      shards S; st_init(&S,dir);
      CHECK(S.sum_n == 0);
      st_read_raw(&S,"w",buf,0);                 /* must not refuse */
      CHECK(st_sum(&S,"w") == NULL);
      rm_fixture(dir); }

    /* 2. matching digest: verifies once, then marks the tensor so a second read
     *    does not hash again -- the property that keeps a hot expert cheap. */
    { const char *dir="tests/tmp_sum_ok";
      write_fixture(dir,1,NULL);
      shards S; st_init(&S,dir);
      CHECK(S.sum_n == 1);
      st_tensor *t = st_find(&S,"w");
      CHECK(t && t->verified == 0);
      st_read_raw(&S,"w",buf,0);
      CHECK(t->verified == 1);                   /* first touch verified it */
      CHECK(st_verify_once(&S,t,buf,t->nbytes) == 0);  /* second call: skipped */
      rm_fixture(dir); }

    /* 3. COLI_NO_VERIFY=1: the escape hatch skips even a WRONG digest, which is
     *    what makes it usable for benchmarking rather than a footgun that only
     *    works on healthy containers. */
    { const char *dir="tests/tmp_sum_off";
      write_fixture(dir,1,"0000000000000000000000000000000000000000000000000000000000000000");
      setenv("COLI_NO_VERIFY","1",1);
      shards S; st_init(&S,dir);
      st_read_raw(&S,"w",buf,0);                 /* must not refuse */
      st_tensor *t = st_find(&S,"w");
      CHECK(t->verified == 0);                   /* skipped, not silently marked done */
      unsetenv("COLI_NO_VERIFY");
      rm_fixture(dir); }

    /* 4. wrong digest with verification on: refuses, and says which tensor. */
#ifndef _WIN32
    { const char *dir="tests/tmp_sum_bad";
      write_fixture(dir,1,"1111111111111111111111111111111111111111111111111111111111111111");
      int pfd[2];
      if(pipe(pfd)==0){
          pid_t pid=fork();
          if(pid==0){
              dup2(pfd[1],2); close(pfd[0]); close(pfd[1]);
              unsigned char b2[4096];
              shards S; st_init(&S,dir);
              st_read_raw(&S,"w",b2,0);          /* must exit(1) */
              _exit(42);
          } else if(pid>0){
              close(pfd[1]);
              char msg[1024]; ssize_t n=read(pfd[0],msg,sizeof msg-1); close(pfd[0]);
              if(n<0) n=0; msg[n]=0;
              int st=0; waitpid(pid,&st,0);
              CHECK(WIFEXITED(st) && WEXITSTATUS(st)==1);
              CHECK(strstr(msg,"CHECKSUM MISMATCH") != NULL);
              CHECK(strstr(msg,"w") != NULL);
              CHECK(strstr(msg,"COLI_NO_VERIFY") != NULL);   /* names the escape hatch */
          }
      }
      rm_fixture(dir); }

    /* 5. a malformed digest is caught at container discovery, not on first read
     *    -- otherwise a typo surfaces thousands of tensors later as a bogus
     *    corruption report. */
    { const char *dir="tests/tmp_sum_malformed";
      write_fixture(dir,1,"nothex");
      int pfd[2];
      if(pipe(pfd)==0){
          pid_t pid=fork();
          if(pid==0){
              dup2(pfd[1],2); close(pfd[0]); close(pfd[1]);
              shards S; st_init(&S,dir);        /* must exit(1) during ingest */
              _exit(42);
          } else if(pid>0){
              close(pfd[1]);
              char msg[1024]; ssize_t n=read(pfd[0],msg,sizeof msg-1); close(pfd[0]);
              if(n<0) n=0; msg[n]=0;
              int st=0; waitpid(pid,&st,0);
              CHECK(WIFEXITED(st) && WEXITSTATUS(st)==1);
              CHECK(strstr(msg,"64-char lowercase hex") != NULL);
          }
      }
      rm_fixture(dir); }
#endif

    if(!fails) printf("test_blob_checksum: OK\n");
    return fails ? 1 : 0;
}
