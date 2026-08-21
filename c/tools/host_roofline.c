/* Can this machine run DeepSeek V4 Flash usefully? Measure, then decide.
 *
 * Moving a 167 GB checkpoint to a candidate machine to find out is an expensive
 * way to ask. Every term in V4's roofline is either a property of the model
 * (known) or a property of the host (measurable in a minute, with no model
 * present): how fast the disk serves a 13 MB random read, how fast RAM streams,
 * and whether the dense weights fit at all.
 *
 * The constants come from measurements on the real checkpoint, recorded in
 * docs/experiments/deepseek-v4-baseline-2026-08-21.md and its follow-ups.
 *
 *   cc -O2 -pthread tools/host_roofline.c -o host_roofline && ./host_roofline
 *
 * Optionally point it at a directory on the disk you would actually use:
 *   ./host_roofline /mnt/nvme
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* ---- what V4 Flash costs, measured on the real checkpoint ---------------- */
#define V4_DENSE_GB      6.73    /* resident: attention + shared expert, every token */
#define V4_ROUTED_GB     3.45    /* 6 of 256 experts x 43 layers, per token */
#define V4_EXPERT_MB     13.37   /* one routed expert, int4 + block scales */
#define V4_DISK_GB     167.0
#define V4_READ_BYTES 13370000   /* the engine's actual read granularity */

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;}

static double total_ram_gb(void){
#if defined(__APPLE__)
    uint64_t v=0; size_t n=sizeof v;
    if(sysctlbyname("hw.memsize",&v,&n,NULL,0)==0) return (double)v/1e9;
#else
    long p=sysconf(_SC_PHYS_PAGES), z=sysconf(_SC_PAGE_SIZE);
    if(p>0&&z>0) return (double)p*(double)z/1e9;
#endif
    return 0.0;
}

/* ---- disk ---------------------------------------------------------------- */
static int drop_cache_fd(int fd){
#if defined(__APPLE__)
    return fcntl(fd, 48 /* F_NOCACHE */, 1);   /* macOS: bypass the page cache */
#elif defined(POSIX_FADV_DONTNEED)
    return posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#else
    (void)fd; return -1;
#endif
}

/* Read the probe file with the page cache genuinely OUT of the way.
 *
 * The first version of this wrote a 3 GiB file, set F_NOCACHE, and read it
 * back: it reported 23 GB/s on a device that does 5.3. Everything it "read" was
 * still in the page cache from the write, and a flag set afterwards does not
 * evict what is already resident. A disk probe that silently measures memory
 * turns the whole verdict optimistic by 4x, so the bypass has to be real:
 * O_DIRECT on Linux, F_NOCACHE on a freshly-opened descriptor on macOS, and a
 * plausibility check at the end for platforms where neither bites.
 */
#if defined(__linux__)
#ifndef O_DIRECT
#define O_DIRECT 040000
#endif
#endif

#define PROBE_ALIGN 4096
static double disk_random_gbs(const char *dir, double *ms_per_read, int *ok,
                              int *bypassed){
    char path[4096];
    snprintf(path,sizeof path,"%s/.coli_roofline_probe",dir);
    const size_t SPAN = 2ULL<<30;
    *bypassed = 0;

    /* Bypass on the WRITE too. Setting it only on the read descriptor leaves
     * every page resident from the write, and the read then measures memory --
     * which is exactly how this probe first reported 23 GB/s on a 5.3 GB/s
     * device. Pages that never enter the cache cannot be served from it. */
    int wflags = O_RDWR|O_CREAT|O_TRUNC;
#if defined(__linux__)
    wflags |= O_DIRECT;
#endif
    int fd = open(path, wflags, 0600);
#if defined(__linux__)
    if(fd<0) fd = open(path, O_RDWR|O_CREAT|O_TRUNC, 0600);
#else
    if(fd>=0) fcntl(fd, 48 /* F_NOCACHE */, 1);
#endif
    if(fd<0){ *ok=0; return 0; }
    unsigned char *chunk = NULL;
    if(posix_memalign((void**)&chunk, PROBE_ALIGN, 1<<20)!=0){
        close(fd); unlink(path); *ok=0; return 0; }
    memset(chunk,0xA5,1<<20);
    for(size_t w=0; w<SPAN; w+=(1<<20))
        if(write(fd,chunk,1<<20)!=(1<<20)){
            free(chunk); close(fd); unlink(path); *ok=0; return 0; }
    free(chunk);
    fsync(fd);
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   /* evict, then reopen */
#endif
    close(fd);

    int flags = O_RDONLY;
#if defined(__linux__)
    flags |= O_DIRECT;
#endif
    fd = open(path, flags);
#if defined(__linux__)
    if(fd<0) fd = open(path, O_RDONLY);             /* O_DIRECT unsupported (tmpfs, some FUSE) */
    else *bypassed = 1;
#else
    if(fd>=0 && fcntl(fd, 48 /* F_NOCACHE */, 1)==0) *bypassed = 1;
#endif
    if(fd<0){ unlink(path); *ok=0; return 0; }

    size_t rd = (V4_READ_BYTES/PROBE_ALIGN)*PROBE_ALIGN;        /* O_DIRECT wants alignment */
    void *buf=NULL;
    if(posix_memalign(&buf, PROBE_ALIGN, rd)!=0){ close(fd); unlink(path); *ok=0; return 0; }

    const int N = 24;
    size_t stride = ((SPAN - rd)/N/PROBE_ALIGN)*PROBE_ALIGN;
    double t0=now(); size_t got=0;
    for(int i=0;i<N;i++){
        ssize_t r = pread(fd, buf, rd, (off_t)((size_t)i*stride));
        if(r>0) got += (size_t)r;
    }
    double el = now()-t0;
    free(buf); close(fd); unlink(path);
    *ok = got>0 && el>0;
    if(!*ok) return 0;
    *ms_per_read = 1000.0*el/N;
    return (double)got/el/1e9;
}

/* ---- RAM ----------------------------------------------------------------- */
static size_t g_bytes; static unsigned char *g_buf; static int g_nt;
static void *reader(void *arg){
    long id=(long)arg; size_t per=g_bytes/g_nt;
    const unsigned long long *p=(const unsigned long long*)(g_buf+per*id);
    volatile unsigned long long acc=0;
    for(size_t i=0;i<per/sizeof(*p);i+=8) acc+=p[i];   /* one per cache line */
    (void)acc; return NULL;
}
static double ram_gbs(int threads){
    g_nt=threads; double best=0;
    for(int r=0;r<3;r++){
        pthread_t t[64]; double t0=now();
        for(long i=0;i<threads;i++) pthread_create(&t[i],NULL,reader,(void*)i);
        for(int i=0;i<threads;i++) pthread_join(t[i],NULL);
        double el=now()-t0, g=(double)g_bytes/el/1e9;
        if(g>best) best=g;
    }
    return best;
}

int main(int argc,char**argv){
    const char *dir = argc>1 ? argv[1] : ".";
    double ram = total_ram_gb();
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);

    printf("host: %.1f GB RAM, %ld logical CPUs, probing disk at %s\n\n", ram, cpus, dir);

    int ok=0, bypassed=0; double ms=0, disk = disk_random_gbs(dir,&ms,&ok,&bypassed);
    if(!ok){ printf("disk probe failed (need ~3 GiB free and write access in %s)\n",dir); return 1; }
    printf("disk   random %.1f MB reads : %6.2f GB/s   %6.2f ms per read%s\n",
           V4_READ_BYTES/1e6, disk, ms, bypassed ? "" : "   [CACHE NOT BYPASSED]");

    g_bytes = 1ULL<<30; g_buf = malloc(g_bytes);
    if(!g_buf){ printf("cannot allocate 1 GiB for the RAM probe\n"); return 1; }
    memset(g_buf,1,g_bytes);
    double r1 = ram_gbs(1);
    int maxt = cpus>16?16:(int)cpus; if(maxt<1) maxt=1;
    double rn = ram_gbs(maxt);
    printf("memory streaming read      : %6.2f GB/s (1 thread)  %6.2f GB/s (%d)\n\n",
           r1, rn, maxt);

    /* ---- the verdict ---- */
    printf("DeepSeek V4 Flash needs %.0f GB on disk and %.2f GB of RAM-resident\n"
           "dense weights that every token reads.\n\n", V4_DISK_GB, V4_DENSE_GB);

    double headroom = 2.0;                      /* OS, KV cache, activations */
    double cache = ram - V4_DENSE_GB - headroom;
    double stream_gb;                           /* bytes per token off the disk */
    const char *regime;
    if(cache < 0.5){
        /* Dense does not fit, so it streams too and nothing is cached. */
        stream_gb = V4_DENSE_GB + V4_ROUTED_GB;
        regime = "dense weights DO NOT FIT: every weight streams, every token";
        cache = 0;
    } else {
        /* Hit rate against cache size, from three points measured on the
         * reference host (17.7 GiB -> 66.2%, 23.6 -> 72.4%, 39.6 -> 77.5%).
         * A log fit through them, clamped: it is an interpolation between real
         * measurements in that range and an EXTRAPOLATION outside it -- treat a
         * number far from ~20-40 GiB as an order of magnitude, not a figure. */
        double hit = 25.85 + 14.04*log(cache>0.5?cache:0.5);
        if(hit<0) hit=0; if(hit>95) hit=95;
        stream_gb = (1.0-hit/100.0)*V4_ROUTED_GB;
        printf("expert cache available     : %6.2f GB  -> ~%.0f%% hit rate (extrapolated)\n",
               cache, hit);
        regime = "dense weights fit; only routed-expert misses stream";
    }
    printf("regime : %s\n", regime);

    double t_disk = stream_gb/disk;
    double t_ram  = (V4_DENSE_GB+V4_ROUTED_GB)/rn;
    double t = t_disk>t_ram ? t_disk : t_ram;
    printf("per token : %.2f GB from disk (%.2f s) | %.2f GB through RAM (%.2f s)\n",
           stream_gb, t_disk, V4_DENSE_GB+V4_ROUTED_GB, t_ram);
    /* An absolute bound, not a ratio: PCIe 5.0 x4 tops out near 14 GB/s, so
     * anything above that is the page cache answering, whatever the flags say.
     * The ratio test this replaces let a 23 GB/s reading through. */
    if(disk > 14.0)
        printf("\nWARNING: the disk probe reports %.2f GB/s. No NVMe reaches that -- the\n"
               "  reads were served from the page cache despite the bypass, so every figure\n"
               "  below is optimistic. Point the tool at a directory on the target disk, or\n"
               "  read the RAM-bound line only.\n", disk);
    printf("\nROOFLINE  %.3f tok/s   (%.2f s per token, %s-bound)\n",
           1.0/t, t, t_disk>t_ram?"disk":"memory");
    printf("          a 128-token reply takes at least %.1f minutes\n", 128*t/60.0);
    printf("\nThis is a CEILING with perfect overlap and free compute. The reference\n"
           "host measured 1.4 tok/s against its own 5.55 tok/s roofline -- about 25%%.\n"
           "Scale accordingly, and note the roofline says nothing about whether the\n"
           "167 GB fits on the disk you probed.\n");
    return 0;
}
