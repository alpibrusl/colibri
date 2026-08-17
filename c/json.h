/* Parser JSON minimale, header-only. Serve per:
 *  - l'header dei file safetensors (un grande oggetto nome->{dtype,shape,data_offsets})
 *  - ref.json (per leggere prompt_ids / full_ids)
 * Non e' completo (niente notazione esotica) ma copre cio' che serve.
 *
 * FAIL-CLOSED: json_parse ritorna NULL su input malformato (delimitatore
 * mancante, oggetto/array non terminato, token sconosciuto, spazzatura dopo
 * la radice, annidamento oltre J_MAX_DEPTH) e su OOM. Storicamente ritornava
 * un albero PARZIALE senza alcun segnale d'errore -- {"a" 1} parsava, un
 * array non chiuso diventava un array valido piu' corto, un token ignoto
 * diventava J_NUM 0 -- e ogni chiamante doveva accorgersene da solo: con
 * input non fidati (header safetensors da mirror, tokenizer.json, schema del
 * client) il parse silenziosamente parziale e' esattamente il comportamento
 * sbagliato. I chiamanti che gia' controllavano !root (st.h, cfse_pack,
 * schema_gbnf, deepseek_v4) erano scritti per questo contratto; ora e' vero. */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminata, allocazione propria) */
    /* array: figli in [0..len); oggetto: chiavi[] e figli[] in parallelo */
    struct jval **kids;
    char        **keys;    /* solo per J_OBJ */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* storico; sempre NULL (vedi j_dup) */
    size_t      acap, aoff;
    int         depth;     /* annidamento corrente: bound contro lo stack-overflow
                            * da JSON malevolo tipo [[[[...]]]] (discesa ricorsiva) */
} jparser;

/* tetto di annidamento: gli header safetensors / config sono piatti (profondita'
 * ~3). 1024 e' larghissimo per input legittimi e ben sotto il limite di stack. */
#define J_MAX_DEPTH 1024

static void json_free(jval *v);

static char *j_dup(const char *b, int n) {
    /* ogni stringa ha la sua allocazione: un'arena con realloc sposterebbe il
     * buffer invalidando i puntatori gia' emessi (use-after-free). */
    char *d = (char *)malloc((size_t)n + 1);
    if (!d) return NULL;                       /* OOM: propagata, non NULL-deref */
    memcpy(d, b, (size_t)n); d[n] = 0;
    return d;
}

static void j_ws(jparser *p) { while (*p->s && isspace((unsigned char)*p->s)) p->s++; }

static jval *j_new(jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    if (!v) return NULL;
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static char *j_parse_str_raw(jparser *p) {
    /* SEC (GHSA-2qrj): fail closed if not actually at a quote. The old comment
     * "assume *p->s == '\"'" was violated on the object-key path, and the
     * unconditional p->s++ would step past the buffer's NUL terminator and scan
     * adjacent heap (OOB read leaking into tensor names). */
    if (*p->s != '"') return NULL;
    p->s++;
    /* buffer su heap che CRESCE: niente troncamento silenzioso a 64KB (le stringhe
     * lunghe di tokenizer.json/config venivano tagliate) e niente 64KB di stack. */
    size_t cap = 64, n = 0; char *tmp = (char *)malloc(cap);
    if (!tmp) return NULL;
    #define J_PUT(ch) do{ if (n + 1 >= cap) { cap *= 2; char *g = (char *)realloc(tmp, cap); \
        if (!g) { free(tmp); return NULL; } tmp = g; } tmp[n++] = (char)(ch); }while(0)
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (c == '\\' && *p->s) {
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {  /* \uXXXX -> codepoint UTF-8 (con coppie surrogate) */
                    if (!p->s[0]||!p->s[1]||!p->s[2]||!p->s[3]) { c='?'; break; }   /* \u troncato: non leggere oltre il NUL */
                    unsigned cp = (unsigned)strtoul((char[]){p->s[0],p->s[1],p->s[2],p->s[3],0}, NULL, 16);
                    p->s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p->s[0]=='\\' && p->s[1]=='u'
                        && p->s[2] && p->s[3] && p->s[4] && p->s[5]) {
                        unsigned lo = (unsigned)strtoul((char[]){p->s[2],p->s[3],p->s[4],p->s[5],0}, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp-0xD800)<<10) + (lo-0xDC00); p->s += 6; }
                    }
                    if (cp < 0x80) { J_PUT(cp); }
                    else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    continue;
                }
                default: c = e; break;
            }
        }
        J_PUT(c);
    }
    #undef J_PUT
    /* stringa non terminata (EOF prima della chiusura): errore, non un valore */
    if (*p->s != '"') { free(tmp); return NULL; }
    p->s++;
    char *out = j_dup(tmp, (int)n); free(tmp);
    return out;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    char c = *p->s;
    if (c == '"') {
        char *str = j_parse_str_raw(p);
        if (!str) return NULL;
        jval *v = j_new(J_STR);
        if (!v) { free(str); return NULL; }
        v->str = str; return v;
    }
    if (c == '{') {
        if (++p->depth > J_MAX_DEPTH) { p->depth--; return NULL; }
        p->s++; jval *v = j_new(J_OBJ);
        if (!v) { p->depth--; return NULL; }
        int cap = 8;
        v->keys = (char **)malloc(cap * sizeof(char*));
        v->kids = (jval **)malloc(cap * sizeof(jval*));
        if (!v->keys || !v->kids) { json_free(v); p->depth--; return NULL; }
        j_ws(p);
        if (*p->s == '}') { p->s++; p->depth--; return v; }
        for (;;) {
            j_ws(p);
            /* SEC (GHSA-2qrj): object key must be a quoted string */
            if (*p->s != '"') goto obj_fail;
            char *key = j_parse_str_raw(p);
            if (!key) goto obj_fail;
            j_ws(p);
            if (*p->s != ':') { free(key); goto obj_fail; }   /* {"a" 1} non parsa piu' */
            p->s++;
            jval *val = j_parse_val(p);
            if (!val) { free(key); goto obj_fail; }
            if (v->len == cap) {
                cap *= 2;
                char **nk = (char **)realloc(v->keys, cap*sizeof(char*));
                if (nk) v->keys = nk;
                jval **nv = (jval **)realloc(v->kids, cap*sizeof(jval*));
                if (nv) v->kids = nv;
                if (!nk || !nv) { free(key); json_free(val); goto obj_fail; }
            }
            v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == '}') { p->s++; break; }
            goto obj_fail;                     /* spazzatura o EOF dentro l'oggetto */
        }
        p->depth--;
        return v;
obj_fail:
        json_free(v); p->depth--; return NULL;
    }
    if (c == '[') {
        if (++p->depth > J_MAX_DEPTH) { p->depth--; return NULL; }
        p->s++; jval *v = j_new(J_ARR);
        if (!v) { p->depth--; return NULL; }
        int cap = 8; v->kids = (jval **)malloc(cap * sizeof(jval*));
        if (!v->kids) { json_free(v); p->depth--; return NULL; }
        j_ws(p);
        if (*p->s == ']') { p->s++; p->depth--; return v; }
        for (;;) {
            jval *val = j_parse_val(p);
            if (!val) goto arr_fail;
            if (v->len == cap) {
                cap *= 2;
                jval **nv = (jval **)realloc(v->kids, cap*sizeof(jval*));
                if (!nv) { json_free(val); goto arr_fail; }
                v->kids = nv;
            }
            v->kids[v->len++] = val;
            j_ws(p);
            if (*p->s == ',') { p->s++; continue; }
            if (*p->s == ']') { p->s++; break; }
            goto arr_fail;                     /* spazzatura o EOF dentro l'array */
        }
        p->depth--;
        return v;
arr_fail:
        json_free(v); p->depth--; return NULL;
    }
    if (c == 't' && !strncmp(p->s, "true", 4))  { p->s += 4; jval *v = j_new(J_BOOL); if (v) v->boolean = 1; return v; }
    if (c == 'f' && !strncmp(p->s, "false", 5)) { p->s += 5; jval *v = j_new(J_BOOL); return v; }
    if (c == 'n' && !strncmp(p->s, "null", 4))  { p->s += 4; return j_new(J_NULL); }
    /* numero: strtod deve consumare almeno un carattere -- prima, qualsiasi
     * token sconosciuto diventava silenziosamente J_NUM 0 */
    { char *end; double d = strtod(p->s, &end);
      if (end == p->s) return NULL;
      p->s = end; jval *v = j_new(J_NUM);
      if (v) v->num = d;
      return v; }
}

/* API */
static jval *json_parse(const char *text, char **arena_out) {
    jparser p = { text, NULL, 0, 0, 0 };
    if (arena_out) *arena_out = NULL;
    jval *v = j_parse_val(&p);
    if (v) {
        /* la radice deve esaurire l'input (spazi finali ammessi: gli header
         * safetensors sono paddati con 0x20). "{}garbage" era accettato. */
        j_ws(&p);
        if (*p.s) { json_free(v); v = NULL; }
    }
    return v;
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

static void json_free(jval *v) {
    if (!v) return;
    if (v->t == J_ARR || v->t == J_OBJ) {
        for (int i = 0; i < v->len; i++) {
            json_free(v->kids[i]);
            if (v->t == J_OBJ) free(v->keys[i]);
        }
        free(v->kids);
        free(v->keys);
    } else if (v->t == J_STR) {
        free(v->str);
    }
    free(v);
}

#endif
