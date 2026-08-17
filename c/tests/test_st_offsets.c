/* Safetensors data_offsets validation, including the exit(1) refusal paths.
 *
 * Companion to test_st_shape.c, same subprocess pattern (st_init terminates
 * the process on a hostile container, so each case runs in a child). What
 * this gates is the SEC fix that validates data_offsets BEFORE the
 * double->int64 cast: on NaN/inf/>=2^63 the cast itself is UB, so the old
 * "a0 < 0" check ran on an indeterminate value, and with b0 near INT64_MAX
 * the bounds check "data_start + b0 > fsz" overflowed (UB, wraps negative)
 * and PASSED. The 9223372036854775807 case below is that exact wrap.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct {
    const char *off;    /* data_offsets JSON text */
    int accept;
} off_case;

/* payload is always 16 bytes of U8 data after the header */
static const off_case CASES[] = {
    {"[0,16]", 1},
    {"[16,0]", 0},                      /* reversed: b < a */
    {"[0,17]", 0},                      /* one byte past the payload */
    {"[-8,8]", 0},                      /* negative start */
    {"[0,nan]", 0},                     /* strtod parses "nan": cast would be UB */
    {"[0,1e300]", 0},                   /* far beyond int64 */
    {"[0,9223372036854775807]", 0},     /* the data_start+b0 signed-overflow wrap */
    {"[0,15.5]", 0},                    /* fractional */
    {"[0,\"16\"]", 0},                  /* wrong type */
};

static int write_case(const char *dir, const off_case *test) {
    char path[512];
    char header[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    int header_bytes = snprintf(
        header, sizeof(header),
        "{\"t\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":%s}}",
        test->off);
    if (header_bytes < 0 || (size_t)header_bytes >= sizeof(header)) return -1;

    FILE *file = fopen(path, "wb");
    if (!file) return -1;
    uint64_t hlen = (uint64_t)header_bytes;
    unsigned char zeros[16] = {0};
    int failed =
        fwrite(&hlen, 8, 1, file) != 1 ||
        fwrite(header, 1, (size_t)header_bytes, file) != (size_t)header_bytes ||
        fwrite(zeros, 1, sizeof(zeros), file) != sizeof(zeros);
    if (fclose(file) != 0) failed = 1;
    return failed ? -1 : 0;
}

static int child_case(int index, const char *dir) {
    if (index < 0 || index >= (int)(sizeof(CASES) / sizeof(CASES[0]))) return 90;
    const off_case *test = &CASES[index];
    shards S;
    st_init(&S, dir);
    /* a refusing case reaching here means st_init incorrectly accepted it */
    if (!test->accept) return 91;
    st_tensor *tensor = st_find(&S, "t");
    if (!tensor || tensor->nbytes != 16) return 92;
    return 0;
}

static int subprocess_exit_code(const char *self, int index, const char *dir) {
    char command[1536];
    char error_path[512];
    snprintf(error_path, sizeof(error_path), "%s/stderr.txt", dir);
#ifdef _WIN32
    int n = snprintf(command, sizeof(command),
                     "call \"%s\" --off-child %d \"%s\" 2>\"%s\"",
                     self, index, dir, error_path);
#else
    int n = snprintf(command, sizeof(command),
                     "\"%s\" --off-child %d \"%s\" 2>\"%s\"",
                     self, index, dir, error_path);
#endif
    if (n < 0 || (size_t)n >= sizeof(command)) return -1;
    int status = system(command);
#ifdef _WIN32
    return status;
#else
    if (status < 0 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
#endif
}

static void remove_case(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    remove(path);
    snprintf(path, sizeof(path), "%s/stderr.txt", dir);
    remove(path);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--off-child"))
        return child_case(atoi(argv[2]), argv[3]);

    CHECK(argc >= 1 && argv[0] && *argv[0]);
    for (int i = 0; i < (int)(sizeof(CASES) / sizeof(CASES[0])); i++) {
        char dir[] = "test_st_offsets_XXXXXX";
        CHECK(mkdtemp(dir) != NULL);
        CHECK(write_case(dir, &CASES[i]) == 0);
        int code = subprocess_exit_code(argv[0], i, dir);
        char error_path[512];
        char error[1024] = {0};
        snprintf(error_path, sizeof(error_path), "%s/stderr.txt", dir);
        FILE *error_file = fopen(error_path, "rb");
        CHECK(error_file != NULL);
        size_t error_bytes = fread(error, 1, sizeof(error) - 1, error_file);
        CHECK(!ferror(error_file));
        fclose(error_file);
        error[error_bytes] = 0;
        remove_case(dir);
        if (CASES[i].accept) {
            CHECK(code == 0);
            CHECK(error_bytes == 0);
        } else {
            CHECK(code == 1);
            CHECK(strstr(error, "data_offsets") != NULL);
        }
    }
    printf("OK st data_offsets: %d cases (accept + refusal, incl. the int64 overflow wrap)\n",
           (int)(sizeof(CASES) / sizeof(CASES[0])));
    return 0;
}
