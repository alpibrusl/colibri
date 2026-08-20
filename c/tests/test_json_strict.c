/* json.h fail-closed contract: malformed input returns NULL, not a partial
 * tree.
 *
 * Historically json_parse could not report failure at all: {"a" 1} parsed
 * (missing colon skipped), an unterminated array became a shorter valid
 * array, an unknown token became J_NUM 0, and {}garbage was accepted. Every
 * caller that checked !root (st.h, schema_gbnf, deepseek_v4) was
 * written for a contract the parser did not honor. This gates the contract:
 * every well-formed document still parses (including the trailing-space
 * padding safetensors headers carry), every malformed one is refused.
 */
#include <stdio.h>
#include <string.h>

#include "../json.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int accepts(const char *text) {
    jval *v = json_parse(text, NULL);
    if (!v) return 0;
    json_free(v);
    return 1;
}

int main(void) {
    /* well-formed documents keep parsing */
    CHECK(accepts("{}"));
    CHECK(accepts("  {} \n"));
    CHECK(accepts("{\"a\":1}"));
    CHECK(accepts("[1,-2.5,3e2]"));
    CHECK(accepts("null"));
    CHECK(accepts("true"));
    CHECK(accepts("-2.5e3"));
    CHECK(accepts("{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}      "));
    CHECK(accepts("{\"esc\":\"a\\n\\\"\\u03bb\"}"));
    CHECK(accepts("{\"nested\":{\"a\":[{\"b\":[1,[2,[3]]]}]}}"));

    /* malformed documents are refused, not partially parsed */
    CHECK(!accepts(""));
    CHECK(!accepts("{"));
    CHECK(!accepts("{\"a\" 1}"));            /* the missing-colon skip */
    CHECK(!accepts("{\"a\":1"));             /* unterminated object */
    CHECK(!accepts("[1,2"));                 /* unterminated array */
    CHECK(!accepts("[1 2]"));                /* missing comma */
    CHECK(!accepts("{\"a\":}"));             /* missing value */
    CHECK(!accepts("tru"));                  /* truncated keyword */
    CHECK(!accepts("{}x"));                  /* trailing junk after the root */
    CHECK(!accepts("{\"a\":1}]"));
    CHECK(!accepts("{,}"));
    CHECK(!accepts("[,]"));
    CHECK(!accepts("\"unterminated"));
    CHECK(!accepts("{\"k\":qqq}"));          /* unknown token: was J_NUM 0 */
    CHECK(!accepts("{1:2}"));                /* non-string key */

    /* the recursion bound refuses instead of silently returning J_NULL */
    {
        char bomb[(J_MAX_DEPTH + 8) * 2 + 1];
        int n = J_MAX_DEPTH + 8;
        for (int i = 0; i < n; i++) { bomb[i] = '['; bomb[n + i] = ']'; }
        bomb[2 * n] = 0;
        CHECK(!accepts(bomb));
    }

    /* NULL-tolerant helpers stay NULL-tolerant */
    CHECK(json_get(NULL, "k") == NULL);
    json_free(NULL);

    /* arena_out contract: set to NULL on both success and failure */
    {
        char *arena = (char *)&arena;
        jval *v = json_parse("{}", &arena);
        CHECK(v && arena == NULL);
        json_free(v);
        arena = (char *)&arena;
        CHECK(json_parse("{", &arena) == NULL && arena == NULL);
    }

    puts("json strict: ok (10 accepted, 15 refused, bomb refused)");
    return 0;
}
