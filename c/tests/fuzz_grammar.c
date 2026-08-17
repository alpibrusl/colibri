/* Structured-mutation fuzz for grammar.h: gr_parse on hostile GBNF, then
 * the PDA walker (gr_state_init / gr_accept / gr_forced / gr_admissible)
 * driven with hostile bytes on every grammar that parses.
 *
 * The server hands gr_parse a client-supplied string (openai_server.py
 * caps it at 1 MiB but does not otherwise validate it), so this is an
 * attacker-facing parser. The walker is fail-safe by design -- overflow
 * kills the draft, never corrupts output -- so what ASan+UBSan assert
 * here is that no input reaches OOB state first: the fixed-size rule /
 * stack / frame arrays (GR_MAX_RULES/STACKS/DEPTH) are exactly the kind
 * of bounds a mutated grammar goes hunting for.
 *
 * Seeded and bounded (6 base grammars x 700 mutants). Build and run via
 * `make fuzz-grammar` (ASan+UBSan, non-recoverable). NOT part of
 * TEST_BINS: the sanitizer flags differ from the normal test build.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fuzz_mut.h"
#include "../grammar.h"

static const char *BASES[] = {
    "root ::= \"{\\\"id\\\":\" [0-9]+ \"}\"",

    "root ::= \"a\" (\"b\" | \"c\" | inner)* \"d\"\n"
    "inner ::= \"x\" [a-z0-9]? \"y\"",

    "root ::= obj\n"
    "obj ::= \"{\" (pair (\",\" pair)*)? \"}\"\n"
    "pair ::= str \":\" val\n"
    "val ::= str | num | obj\n"
    "str ::= \"\\\"\" [a-zA-Z]* \"\\\"\"\n"
    "num ::= \"-\"? [0-9]+",

    "root ::= a\na ::= b\nb ::= c\nc ::= \"deep\" a?",

    "root ::= [^\\n\"]+ \"\\n\"",

    "root ::= \"tool\" ( \"_call\" | \"_result\" ) \":\" [ -~]*",
};

/* one static Grammar: gr_parse memsets it on entry, and gr_free after
 * every attempt (success or refusal) releases whatever alternates were
 * allocated before the refusal point */
static Grammar G;

static void drive(const uint8_t *bytes, size_t n) {
    GrState S;
    char forced[512];
    unsigned char mask[32];
    int can_end;
    gr_state_init(&S, &G);
    (void)gr_forced(&S, forced, (int)sizeof forced);
    (void)gr_admissible(&S, mask, &can_end);
    size_t steps = n < 512 ? n : 512;
    for (size_t i = 0; i < steps && S.alive; i++) {
        if (gr_accept(&S, bytes[i]) != 1) break;
        if ((i & 15) == 0) {
            (void)gr_forced(&S, forced, (int)sizeof forced);
            (void)gr_admissible(&S, mask, &can_end);
        }
    }
}

int main(void) {
    long cases = 0, parsed = 0, refused = 0;
    const size_t nbases = sizeof(BASES) / sizeof(BASES[0]);
    for (size_t b = 0; b < nbases; b++) {
        size_t blen = strlen(BASES[b]);
        for (int it = 0; it < 700; it++) {
            size_t mlen;
            uint8_t *m = fm_mutate((const uint8_t *)BASES[b], blen, &mlen);
            if (!m) continue;
            cases++;
            if (gr_parse(&G, (const char *)m) == 0) {
                parsed++;
                /* walk the accepted grammar with hostile bytes: another
                 * mutant of a different base, so the walker sees input
                 * the grammar was never written for */
                size_t dlen;
                uint8_t *d = fm_mutate((const uint8_t *)BASES[(b + 1) % nbases],
                                       strlen(BASES[(b + 1) % nbases]), &dlen);
                if (d) { drive(d, dlen); free(d); }
            } else {
                refused++;
                /* the error string must be a bounded, terminated C string */
                if (strlen(G.err) >= sizeof G.err) {
                    fprintf(stderr, "unterminated G.err\n");
                    return 2;
                }
            }
            gr_free(&G);
            free(m);
        }
    }
    printf("fuzz_grammar: %ld cases, %ld parsed+walked, %ld refused -- clean\n",
           cases, parsed, refused);
    return 0;
}
