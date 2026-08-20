/* test_bytes_per_token.c -- the BYTES telemetry line's arithmetic (#37).
 *
 * #37 asks for "bytes fetched per token accounting per session and in
 * aggregate", and names bytes/token as the primary metric precisely because
 * tok/s mixes in the separate multi-row matmul problem. The two ratios are
 * split out of mux_done so they can be pinned without a Model, a KV slot or a
 * checkpoint -- mux_done itself persists usage, rebinds KV and writes the
 * protocol block, none of which this contract depends on.
 *
 * The zero-emitted case is not hypothetical: a turn ends with emitted==0
 * whenever the client sends STOP before the first token, the model stops on its
 * first sample, or a request is refused after its window opened. Observed
 * directly on a fixture whose greedy argmax terminates immediately.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }
static int close_to(double a, double b){ double d=a-b; if(d<0) d=-d; return d < 1e-9; }

int main(void){
    int bad=0;
    double per_turn, per_all;

    /* ordinary turn: 60 MB over 20 tokens, engine at 300 MB over 100 */
    bytes_ratios(60000000, 20, 300000000, 100, &per_turn, &per_all);
    if(!close_to(per_turn,3000000.0)) bad|=fail("per-turn ratio wrong");
    if(!close_to(per_all,3000000.0))  bad|=fail("aggregate ratio wrong");

    /* a turn that emitted nothing reports 0, does not divide by zero, and does
     * not leak the aggregate into the per-turn slot */
    per_turn=per_all=-1.0;
    bytes_ratios(41943040, 0, 300000000, 100, &per_turn, &per_all);
    if(!close_to(per_turn,0.0)) bad|=fail("emitted==0 must report 0 bytes/token");
    if(!close_to(per_all,3000000.0)) bad|=fail("aggregate must survive an empty turn");

    /* first turn of a run: nothing emitted anywhere yet, both denominators 0 */
    per_turn=per_all=-1.0;
    bytes_ratios(0, 0, 0, 0, &per_turn, &per_all);
    if(!close_to(per_turn,0.0)||!close_to(per_all,0.0)) bad|=fail("cold start must be 0/0 -> 0");

    /* aggregate is engine-wide: the same total spread over more tokens is a
     * LOWER bytes/token, which is the direction #37 predicts as sessions are
     * added and a shared fetch stops being paid twice */
    double one_session, many_sessions, ignored;
    bytes_ratios(0,0, 800000000, 100, &ignored, &one_session);
    bytes_ratios(0,0, 800000000, 400, &ignored, &many_sessions);
    if(!(many_sessions < one_session))
        bad|=fail("aggregate must fall when more tokens share the same bytes");
    if(!close_to(one_session,8000000.0)||!close_to(many_sessions,2000000.0))
        bad|=fail("aggregate arithmetic wrong");

    /* a negative window is clamped by the caller, but the ratio must not invent
     * a sign if one ever reaches it */
    bytes_ratios(0, 5, 0, 5, &per_turn, &per_all);
    if(!close_to(per_turn,0.0)||!close_to(per_all,0.0)) bad|=fail("zero bytes must be 0/token");

    if(!bad) printf("test_bytes_per_token: OK\n");
    return bad;
}
