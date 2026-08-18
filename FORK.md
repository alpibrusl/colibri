# About this fork

This is `alpibrusl/colibri`, a development fork of
[JustVugg/colibri](https://github.com/JustVugg/colibri). The upstream
project — its website, Discord, releases, and documentation — is the
original authors' work; go there for the engine itself. This fork exists
to pursue a specific program on top of it, driven by a full six-subsystem
code audit of upstream (2026-08-17): harden the trust boundaries, break
the monolith along the seams upstream's own design already implies, and
graft on the provenance layers (content addressing, attestations,
deterministic replay) prototyped in
[alpibrusl/lex-moe](https://github.com/alpibrusl/lex-moe).

## Branch model

| branch | role |
|---|---|
| `main` | pristine mirror of upstream `main` — never committed to |
| `next` | this fork's development trunk; all PRs land here |

Upstream is merged into `next` regularly (merge, not rebase). The
divergence-budget rule: changes upstream would plausibly accept are
candidates to offer upstream, so the fork carries only what needs the
freedom to diverge.

## What has diverged (by phase)

**Phase 0 — instrumentation** (done): a ThreadSanitizer CI gate
(`make test-tsan`) and structured-mutation fuzzers for the three
untrusted-input parsers `fuzz-rans` never covered (`json.h`,
`grammar.h`, `st.h`'s header parse). #4, #5.

**Phase 1 — trust-boundary hardening** (done):

- `st.h` validates `data_offsets` before the double→int64 cast; the
  bounds check can no longer signed-overflow. `cfse_pack` gets the same
  discipline; `kv_persist` checks every write so ENOSPC can't persist a
  corrupt record count. #6
- `json.h` fails closed: malformed input returns NULL instead of a
  silent partial tree (19-call-site audit). #7
- **Control-token injection fixed across all five architectures**: user
  content containing special-token text (`<|system|>`, …) can no longer
  become real control tokens. `tok_encode_raw` /
  `tok_encode_guarded` in `tok.h`, the `CGUARD1` wire format
  (`guard_wire.h`), and gateway-side `guard()` spans on every
  client-derived splice. Operational notes: legacy flat wire payloads
  still work; `COLI_GUARD=0` on the gateway disables the header for
  older engine binaries. #8
- The main/pilot shared cache metadata uses relaxed-atomic accessors
  (formal C11 races removed); the residency scan and eviction-guard
  logic exist once instead of five and two hand-kept copies. #9

**Phase 2 — structural surgery** (in progress): the tier cache is being
extracted behind an owned `TierCache` module (`tier_cache.h`) with
engine-free concurrency tests (#10); a `coli_gpu_ops` vtable to collapse
the per-backend `#ifdef` weave (#11); de-amalgamating `deepseek_v4.c`
and extracting the per-engine boilerplate into shared units (#12).

**Phase 3 — provenance layers** (planned): mandatory on-disk format tag
+ blob checksums (#13), a content-addressed expert container (#14), and
signed parity attestations + deterministic replay (#15).

The full rationale for each item lives in the linked issues, each of
which quotes the audit finding it answers.

## Reporting problems

Problems in the diverged surface (anything above) belong in
[this fork's issues](https://github.com/alpibrusl/colibri/issues).
Problems reproducible on upstream belong in
[upstream's tracker](https://github.com/JustVugg/colibri/issues) —
please report them there even if we fix them here first.
