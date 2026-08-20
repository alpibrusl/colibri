# Packs do not rescue content addressing (lex-moe#50)

**Date** 2026-08-20 · **Issues** [lex-moe#50](https://github.com/alpibrusl/lex-moe/issues/50), [#14](https://github.com/alpibrusl/colibri/issues/14) · **Follows** #55

`castore-io-cost-2026-08-20.md` measured per-tensor content addressing at **3.6×**
the reads of a co-activation-clustered container. The proposed fix was to
content-address **packs** of co-activating experts. lex-moe#50 stated the gate
before any format work, so the answer could be a clean negative:

> a pack size exists that gets within ~1.2× of the clustered layout's read count
> while keeping a realistic fine-tune's delta under ~2× the per-tensor baseline —
> or no such point exists, and CA is for distribution rather than streaming.

**No such point exists.** The two requirements need pack sizes that do not overlap.

## The curve

Corpus A (915 forwards, OLMoE, mean expert 6.31 MB). Clustered baseline 70,026
reads; per-tensor CA 253,754.

| pack | reads | vs clustered | byte amp | delta@1% | delta@3% | delta@10% |
|---|---|---|---|---|---|---|
| 1 | 126,877 | 1.81× | 1.00× | 1.00× | 1.00× | 1.00× |
| 2 | 97,371 | 1.39× | 1.53× | 2.00× | 2.00× | 1.96× |
| **4** | 82,795 | **1.18×** ✓ | 2.42× | 4.00× | **3.88×** ✗ | 3.62× |
| 8 | 75,799 | 1.08× ✓ | 3.80× | 8.00× | 7.50× ✗ | 5.92× |
| 16 | 72,337 | 1.03× ✓ | 5.65× | 16.00× | 14.00× ✗ | 8.33× |
| 32 | 70,674 | 1.01× ✓ | 7.00× | 32.00× | 25.00× ✗ | 10.67× |
| 64 | 70,026 | 1.00× ✓ | 7.38× | 64.00× | 32.00× ✗ | 10.67× |

Corpus B (code/mathematics) is the same shape: reads 1.70× → 1.00×, identical
delta column, same verdict.

**The read gate needs P ≥ 4. The delta gate needs P ≤ 2. The windows do not
meet.**

## Why, structurally

Delta scales roughly **linearly** in pack size — a changed expert dirties its
whole pack, so re-storing costs `P ×` what the change was — while reads improve
only **sub-linearly**, because clustering has already captured most of the
locality by P = 4. One cost grows faster than the other benefit, and there is no
crossing point.

The cruelest detail is which case suffers most. `delta@1%` is *worse* than
`delta@10%` at every pack size (8.00× against 5.92× at P = 8): a large fine-tune
dirties packs it would have paid for anyway, while a small one pays for
neighbours it never touched. **Packing damages precisely the case content
addressing was sold on** — "a fine-tune that touched 3% of experts costs GBs, not
another full copy" is the claim in colibri#14, and at P = 4 that fine-tune costs
3.88× what per-tensor addressing would.

## A third cost the proposal did not name

A pack is **one content-addressed object**, so verifying it means reading all of
it. A reader that wants the address to remain the checksum must read whole packs:
the `byte amp` column is what that costs, **2.42× the bytes at P = 4 and 3.80× at
P = 8**.

A reader can instead `pread` a sub-range and get the bytes back — that is what the
`reads` column assumes — but it then cannot check that range against the pack's
hash without reading the rest. So packs force a choice between *integrity by
construction* and *not reading bytes you do not need*, which is exactly the
property that made content addressing attractive for this engine.

## Verdict

**Close lex-moe#50 as not-planned.** Packs do not rescue CA on the streaming
path; they trade a read problem for a delta problem and add a byte-amplification
problem on the way.

This is not an argument against content addressing. Its wins are real and **none
of them are on the read path**:

- dedupe across models and quantizations
- fine-tunes as deltas — *at per-tensor granularity, where the gate says it works*
- integrity where the address is the checksum (colibri#53)
- a Nix-style network cache

The honest conclusion is that **content addressing is for distribution and
safetensors for streaming**, and the two should be said to be for different jobs
rather than merged badly. colibrì's expert path already refuses a CA container
(`coli_expert_path_supports_ca`, #54); on this evidence that refusal is
permanent, not provisional, and its message should say so.

## What stays useful

`castore.h` reads a CA store today, and `st.h`'s ordinary readers work against
one (#54) — so a CA container remains a first-class way to *distribute*,
*verify* and *dedupe* a model, and to load dense/resident tensors. Only routed
expert streaming is excluded, and only because it was measured.

## Reproducing

```sh
python3 tools/pack_size_curve.py --trace trace.txt --perm perm.json --container SNAP
python3 tools/pack_size_curve.py --selftest
```
