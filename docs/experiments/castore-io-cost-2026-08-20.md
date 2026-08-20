# What content addressing costs the expert path (#14, #36)

**Date** 2026-08-20 · **Issues** [#14](https://github.com/alpibrusl/colibri/issues/14), [#36](https://github.com/alpibrusl/colibri/issues/36) · **Follows** #53, #54

#54 wired a content-addressed store into `shards` and made the expert path
*refuse* one, deferring the decision to a measurement. This is that measurement,
and the answer is unambiguous: **a CA store at per-tensor granularity is worse on
the streaming path than the container colibrì already had.**

| layout | reads | bytes | mean read | vs relaid |
|---|---|---|---|---|
| original safetensors | 213,081 | 800.32 GB | 3.76 MB | 3.04× fewer |
| **relaid** (#43) | **70,187** | 800.32 GB | **11.40 MB** | — |
| **castore** | **253,754** | 800.32 GB | **3.15 MB** | **3.62× more** |

Cross-domain (corpus B): 168,908 vs 49,597 — **3.41× more**, and **1.18×** the
original. Bytes moved are identical in every row; only the number of reads
differs.

## Why

#36 made the container faster by putting co-activating experts **adjacent**, so
the K experts a layer routes to collapse into fewer, longer reads. Content
addressing makes adjacency **impossible by construction**: a blob's location is
its hash, so two tensors that fire together are as likely to be far apart as any
other pair, and no read can ever merge with another.

The two ideas pull in opposite directions, and this is the price.

CA is worse than even the *unoptimised* original for a reason worth noting: in
the merged safetensors container some tensors do happen to sit adjacent — all
the `.qs` scale arrays land in one region, all the `merged_weight` blobs in
another — so a routed set picks up incidental coalescing. In a CA store that
accident cannot happen.

## Method

`tools/expert_relayout.py --measure` already scored two containers against a
trace using their real header offsets. Modelling CA needed no store on disk and
no new scorer: in a CA store every tensor is its own file, so giving each tensor
its **own synthetic shard id** makes `count_reads`' merge condition (same shard,
touching ranges) unable to fire. That is not an approximation of the layout — it
is exactly what the layout does.

## What this does and does not say

**Does:** per-tensor content addressing costs 3.4–3.6× the reads of a
co-activation-clustered container on the routed-expert path, at identical bytes.
#44 measured the relaid layout as 13% faster cold for a 3.2× read *reduction*;
moving 3.6× in the other direction is materially worse, not a wash.

**Does not** condemn content addressing. Its wins — dedupe across models and
quantizations, fine-tunes as deltas, integrity where the address *is* the
checksum (#53), a Nix-style network cache — are real and none of them are on the
read path. What the measurement condemns is *per-tensor granularity* for
streamed experts.

**Does not** measure dense/resident tensors. Those load once at startup, where
read count barely matters; this is a streaming-path result.

## The resolution it points to

**Address packs, not tensors.** Group co-activating experts into one
content-addressed blob and let the manifest map `(layer, expert, tensor)` to an
offset *inside* a pack. Adjacency is restored — #43's permutation decides pack
membership — while the pack itself stays content-addressed, so dedupe,
integrity and delta all survive at pack granularity. A fine-tune that touched 3%
of experts still costs the packs it touched rather than the whole model.

That is a format change on the lex-moe side (`moe-store` writes it, this reads
it), not something the C reader can decide alone. Until it exists,
`coli_expert_path_supports_ca` refusing a CA container on the streaming path is
the correct behaviour, and #54's refusal message should stay.

## Reproducing

```sh
python3 tools/expert_relayout.py --in SNAP --perm perm.json --out SNAP_RELAID \
    --measure trace.txt
```

The `castore` row is derived from the input container's own tensor list, so no
CA store is required to reproduce it.
