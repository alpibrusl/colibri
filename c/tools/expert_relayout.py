#!/usr/bin/env python3
"""Rewrite a safetensors container so co-activating experts are adjacent (#36).

The measurement in `docs/experiments/olmoe-routing-structure-2026-08-20.md` said
this is worth building: on a real trained OLMoE, a co-activation-clustered layout
issues **31.4% fewer reads** than today's container order on held-out forwards,
and 30.7% fewer than a random control that barely moves. This is the container
half of that slice. `tools/expert_layout.py --perm=` produces the permutation;
this consumes it.

## Nothing about the model changes

Same tensors, same names, same dtypes, same shapes, same bytes. Only their ORDER
in the file differs, and the header's `data_offsets` are rewritten to match.
`--verify` re-reads the output and compares every tensor's sha256 against the
input, because "the bytes are identical" is the whole safety argument and it
should be checked rather than asserted.

## Why the loader needs no change

#36 scoped this as "the container rewrite **+ the loader's `expert_id -> slot`
mapping**". The second half turns out to be unnecessary, and that is worth
stating plainly so nobody writes it:

`expert_load_impl` (colibri.c) and `expert_read` (olmoe.c) both locate an
expert's bytes by building its tensor NAME and calling `st_find`, which resolves
through the safetensors header's `data_offsets`. No offset is ever derived
arithmetically from an expert id. So a container whose data blocks are permuted
and whose header is updated to match is already correct to every existing
reader -- expert 5 is still named expert 5, it just lives elsewhere in the file.
A `expert_id -> slot` indirection would be a second source of truth for
something the header already states.

## Two wins, not one

1. **Across experts** -- the permutation, which is what #36 measured.

2. **Within an expert** -- the merged OLMoE container stores every `.qs` scale
   array in one region and every `merged_weight` in another, so the two reads
   olmoe.c issues per expert (`st_read_raw` then `st_read_f32`) land megabytes
   apart. Emitting an expert's tensors adjacently costs nothing and puts them on
   the same track. colibri.c's GLM path already depends on this kind of
   adjacency: it sorts gate/up/down by offset and takes a single coalesced pread
   only `if(contig)`, falling back to three reads otherwise. This rewrite
   preserves that property rather than disturbing it.

Dense (non-expert) tensors keep their relative order and are emitted first: they
load once at startup and never compete with the streaming path.

## Sharding

Input shard boundaries are not preserved. 5 of 16 layers in the reference
checkpoint have their experts split across two shards, so honouring the old
boundaries would fragment exactly the layers the permutation is trying to make
contiguous. Output is packed to --shard-bytes, and one expert's tensor group is
never split across a boundary.

Usage:
  python3 tools/expert_relayout.py --in SNAP --perm perm.json --out SNAP2 [--verify]
  python3 tools/expert_relayout.py --in SNAP --perm perm.json --out SNAP2 --measure trace.txt
  python3 tools/expert_relayout.py --selftest        (no weights needed)
"""
import hashlib
import json
import os
import re
import shutil
import struct
import sys

# model.layers.<L>.mlp.experts.<E>.<rest>  -- matches both the merged OLMoE
# container (merged_weight/qs) and the split GLM one (gate_proj/up_proj/
# down_proj/.qs). Anything that does not match is dense by definition.
RE_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")

DEFAULT_SHARD_BYTES = 2 << 30

# Within one expert, big tensors first: the weight is the read worth optimizing
# and the scales ride along behind it. Ordering is by descending size, with the
# name as a tiebreak so the output is deterministic.
def _expert_tensor_key(entry):
    return (-entry["nbytes"], entry["name"])


def read_header(path):
    """(header dict, data section start). Raises on a malformed container."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors length prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (512 << 20):      # same ceiling as st.h ST_MAX_HEADER
            raise ValueError(f"{path}: implausible header length {hlen}")
        header = json.loads(fh.read(hlen))
    return header, 8 + hlen


def index_container(src):
    """Every tensor in every shard: name -> {path, off, nbytes, dtype, shape}."""
    shards = sorted(
        os.path.join(src, f) for f in os.listdir(src) if f.endswith(".safetensors")
    )
    if not shards:
        raise SystemExit(f"{src}: no .safetensors shards")
    tensors, order, metadata = {}, [], {}
    for path in shards:
        header, data_start = read_header(path)
        for name, meta in header.items():
            if name == "__metadata__":
                # Keep the first shard's metadata; they agree in practice and a
                # disagreement is not this tool's to arbitrate.
                metadata = metadata or meta
                continue
            if name in tensors:
                raise SystemExit(f"{name}: appears in two shards, refusing")
            begin, end = meta["data_offsets"]
            tensors[name] = {
                "name": name,
                "path": path,
                "off": data_start + begin,
                "nbytes": end - begin,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
            }
            order.append(name)
    return tensors, order, metadata


def group_experts(tensors, order):
    """(dense names in original order, {layer: {expert: [entries]}})."""
    dense, experts = [], {}
    for name in order:
        m = RE_EXPERT.match(name)
        if not m:
            dense.append(name)
            continue
        layer, eid = int(m.group(1)), int(m.group(2))
        experts.setdefault(layer, {}).setdefault(eid, []).append(tensors[name])
    for layer in experts:
        for eid in experts[layer]:
            experts[layer][eid].sort(key=_expert_tensor_key)
    return dense, experts


def emission_order(tensors, order, perm):
    """The new file order: dense first, then each layer's experts in slot order.

    An expert present in the container but absent from the permutation keeps a
    stable place after the permuted ones rather than being dropped -- a
    permutation fitted on a trace only names experts the trace actually routed
    to, and a container must carry all of them regardless.
    """
    dense, experts = group_experts(tensors, order)
    out = [tensors[n] for n in dense]
    dropped = 0
    for layer in sorted(experts):
        want = [int(e) for e in perm.get("layers", {}).get(str(layer), [])]
        present = experts[layer]
        seen, seq = set(), []
        for eid in want:
            if eid in present and eid not in seen:
                seen.add(eid)
                seq.append(eid)
        tail = sorted(e for e in present if e not in seen)
        dropped += len(tail)
        for eid in seq + tail:
            out.extend(present[eid])
    return out, dropped


def write_container(entries, dst, shard_bytes, metadata):
    """Stream `entries` into shards of at most shard_bytes. Returns shard paths.

    Tensors belonging to one expert are kept in the same shard: they arrive
    adjacent in `entries`, and a boundary is only taken between experts.
    """
    os.makedirs(dst, exist_ok=True)
    written, shard, used = [], [], 0

    def flush():
        nonlocal shard, used
        if not shard:
            return
        path = os.path.join(dst, f"model-{len(written):05d}.safetensors")
        header, cursor = {}, 0
        for e in shard:
            header[e["name"]] = {
                "dtype": e["dtype"],
                "shape": e["shape"],
                "data_offsets": [cursor, cursor + e["nbytes"]],
            }
            cursor += e["nbytes"]
        if metadata:
            header["__metadata__"] = metadata
        blob = json.dumps(header, separators=(",", ":")).encode()
        # 8-byte align the data section: st.h's mmap path refuses an offset that
        # is not 4-byte aligned, and 8 keeps every dtype's natural alignment.
        pad = (-len(blob)) % 8
        blob += b" " * pad
        with open(path, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for e in shard:
                with open(e["path"], "rb") as src:
                    src.seek(e["off"])
                    remaining = e["nbytes"]
                    while remaining:
                        chunk = src.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise SystemExit(f"{e['name']}: short read from {e['path']}")
                        out.write(chunk)
                        remaining -= len(chunk)
        written.append(path)
        shard, used = [], 0

    i = 0
    while i < len(entries):
        # take the whole run of tensors sharing this (layer, expert), or one dense tensor
        run = [entries[i]]
        m = RE_EXPERT.match(entries[i]["name"])
        if m:
            key = (m.group(1), m.group(2))
            j = i + 1
            while j < len(entries):
                mj = RE_EXPERT.match(entries[j]["name"])
                if not mj or (mj.group(1), mj.group(2)) != key:
                    break
                run.append(entries[j])
                j += 1
            i = j
        else:
            i += 1
        run_bytes = sum(e["nbytes"] for e in run)
        if shard and used + run_bytes > shard_bytes:
            flush()
        shard.extend(run)
        used += run_bytes
    flush()
    return written


def sha_of(path, off, nbytes):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(off)
        while nbytes:
            chunk = fh.read(min(nbytes, 8 << 20))
            if not chunk:
                raise SystemExit(f"{path}: short read verifying at {off}")
            h.update(chunk)
            nbytes -= len(chunk)
    return h.hexdigest()


def verify(src_tensors, dst):
    """Every tensor in dst must be byte-identical to the one src named the same."""
    dst_tensors, _, _ = index_container(dst)
    problems = []
    if set(dst_tensors) != set(src_tensors):
        missing = sorted(set(src_tensors) - set(dst_tensors))[:5]
        extra = sorted(set(dst_tensors) - set(src_tensors))[:5]
        problems.append(f"tensor set differs (missing e.g. {missing}, extra e.g. {extra})")
    for name in sorted(set(src_tensors) & set(dst_tensors)):
        a, b = src_tensors[name], dst_tensors[name]
        if a["dtype"] != b["dtype"] or a["shape"] != b["shape"] or a["nbytes"] != b["nbytes"]:
            problems.append(f"{name}: dtype/shape/size changed")
            continue
        if sha_of(a["path"], a["off"], a["nbytes"]) != sha_of(b["path"], b["off"], b["nbytes"]):
            problems.append(f"{name}: CONTENT CHANGED")
    return problems


def locality(entries):
    """Mean gap, in bytes, between an expert's tensors — the within-expert win.

    Reported before and after so the rewrite's second effect is visible without
    a drive: today's merged container puts every .qs in one region and every
    merged_weight in another, so this number starts in the megabytes.
    """
    pos, cursor = {}, 0
    for e in entries:
        pos[e["name"]] = (cursor, e["nbytes"])
        cursor += e["nbytes"]
    gaps = []
    by_expert = {}
    for name, (off, nb) in pos.items():
        m = RE_EXPERT.match(name)
        if m:
            by_expert.setdefault((m.group(1), m.group(2)), []).append((off, nb))
    for parts in by_expert.values():
        if len(parts) < 2:
            continue
        parts.sort()
        span = (parts[-1][0] + parts[-1][1]) - parts[0][0]
        gaps.append(span - sum(nb for _, nb in parts))
    return (sum(gaps) / len(gaps)) if gaps else 0.0


def copy_sidecars(src, dst):
    """config.json, tokenizer files and friends — everything but the shards."""
    for f in sorted(os.listdir(src)):
        s = os.path.join(src, f)
        if f.endswith(".safetensors") or not os.path.isfile(s):
            continue
        shutil.copy2(s, os.path.join(dst, f))


def byte_ranges(src):
    """(layer, expert) -> [(shard_index, start, end)] from the real headers."""
    out = {}
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    for sid, f in enumerate(shards):
        header, data_start = read_header(os.path.join(src, f))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            m = RE_EXPERT.match(name)
            if not m:
                continue
            begin, end = meta["data_offsets"]
            out.setdefault((int(m.group(1)), int(m.group(2))), []).append(
                (sid, data_start + begin, data_start + end))
    return out


def routed_sets(trace):
    """{(call, layer): set(expert)} from a ROUTE_TRACE dump.

    The unit that matters for I/O is the UNION over the batch rows of one
    (call, layer): that is the set the engine must have resident before the
    layer can run, so it is the set whose byte ranges get read.
    """
    per = {}
    with open(trace) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                call, layer = int(p[0]), int(p[2])
            except ValueError:
                continue
            s = per.setdefault((call, layer), set())
            for tok in p[3:]:
                try:
                    s.add(int(tok.split(":", 1)[0]))
                except ValueError:
                    pass
    return per


def count_reads(per, rng):
    """Maximal contiguous runs, i.e. the preads the engine must issue."""
    reads = moved = 0
    for (_call, layer), experts in per.items():
        segs = []
        for e in experts:
            segs.extend(rng.get((layer, e), []))
        if not segs:
            continue
        segs.sort()
        cs, ca, cb = segs[0]
        for sid, a_, b_ in segs[1:]:
            if sid == cs and a_ <= cb:          # touching or overlapping -> one read
                cb = max(cb, b_)
            else:
                reads += 1
                moved += cb - ca
                cs, ca, cb = sid, a_, b_
        reads += 1
        moved += cb - ca
    return reads, moved


def measure(trace, before_dir, after_dir):
    """Score both containers' REAL offsets against a trace. No drive needed."""
    per = routed_sets(trace)
    layers = {l for _, l in per}
    n = len(per) // max(1, len(layers))
    print(f"{os.path.basename(trace)}: {len(per)} moe() calls over {len(layers)} layers = {n} forwards")
    base = None
    for label, d in (("original", before_dir), ("relaid", after_dir)):
        r, moved = count_reads(per, byte_ranges(d))
        if base is None:
            base = r
        print(f"  {label:9s} reads {r:8d}   bytes {moved/1e9:7.2f} GB   mean read {moved/r/1e6:6.2f} MB")
    after, _ = count_reads(per, byte_ranges(after_dir))
    print(f"  read reduction {1 - after/base:+.3f} at identical bytes moved")


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        n_layers, n_experts = 2, 8
        # Two shards, and the pathological real-world layout: all scales first,
        # all weights after, experts in lexicographic id order.
        entries = []
        for kind, size in (("qs", 32), ("merged_weight", 256)):
            for layer in range(n_layers):
                for eid in sorted(range(n_experts), key=str):
                    entries.append((f"model.layers.{layer}.mlp.experts.{eid}.{kind}", size))
        entries.insert(0, ("model.embed_tokens.weight", 128))
        blobs = {n: bytes(((i * 37 + j) % 251 for j in range(sz)))
                 for i, (n, sz) in enumerate(entries)}
        half = len(entries) // 2
        for idx, chunk in enumerate((entries[:half], entries[half:])):
            header, cursor = {}, 0
            for n, sz in chunk:
                header[n] = {"dtype": "U8", "shape": [sz], "data_offsets": [cursor, cursor + sz]}
                cursor += sz
            blob = json.dumps(header, separators=(",", ":")).encode()
            with open(os.path.join(src, f"model-{idx:05d}.safetensors"), "wb") as fh:
                fh.write(struct.pack("<Q", len(blob)))
                fh.write(blob)
                for n, _ in chunk:
                    fh.write(blobs[n])

        tensors, order, metadata = index_container(src)
        if len(tensors) != len(entries):
            print(f"selftest: indexed {len(tensors)}, expected {len(entries)}")
            ok = False

        # a permutation that reverses each layer, i.e. definitely not identity
        perm = {"version": 1,
                "layers": {str(l): list(reversed(range(n_experts))) for l in range(n_layers)}}
        new, dropped = emission_order(tensors, order, perm)
        before = locality([tensors[n] for n in order])
        after = locality(new)
        if dropped:
            print(f"selftest: {dropped} experts fell outside the permutation, expected 0")
            ok = False
        if not (after < before):
            print(f"selftest: within-expert gap did not shrink ({before} -> {after})")
            ok = False

        dst = os.path.join(tmp, "dst")
        write_container(new, dst, shard_bytes=1024, metadata=metadata)
        problems = verify(tensors, dst)
        if problems:
            print("selftest: verify failed:", problems[:3])
            ok = False

        # the emitted order must actually follow the permutation
        dst_t, dst_order, _ = index_container(dst)
        seq = [int(RE_EXPERT.match(n).group(2))
               for n in dst_order if RE_EXPERT.match(n) and RE_EXPERT.match(n).group(1) == "0"]
        first_seen = list(dict.fromkeys(seq))
        if first_seen != list(reversed(range(n_experts))):
            print(f"selftest: layer 0 slot order is {first_seen}, expected reversed")
            ok = False

        # an expert missing from the permutation must survive the rewrite
        partial = {"version": 1, "layers": {"0": [3, 1], "1": []}}
        new2, dropped2 = emission_order(tensors, order, partial)
        if len(new2) != len(entries):
            print(f"selftest: partial permutation lost tensors ({len(new2)} vs {len(entries)})")
            ok = False
        if dropped2 != (n_experts - 2) + n_experts:
            print(f"selftest: unexpected unpermuted count {dropped2}")
            ok = False

    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    src = dst = perm_path = None
    shard_bytes, do_verify, measure_trace = DEFAULT_SHARD_BYTES, False, None
    i = 0

    def value(flag):
        nonlocal i
        a = args[i]
        if "=" in a:
            i += 1
            return a.split("=", 1)[1]
        if i + 1 >= len(args):
            raise SystemExit(f"{flag}: expected a value")
        i += 2
        return args[i - 1]

    while i < len(args):
        a = args[i]
        if a.startswith("--in"):
            src = value("--in")
        elif a.startswith("--out"):
            dst = value("--out")
        elif a.startswith("--perm"):
            perm_path = value("--perm")
        elif a.startswith("--shard-bytes"):
            shard_bytes = int(value("--shard-bytes"))
        elif a == "--verify":
            do_verify = True
            i += 1
        elif a.startswith("--measure"):
            measure_trace = value("--measure")
        else:
            raise SystemExit(f"unknown argument: {a}")
    if not (src and dst and perm_path):
        raise SystemExit(__doc__)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("--in and --out must differ (this rewrites, it does not edit in place)")

    with open(perm_path) as fh:
        perm = json.load(fh)
    if perm.get("version") != 1:
        raise SystemExit(f"{perm_path}: unsupported permutation version {perm.get('version')}")

    tensors, order, metadata = index_container(src)
    new, dropped = emission_order(tensors, order, perm)
    before, after = locality([tensors[n] for n in order]), locality(new)
    total = sum(e["nbytes"] for e in new)

    print(f"{len(tensors)} tensors, {total/1e9:.2f} GB")
    print(f"  within-expert gap: {before/1e6:.1f} MB -> {after/1e6:.1f} MB (mean bytes between an expert's tensors)")
    if dropped:
        print(f"  {dropped} expert(s) not named by the permutation — appended in id order, not dropped")

    shards = write_container(new, dst, shard_bytes, metadata)
    copy_sidecars(src, dst)
    with open(os.path.join(dst, ".coli_layout"), "w") as fh:
        json.dump({"version": 1, "source": os.path.abspath(src),
                   "permutation": perm, "shards": [os.path.basename(s) for s in shards]},
                  fh, indent=1)
    print(f"  wrote {len(shards)} shard(s) -> {dst}  (+ .coli_layout provenance)")

    if do_verify:
        problems = verify(tensors, dst)
        if problems:
            print(f"VERIFY FAILED: {len(problems)} problem(s)")
            for p in problems[:10]:
                print(f"  {p}")
            return 1
        print(f"  verify: {len(tensors)} tensors byte-identical to the source")
    if measure_trace:
        print()
        measure(measure_trace, src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
