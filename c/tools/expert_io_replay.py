#!/usr/bin/env python3
"""Replay a trace's expert reads against a real container, page cache bypassed (#36).

`expert_relayout.py --measure` counts reads and bytes. It cannot say whether
fewer, longer reads are actually FASTER on a given drive, and #36 names that gap
as one of the three things that could kill the idea:

> **The page cache may already absorb it** on machines with RAM to spare — most
> likely to matter on the small-RAM boxes, which is also where it is most valuable.

This closes it. For each (moe call, layer) in a ROUTE_TRACE, it takes the union
of routed experts, resolves their real byte ranges from the container's
safetensors headers, merges the ranges that touch, and issues exactly those
preads — the I/O the engine would issue for that layout, and nothing else.

Run it against the original and the relaid container and the only variable is
the order of bytes in the file.

## Cold vs warm, and why both are reported

`--cold` bypasses the page cache: `F_NOCACHE` on macOS, `POSIX_FADV_DONTNEED`
per range on Linux. That is the small-RAM box the issue cares about, simulated
on a large one, without needing root.

`--warm` (the default) leaves the cache alone. On a host whose RAM exceeds the
container this measures the cache rather than the layout, and is expected to
show little difference — reporting it is what makes the cold number meaningful
instead of a lone figure with nothing to compare against.

Neither number is a claim about a real workload's tok/s: this replays reads
with no compute in between, so it isolates I/O rather than predicting decode
throughput. It answers "are these reads cheaper", which is the question #36
actually asks.

Usage:
  python3 tools/expert_io_replay.py --trace T --container DIR [--cold] [--repeat 3]
  python3 tools/expert_io_replay.py --trace T --a DIR1 --b DIR2 --cold
  python3 tools/expert_io_replay.py --selftest
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_relayout as er           # byte_ranges / routed_sets / count_reads

F_NOCACHE = 48                          # <sys/fcntl.h>, macOS only


def plan(trace, container):
    """[(shard_index, offset, length)] in the order the engine would issue them.

    Same merge rule as expert_relayout.count_reads, deliberately: the benchmark
    must execute exactly the reads the measurement counted, or the two numbers
    describe different things.
    """
    per = er.routed_sets(trace)
    rng = er.byte_ranges(container)
    reads = []
    for (call, layer) in sorted(per):
        segs = []
        for e in per[(call, layer)]:
            segs.extend(rng.get((layer, e), []))
        if not segs:
            continue
        segs.sort()
        cs, ca, cb = segs[0]
        for sid, a, b in segs[1:]:
            if sid == cs and a <= cb:
                cb = max(cb, b)
            else:
                reads.append((cs, ca, cb - ca))
                cs, ca, cb = sid, a, b
        reads.append((cs, ca, cb - ca))
    return reads


def open_shards(container, cold):
    shards = sorted(f for f in os.listdir(container) if f.endswith(".safetensors"))
    fds = []
    for f in shards:
        fd = os.open(os.path.join(container, f), os.O_RDONLY)
        if cold and sys.platform == "darwin":
            import fcntl
            fcntl.fcntl(fd, F_NOCACHE, 1)
        fds.append(fd)
    return fds


def drop_cache(fds, cold):
    """Linux: evict what the previous pass left resident. macOS: F_NOCACHE did it."""
    if not cold or sys.platform == "darwin":
        return
    for fd in fds:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass


def run(reads, fds, cold):
    drop_cache(fds, cold)
    moved = 0
    t0 = time.monotonic()
    for sid, off, length in reads:
        got = os.pread(fds[sid], length, off)
        if len(got) != length:
            raise SystemExit(f"short read: wanted {length} at {off} in shard {sid}")
        moved += len(got)
    return time.monotonic() - t0, moved


def bench(trace, container, cold, repeat):
    reads = plan(trace, container)
    fds = open_shards(container, cold)
    try:
        best = None
        for _ in range(repeat):
            secs, moved = run(reads, fds, cold)
            if best is None or secs < best[0]:
                best = (secs, moved)
    finally:
        for fd in fds:
            os.close(fd)
    secs, moved = best
    return {"reads": len(reads), "bytes": moved, "secs": secs,
            "mb_s": moved / secs / 1e6, "mean_read": moved / len(reads)}


def show(label, r):
    print(f"  {label:10s} {r['reads']:8d} reads  {r['bytes']/1e9:7.2f} GB  "
          f"{r['mean_read']/1e6:6.2f} MB/read  {r['secs']:8.2f}s  {r['mb_s']:8.1f} MB/s")


# ---------------------------------------------------------------- selftest
def selftest():
    import json
    import struct
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # one shard, 4 experts x 2 tensors, scales-then-weights (the real pathology)
        names, size = [], 4096
        for kind in ("qs", "w"):
            for eid in range(4):
                names.append(f"model.layers.0.mlp.experts.{eid}.{kind}")
        header, cursor = {}, 0
        for n in names:
            header[n] = {"dtype": "U8", "shape": [size], "data_offsets": [cursor, cursor + size]}
            cursor += size
        blob = json.dumps(header, separators=(",", ":")).encode()
        path = os.path.join(tmp, "model-00000.safetensors")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            for i, _ in enumerate(names):
                fh.write(bytes([i % 251]) * size)

        trace = os.path.join(tmp, "t.txt")
        with open(trace, "w") as fh:
            for c in range(4):
                fh.write(f"{c} 0 0 0:0.5 1:0.5\n")

        reads = plan(trace, tmp)
        # experts 0 and 1 have their qs adjacent and their w adjacent, but the two
        # groups are apart -> 2 reads per call, 4 calls
        if len(reads) != 8:
            print(f"selftest: expected 8 reads, got {len(reads)}")
            ok = False
        want = sum(l for _, _, l in reads)

        for cold in (False, True):
            r = bench(trace, tmp, cold=cold, repeat=2)
            if r["bytes"] != want:
                print(f"selftest: cold={cold} moved {r['bytes']}, expected {want}")
                ok = False
            if r["secs"] <= 0:
                print(f"selftest: cold={cold} non-positive elapsed")
                ok = False

        # the plan must agree with what expert_relayout counts, or the benchmark
        # and the measurement are describing different things
        counted, _ = er.count_reads(er.routed_sets(trace), er.byte_ranges(tmp))
        if counted != len(reads):
            print(f"selftest: plan has {len(reads)} reads, count_reads says {counted}")
            ok = False

    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    trace = container = a = b = None
    cold, repeat = False, 3
    i = 0

    def value(flag):
        nonlocal i
        arg = args[i]
        if "=" in arg:
            i += 1
            return arg.split("=", 1)[1]
        if i + 1 >= len(args):
            raise SystemExit(f"{flag}: expected a value")
        i += 2
        return args[i - 1]

    while i < len(args):
        arg = args[i]
        if arg.startswith("--trace"):
            trace = value("--trace")
        elif arg.startswith("--container"):
            container = value("--container")
        elif arg.startswith("--repeat"):
            repeat = int(value("--repeat"))
        elif arg.startswith("--a"):
            a = value("--a")
        elif arg.startswith("--b"):
            b = value("--b")
        elif arg == "--cold":
            cold = True
            i += 1
        elif arg == "--warm":
            cold = False
            i += 1
        else:
            raise SystemExit(f"unknown argument: {arg}")

    if not trace or not (container or (a and b)):
        raise SystemExit(__doc__)

    mode = "cold (page cache bypassed)" if cold else "warm (page cache in play)"
    plat = "F_NOCACHE" if sys.platform == "darwin" else "POSIX_FADV_DONTNEED"
    print(f"{os.path.basename(trace)} — {mode}"
          + (f", {plat}" if cold else "")
          + f", best of {repeat}")

    if container:
        show(os.path.basename(container.rstrip("/")), bench(trace, container, cold, repeat))
        return 0

    ra = bench(trace, a, cold, repeat)
    rb = bench(trace, b, cold, repeat)
    show(os.path.basename(a.rstrip("/")), ra)
    show(os.path.basename(b.rstrip("/")), rb)
    if ra["bytes"] != rb["bytes"]:
        print(f"  WARNING: byte totals differ ({ra['bytes']} vs {rb['bytes']}) — "
              "these containers do not hold the same tensors")
    print(f"\n  reads   {1 - rb['reads']/ra['reads']:+.3f}")
    print(f"  wall    {1 - rb['secs']/ra['secs']:+.3f}   "
          f"({ra['secs']:.2f}s -> {rb['secs']:.2f}s)")
    print(f"  throughput {ra['mb_s']:.0f} -> {rb['mb_s']:.0f} MB/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
