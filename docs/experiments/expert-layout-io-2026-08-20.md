# Does the clustered layout actually read faster? (#36)

**Date** 2026-08-20 · **Issue** [#36](https://github.com/alpibrusl/colibri/issues/36) · **Follows** #42 (the gate), #43 (the rewrite)

#36 listed three things that could kill the idea. Two were settled by #42:
routing is not too diffuse, and one layout does serve two workloads. The third
was still open:

> **The page cache may already absorb it** on machines with RAM to spare — most
> likely to matter on the small-RAM boxes, which is also where it is most
> valuable.

This is that measurement. **Both halves of the prediction are correct**: cold,
the clustered layout is meaningfully faster; warm, it is slightly *slower*.

| regime | container | reads | bytes | mean read | wall | throughput |
|---|---|---|---|---|---|---|
| **cold** | original | 2,324 | 11.85 GB | 5.10 MB | 1.53 s | 7,743 MB/s |
| **cold** | **relaid** | **720** | 11.85 GB | **16.45 MB** | **1.33 s** | **8,937 MB/s** |
| warm | original | 2,324 | 11.85 GB | 5.10 MB | 0.59 s | 19,920 MB/s |
| warm | relaid | 720 | 11.85 GB | 16.45 MB | 0.63 s | 18,713 MB/s |

**Cold: −13% wall, +15% throughput, 3.2× fewer reads, identical bytes.**
**Warm: +6.5% wall — the cache absorbs the win and the larger reads cost a little.**

## Method

`tools/expert_io_replay.py` replays the reads a trace implies. For each
(moe call, layer) it takes the union of routed experts, resolves their real byte
ranges from the container's safetensors headers, merges ranges that touch, and
issues exactly those preads — the I/O the engine would issue for that layout,
and nothing else. Run against the original and the relaid container, the only
variable is the order of bytes in the file.

The read plan is asserted in CI to match `expert_relayout.count_reads` exactly,
so the benchmark executes the reads the measurement counts rather than something
adjacent to them.

| | |
|---|---|
| Host | Apple M4 Max, 16 CPU cores, 64 GB RAM, macOS 25.2, internal NVMe |
| Containers | `olmoe_merged` (as converted) and `olmoe_relaid` (#43), 7.41 GB each |
| Trace | corpus A from #42, first 150 moe calls — 11.85 GB of reads |
| Cold | `F_NOCACHE`, page cache evicted between runs, **one pass** per measurement |
| Warm | cache left alone, best of 3 |

### Getting a genuinely cold number was the hard part

Worth recording, because the first attempt was wrong and looked fine.

`F_NOCACHE` alone is **not** sufficient. It is advisory on macOS and still
serves pages already resident from other descriptors, so an "F_NOCACHE" run over
a container the machine had already touched reported **22 GB/s** — no NVMe does
that, and the number was measuring memcpy from the unified buffer cache. Any
result in that range should be discarded on sight.

`purge` requires root. The eviction used here instead reads ~70 GB of unrelated
files already on disk before each measured pass, which is more than the host will
hold, so the container's pages are gone by the time the pass starts. Cold runs
then report 7–9 GB/s, which is plausible for this drive, and are 2.6× slower than
the warm runs — the sanity check that the eviction worked.

Cold measurements are single-pass by construction: a second pass would be warm.
Three alternating trials per container, to catch drift rather than to average
away a real difference:

```
trial 1   original 1.52s  7785 MB/s     relaid 1.32s  8992 MB/s
trial 2   original 1.53s  7743 MB/s     relaid 1.34s  8846 MB/s
trial 3   original 1.66s  7139 MB/s     relaid 1.33s  8937 MB/s
```

The relaid container is also the more *stable* of the two (1.32–1.34 against
1.52–1.66), which is what fewer, longer reads should do to variance.

## What this does and does not establish

**Does:** on this drive, in the streaming regime, the clustered layout moves the
same bytes 13% faster and with 69% fewer syscalls. The direction is not
ambiguous and the trials do not overlap.

**Does not:** predict decode tok/s. This replays reads with no compute between
them, so it isolates I/O rather than modelling a decode loop that overlaps the
two. A real run also has PILOT/COUPLE prefetch hiding some of this latency
already.

**Understates the case, probably.** Two reasons. The measurement host has a fast
internal NVMe, and #36's own benchmark table shows the random-vs-sequential gap
is widest on the cheap QLC/Gen3 drives this engine exists to serve — a Crucial P3
at 1.51 GB/s buffered, against a Samsung 9100 PRO at 8.81 GB/s. And even the cold
pass re-touches experts (11.85 GB of reads over a 7.41 GB model), so a fraction
of it is warm.

**The warm regression is real and should be stated.** On a host whose RAM exceeds
the container, the relaid layout is ~6.5% slower. Nothing is being read from
disk, so the read count is irrelevant and the larger single copies are marginally
worse. Anyone running a model that fits comfortably in RAM should not expect this
to help them, and the layout is not free for them either.

## Verdict for #36

The last stated risk resolves the way the issue guessed it might: the page cache
does absorb the win, exactly on the machines that have RAM to spare, and the win
is real on the machines that do not. Since the container rewrite is offline,
one-time and semantics-preserving (#43), the sensible default is to ship the
relaid layout for streaming deployments and leave full-residency deployments
alone.

## Reproducing

```sh
python3 tools/expert_relayout.py --in SNAP --perm perm.json --out SNAP_RELAID --verify
python3 tools/expert_io_replay.py --trace trace.txt --a SNAP --b SNAP_RELAID --warm --repeat 3

# cold: evict first (needs more than RAM), then ONE pass per container
cat /path/to/70GB/of/other/files > /dev/null
python3 tools/expert_io_replay.py --trace trace.txt --container SNAP        --cold --repeat 1
cat /path/to/70GB/of/other/files > /dev/null
python3 tools/expert_io_replay.py --trace trace.txt --container SNAP_RELAID --cold --repeat 1
```

Discard any cold result reporting double-digit GB/s: that is the page cache, not
the drive.
