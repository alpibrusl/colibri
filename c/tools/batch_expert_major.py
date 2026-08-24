#!/usr/bin/env python3
"""#37 proper: token-major (round-robin) vs EXPERT-MAJOR scheduling.

The difference is the whole proposal. Within one decode step:

  token-major  : for each session, for each of its experts -> fetch/use.
                 Session 1's fetches can evict what session 2 is about to need,
                 so a small cache thrashes and S sessions can cost MORE than S
                 separate runs.
  expert-major : take the UNION of experts the step needs, touch each ONCE, and
                 run every queued token that routes to it before moving on.
                 Each distinct expert is fetched at most once per step whatever
                 the cache size -- that is the amortization #37 is claiming.

Both keep the same bounded per-layer LRU across steps, so temporal reuse is
modelled identically and only the intra-step ORDER differs.
"""
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from batch_union import parse

def simulate(sessions, S, cap, expert_major, warm=4):
    cache = defaultdict(OrderedDict)
    T = min(len(s) for s in sessions)
    miss = hit = tokens = 0
    for t in range(T):
        if expert_major:
            need = defaultdict(set)                      # layer -> union of experts
            for i in range(S):
                for L, experts in sessions[i][t].items():
                    need[L] |= experts
            for L, experts in need.items():
                lru = cache[L]
                for e in sorted(experts):                # each distinct expert ONCE
                    if e in lru:
                        lru.move_to_end(e)
                        if t >= warm: hit += 1
                    else:
                        if t >= warm: miss += 1
                        lru[e] = True
                        while len(lru) > cap: lru.popitem(last=False)
        else:
            for i in range(S):
                for L, experts in sessions[i][t].items():
                    lru = cache[L]
                    for e in sorted(experts):
                        if e in lru:
                            lru.move_to_end(e)
                            if t >= warm: hit += 1
                        else:
                            if t >= warm: miss += 1
                            lru[e] = True
                            while len(lru) > cap: lru.popitem(last=False)
        if t >= warm: tokens += S
    return miss, hit, tokens

def main(paths):
    sessions = [s for s in (parse(p) for p in paths) if s]
    n = len(sessions)
    print(f"sessions: {n}   decode steps: {min(len(s) for s in sessions)}")
    print("miss/token = distinct expert fetches per token produced (lower is better)\n")
    print(f"{'cap':>5} {'S':>3} {'token-major':>13} {'expert-major':>14} {'gain':>7} {'vs S=1':>8}")
    print("-"*56)
    for cap in (32, 48, 64, 96, 128, 192):
        base = None
        for S in (1, 3, 5):
            if S > n: continue
            m1,_,tok = simulate(sessions, S, cap, False)
            m2,_,_   = simulate(sessions, S, cap, True)
            a, b = m1/tok, m2/tok
            if base is None: base = b
            print(f"{cap:>5} {S:>3} {a:>13.1f} {b:>14.1f} {a/b:>6.2f}x {b/base:>7.2f}x")
        print()

if __name__ == "__main__":
    main(sys.argv[1:])
