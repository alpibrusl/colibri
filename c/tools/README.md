# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `convert_fmt4_to_fmt2.py`: fmt=4 (grouped int4) -> fmt=2 (per-row int4)
  re-quant of a GLM-5.2 container, for Metal-backend compatibility
  (see `docs/METAL-M1ULTRA-FMT2-REPORT.md`)
- `repack_fp8_passthrough.py`: fmt=8 repack (byte-preserved FP8, resident kinds only;
  see the module docstring -- synthetic-fixture-tested only, no real-shard runs yet)
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `gen_unicode.py`: tokenizer table generation
- `route_temporal.py`, `expert_redundancy.py`, `expert_layout.py`: the
  #32 / #31 / #36 offline measurements — each answers a research question from
  artifacts the engine already emits, before any engine code is written.
  `expert_layout.py` scores its layout on a HELD-OUT split against both the
  current order and a random control, because a seriation heuristic scored
  in-sample reports a read reduction on routing that has no structure at all
- `trace_health.py`: whether a ROUTE_TRACE can support a conclusion at all.
  Both routing tools consult it and withhold their verdict on a degenerate
  capture — a random-weight checkpoint through the real engine produced
  +0.135 temporal gain and a 21.6% read reduction, from a router whose gates
  were within rounding of uniform (entropy 0.9999)
- `ledger.py`: one scaling convention and one record shape, so a measurement
  that costs a real checkpoint can be appended to a run ledger
  ([lex-notebooklab](https://github.com/alpibrusl/lex-notebooklab)) instead of
  living in a terminal. Both measurement tools write it with `--out`:

  ```sh
  python3 tools/route_temporal.py --out=report.json trace.txt
  python3 tools/ledger.py --report=report.json --series=route-temporal \
      --attempt=1 --out=entry.json          # then: notebooklab record entry.json
  ```

  Claims are integers (rates in millionths, verdicts as 0/1) so two readers
  compare the same number with no rounding policy to argue about. A ledger
  cannot yet RE-DERIVE these — it has no deriver for a colibri report, so the
  claims record as `UNVERIFIABLE` — which is the honest state, and what
  changes it is engine-side evidence with its own integrity, not a reporting
  change. See `ledger.py`'s `entry()`.

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
```
