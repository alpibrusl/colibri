#!/usr/bin/env python3
"""Add per-tensor SHA-256 checksums to a safetensors container (#13).

The engine verifies a tensor's bytes against `__metadata__["colibri.sha256"]`
the first time it reads them (see st_verify_once in c/st.h). This writes that
map. Weights otherwise load on trust-by-filename plus size arithmetic: a
flipped bit in an int4 shard is read as valid weights, and the NaN policy
carries corrupt fp8 all the way to the sampler.

The map is a JSON object {tensor_name: 64-char lowercase hex}, stored as a
string inside `__metadata__` because safetensors metadata values are always
strings -- the same convention `colibri.fmt` already uses.

Adding metadata changes the header length, which moves every data offset, so
this rewrites the shards rather than patching them in place. Tensor order,
names, dtypes, shapes and bytes are all preserved; only the header changes.
`--verify` re-reads the output and confirms every digest, so the map is checked
against the file it ships with rather than against the file it was computed
from.

Usage:
  python3 tools/checksum_container.py --in SNAP --out SNAP2 [--verify]
  python3 tools/checksum_container.py --selftest
"""
import hashlib
import json
import os
import shutil
import struct
import sys

CHUNK = 8 << 20


def read_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: truncated safetensors length prefix")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > (512 << 20):
            raise SystemExit(f"{path}: implausible header length {hlen}")
        return json.loads(fh.read(hlen)), 8 + hlen


def digest_of(fh, off, nbytes):
    h = hashlib.sha256()
    fh.seek(off)
    while nbytes:
        chunk = fh.read(min(nbytes, CHUNK))
        if not chunk:
            raise SystemExit("short read while hashing")
        h.update(chunk)
        nbytes -= len(chunk)
    return h.hexdigest()


def rewrite(src, dst):
    """Copy every shard, adding the checksum map. Returns {name: digest}."""
    os.makedirs(dst, exist_ok=True)
    shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not shards:
        raise SystemExit(f"{src}: no .safetensors shards")
    all_sums = {}
    for f in shards:
        header, data_start = read_header(os.path.join(src, f))
        meta = dict(header.get("__metadata__") or {})
        tensors = [(k, v) for k, v in header.items() if k != "__metadata__"]

        sums = {}
        with open(os.path.join(src, f), "rb") as fh:
            for name, m in tensors:
                begin, end = m["data_offsets"]
                sums[name] = digest_of(fh, data_start + begin, end - begin)
        all_sums.update(sums)

        # Offsets are relative to the data section, so they survive the header
        # growing -- only data_start moves, and it is recomputed on read.
        meta["colibri.sha256"] = json.dumps(sums, separators=(",", ":"), sort_keys=True)
        out_header = {k: v for k, v in tensors}
        out_header["__metadata__"] = meta
        blob = json.dumps(out_header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)          # keep the data section 8-aligned

        with open(os.path.join(src, f), "rb") as fh, open(os.path.join(dst, f), "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            fh.seek(data_start)
            remaining = max((m["data_offsets"][1] for _, m in tensors), default=0)
            while remaining:
                chunk = fh.read(min(remaining, CHUNK))
                if not chunk:
                    raise SystemExit(f"{f}: short read copying the data section")
                out.write(chunk)
                remaining -= len(chunk)
    for f in sorted(os.listdir(src)):
        s = os.path.join(src, f)
        if not f.endswith(".safetensors") and os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, f))
    return all_sums


def verify(d):
    """Re-read a container and check every tensor against its own stored map."""
    problems, checked = [], 0
    for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
        header, data_start = read_header(os.path.join(d, f))
        meta = header.get("__metadata__") or {}
        if "colibri.sha256" not in meta:
            problems.append(f"{f}: no colibri.sha256 in __metadata__")
            continue
        sums = json.loads(meta["colibri.sha256"])
        with open(os.path.join(d, f), "rb") as fh:
            for name, m in header.items():
                if name == "__metadata__":
                    continue
                if name not in sums:
                    problems.append(f"{name}: not in the checksum map")
                    continue
                begin, end = m["data_offsets"]
                got = digest_of(fh, data_start + begin, end - begin)
                checked += 1
                if got != sums[name]:
                    problems.append(f"{name}: digest mismatch")
    return checked, problems


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        os.makedirs(src)
        names = [("w", 300), ("w.qs", 64), ("x", 128)]
        blobs = {n: bytes((i * 41 + j) % 251 for j in range(sz))
                 for i, (n, sz) in enumerate(names)}
        header, cur = {}, 0
        for n, sz in names:
            header[n] = {"dtype": "U8", "shape": [sz], "data_offsets": [cur, cur + sz]}
            cur += sz
        blob = json.dumps(header, separators=(",", ":")).encode()
        with open(os.path.join(src, "model-00000.safetensors"), "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            for n, _ in names:
                fh.write(blobs[n])

        sums = rewrite(src, dst)
        if set(sums) != {n for n, _ in names}:
            print("selftest: checksum map does not cover every tensor"); ok = False
        for n, _ in names:
            if sums[n] != hashlib.sha256(blobs[n]).hexdigest():
                print(f"selftest: wrong digest for {n}"); ok = False

        checked, problems = verify(dst)
        if problems or checked != len(names):
            print(f"selftest: verify failed ({checked} checked): {problems[:3]}"); ok = False

        # the bytes must survive the rewrite untouched
        h2, ds2 = read_header(os.path.join(dst, "model-00000.safetensors"))
        with open(os.path.join(dst, "model-00000.safetensors"), "rb") as fh:
            for n, _ in names:
                b, e = h2[n]["data_offsets"]
                fh.seek(ds2 + b)
                if fh.read(e - b) != blobs[n]:
                    print(f"selftest: {n} bytes changed in the rewrite"); ok = False

        # a flipped byte must be caught
        path = os.path.join(dst, "model-00000.safetensors")
        with open(path, "r+b") as fh:
            fh.seek(ds2 + 5); cur = fh.read(1); fh.seek(ds2 + 5); fh.write(bytes([cur[0] ^ 1]))
        _, problems = verify(dst)
        if not any("mismatch" in p for p in problems):
            print("selftest: a flipped byte was not detected"); ok = False

    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    src = dst = None
    do_verify = False
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
        elif a == "--verify":
            do_verify = True
            i += 1
        else:
            raise SystemExit(f"unknown argument: {a}")
    if not (src and dst):
        raise SystemExit(__doc__)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit("--in and --out must differ (this rewrites, it does not edit in place)")

    sums = rewrite(src, dst)
    print(f"checksummed {len(sums)} tensors -> {dst}")
    if do_verify:
        checked, problems = verify(dst)
        if problems:
            print(f"VERIFY FAILED: {len(problems)} problem(s)")
            for p in problems[:10]:
                print(f"  {p}")
            return 1
        print(f"  verify: {checked} tensors match their stored digest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
