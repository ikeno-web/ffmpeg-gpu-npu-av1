#!/usr/bin/env python3
"""
vmaf_tune.py — target-VMAF encoder tuner (closed-loop, binary search).

Finds the lowest-bitrate quality setting (`-cq` for NVENC, `-crf` for
SVT-AV1/x264/x265) that still meets a target VMAF, by binary-searching the
quality parameter and measuring VMAF with this build's libvmaf.

Needs an ffmpeg with libvmaf + the encoder (default `ffmpeg` in PATH; override
with the FFMPEG env var) and the VMAF model JSON (vmaf_v0.6.1.json).

Example:
  FFMPEG=./ffmpeg.exe python tools/vmaf_tune.py -i in.mp4 -o out.mp4 \
      --encoder av1_nvenc --target-vmaf 93 --preset p6
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

NVENC = {"av1_nvenc", "hevc_nvenc", "h264_nvenc"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, help="Source video")
    p.add_argument("-o", "--output", help="If set, do a final full encode at the chosen q")
    p.add_argument("--encoder", required=True,
                   choices=["av1_nvenc", "hevc_nvenc", "h264_nvenc",
                            "libsvtav1", "libx264", "libx265"])
    p.add_argument("--target-vmaf", type=float, required=True)
    p.add_argument("--tolerance", type=float, default=1.0,
                   help="VMAF band; flag over-delivery beyond target+tol")
    p.add_argument("--qmin", type=int, default=18)
    p.add_argument("--qmax", type=int, default=40)
    p.add_argument("--sample-seconds", type=float, default=15.0,
                   help="Encode only the first N seconds per search step")
    p.add_argument("--preset", help="Encoder preset passed through")
    p.add_argument("--vmaf-model", help="Path to vmaf_v0.6.1.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def find_vmaf_model(explicit):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "vmaf_v0.6.1.json"),
              os.path.join(here, "..", "vmaf_v0.6.1.json"),
              "vmaf_v0.6.1.json"):
        if os.path.isfile(c):
            return os.path.abspath(c)
    return "vmaf_v0.6.1.json"   # let ffmpeg fail with a clear message


def quality_args(encoder, q, preset):
    args = ["-rc", "vbr", "-cq", str(q)] if encoder in NVENC else ["-crf", str(q)]
    if preset:
        args += ["-preset", preset]
    return args


def run(cmd, verbose, cwd=None):
    if verbose:
        print("$ " + " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if r.returncode != 0:
        tail = "\n".join(r.stderr.decode("utf-8", "replace").splitlines()[-20:])
        raise RuntimeError(f"ffmpeg failed ({r.returncode}):\n  {' '.join(cmd)}\n{tail}")
    return r


def encode(ffmpeg, encoder, q, preset, src, out, sample, verbose, full=False):
    t = [] if full else ["-t", str(sample)]
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           *t, "-i", src, *quality_args(encoder, q, preset), "-c:v", encoder]
    cmd += (["-c:a", "copy"] if full else ["-an"]) + [out]
    run(cmd, verbose)
    return out


def measure_vmaf(ffmpeg, model, src, encoded, sample, verbose):
    # input 0 = reference (source), input 1 = distorted (encoded);
    # libvmaf takes [distorted][reference].
    # A Windows drive colon (C:) in the model path breaks the lavfi option
    # parser, so reference the model by basename and run from its directory.
    model_dir = os.path.dirname(os.path.abspath(model)) or "."
    model_name = os.path.basename(model)
    cmd = [ffmpeg, "-hide_banner",
           "-t", str(sample), "-i", os.path.abspath(src),
           "-t", str(sample), "-i", os.path.abspath(encoded),
           "-lavfi", f"[1:v][0:v]libvmaf=model='path={model_name}'",
           "-f", "null", "-"]
    err = run(cmd, verbose, cwd=model_dir).stderr.decode("utf-8", "replace")
    m = re.findall(r"VMAF score:\s*([0-9.]+)", err)
    if not m:
        tail = "\n".join(err.splitlines()[-30:])
        raise RuntimeError("No VMAF score in output:\n" + tail)
    return float(m[-1])


def kbps(path, seconds):
    return os.path.getsize(path) * 8 / 1000.0 / seconds if seconds > 0 else 0.0


def main():
    a = parse_args()
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    model = find_vmaf_model(a.vmaf_model)
    if a.qmin > a.qmax:
        sys.exit("Error: --qmin must be <= --qmax")
    if not os.path.isfile(model):
        print(f"Warning: VMAF model '{model}' not found; measurement may fail.",
              file=sys.stderr)

    cache = {}            # q -> (vmaf, sample_kbps)
    tried = []            # ordered (q, vmaf, kbps)

    with tempfile.TemporaryDirectory() as tmp:
        def evaluate(q):
            if q in cache:
                return cache[q]
            enc = encode(ffmpeg, a.encoder, q, a.preset, a.input,
                         os.path.join(tmp, f"s_q{q}.mp4"), a.sample_seconds, a.verbose)
            v = measure_vmaf(ffmpeg, model, a.input, enc, a.sample_seconds, a.verbose)
            res = (v, kbps(enc, a.sample_seconds))
            cache[q] = res
            tried.append((q, v, res[1]))
            if a.verbose:
                print(f"  q={q}: VMAF={v:.3f}, {res[1]:.0f} kbps", file=sys.stderr)
            return res

        # We want the SMALLEST file that still meets the target. Both VMAF and
        # bitrate fall as q rises, so that is the LARGEST q whose VMAF >= target.
        lo, hi, best = a.qmin, a.qmax, None
        while lo <= hi:
            mid = (lo + hi) // 2
            v, _ = evaluate(mid)
            if v >= a.target_vmaf:
                best = mid          # meets target — try a higher q (smaller file)
                lo = mid + 1
            else:
                hi = mid - 1        # quality too low — drop q

        reachable = best is not None
        if not reachable:
            # even the highest quality (qmin) misses the target
            print(f"Warning: target VMAF {a.target_vmaf} not reached even at the "
                  f"highest quality q={a.qmin}; using q={a.qmin}.", file=sys.stderr)
            best = a.qmin
            evaluate(best)

        best_vmaf, best_kbps = cache[best]

        print("\n q   VMAF      kbps")
        for q, v, k in sorted(tried):
            mark = " *" if q == best else ""
            print(f"{q:>3}  {v:7.3f}  {k:8.0f}{mark}")

        note = ""
        if reachable and best == a.qmax and best_vmaf - a.target_vmaf > a.tolerance:
            note = (f"  (qmax-limited: still {best_vmaf - a.target_vmaf:.1f} over target — "
                    f"raise --qmax to compress further)")
        elif reachable and best_vmaf - a.target_vmaf > a.tolerance:
            note = f"  (nearest step is {best_vmaf - a.target_vmaf:.2f} over target)"
        print(f"\nBest: q={best}  VMAF={best_vmaf:.3f}  "
              f"sample_bitrate={best_kbps:.0f} kbps{note}")

        if a.output:
            print(f"Final encode -> {a.output} (q={best}) ...")
            encode(ffmpeg, a.encoder, best, a.preset, a.input, a.output,
                   a.sample_seconds, a.verbose, full=True)
            print(f"Done: {a.output} ({os.path.getsize(a.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
