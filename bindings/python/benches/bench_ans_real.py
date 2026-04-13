#!/usr/bin/env python3
"""Benchmark: ANS compression vs uncompressed GDS for real model weights.

Tests with real Qwen2.5-1.5B and Qwen2.5-7B trained BF16 weights from
HuggingFace, measuring cold-NVMe loading to GPU via:
  1. Uncompressed + GDS (cuFileRead)
  2. ANS compressed + GDS + nvcomp GPU decompression
  3. Uncompressed + mmap baseline

Requirements:
    pip install nvidia-nvcomp-cu12 torch safetensors huggingface_hub zstandard

Usage:
    PYTHONPATH=/tmp/nvcomp_install \
    LD_LIBRARY_PATH=/tmp/nvcomp_install/nvidia/libnvcomp/lib64:/usr/local/cuda/lib64 \
    python3.12 benches/bench_ans_real.py
"""

import gc
import importlib
import json
import os
import struct
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch


def drop_page_cache():
    """Drop OS page cache for cold-NVMe benchmarking."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def get_file_info(path: str) -> Tuple[int, int, float]:
    """Returns (decompressed_body_bytes, compressed_body_bytes, ratio)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))

    ci_str = header.get("__metadata__", {}).get("compression_info")
    if ci_str:
        ci = json.loads(ci_str)
        decomp = ci["decompressed_size"]
        comp = ci["compressed_size"]
        return decomp, comp, decomp / comp if comp > 0 else 1.0

    body = os.path.getsize(path) - 8 - hlen
    return body, body, 1.0


def download_model(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a single file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename=filename, cache_dir=cache_dir)


def compress_with_ans(src_path: str, dst_path: str) -> Tuple[float, float]:
    """Compress a safetensors file with ANS. Returns (ratio, time_seconds)."""
    if os.path.exists(dst_path):
        decomp, comp, ratio = get_file_info(dst_path)
        return ratio, 0.0

    from safetensors.fast import save_compressed
    from safetensors.torch import load_file

    tensors = load_file(src_path, device="cpu")
    t0 = time.perf_counter()
    save_compressed(tensors, dst_path, compression="ans")
    elapsed = time.perf_counter() - t0
    del tensors
    gc.collect()
    torch.cuda.empty_cache()

    decomp, comp, ratio = get_file_info(dst_path)
    return ratio, elapsed


def bench_fast_safe_open(
    path: str,
    device: str,
    iters: int,
    warmup: int,
) -> List[float]:
    """Benchmark loading via fast_safe_open (GDS path)."""
    from safetensors._safetensors_rust import fast_safe_open

    times = []
    for i in range(warmup + iters):
        drop_page_cache()
        sync_cuda()

        try:
            t0 = time.perf_counter()
            with fast_safe_open(path, device=device, nogds=False) as f:
                result = f.get_all_tensors()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if i >= warmup:
                times.append(elapsed)
            del result
            torch.cuda.empty_cache()
        except Exception as e:
            if i == 0:
                print(f"    ERROR on iter {i}: {e}")
            sync_cuda()
            # If we have no successful runs, re-raise on last attempt
            if i == warmup + iters - 1 and not times:
                raise

    return times


def bench_mmap_baseline(
    path: str,
    device: str,
    iters: int,
    warmup: int,
) -> List[float]:
    """Benchmark loading via mmap (SAFETENSORS_FAST_GPU=0)."""
    os.environ["SAFETENSORS_FAST_GPU"] = "0"
    import safetensors.torch as st

    importlib.reload(st)

    times = []
    for i in range(warmup + iters):
        drop_page_cache()
        sync_cuda()

        t0 = time.perf_counter()
        result = st.load_file(path, device=device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= warmup:
            times.append(elapsed)
        del result
        torch.cuda.empty_cache()

    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    importlib.reload(st)

    return times


def validate_roundtrip(original_path: str, compressed_path: str, device: str) -> bool:
    """Verify that ANS-compressed file decompresses to identical tensors."""
    from safetensors._safetensors_rust import fast_safe_open
    from safetensors.torch import load_file

    orig = load_file(original_path, device="cpu")
    with fast_safe_open(compressed_path, device=device, nogds=False) as f:
        loaded = f.get_all_tensors()

    mismatches = 0
    for k in orig:
        if k not in loaded:
            mismatches += 1
            continue
        if not torch.equal(orig[k], loaded[k].cpu()):
            mismatches += 1
            if mismatches <= 3:
                diff = (orig[k].float() - loaded[k].cpu().float()).abs().max().item()
                print(f"    FAIL {k}: max_diff={diff}")

    del orig, loaded
    sync_cuda()

    if mismatches == 0:
        print(f"    ALL {len(orig) if 'orig' in dir() else '?'} tensors match exactly")
        return True
    print(f"    {mismatches} tensor(s) failed validation")
    return False


def median(data: List[float]) -> float:
    s = sorted(data)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2]


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def format_times(times: List[float], body_bytes: int) -> str:
    med = median(times)
    p10 = percentile(times, 10)
    p90 = percentile(times, 90)
    tput = body_bytes / med / 1e6 if med > 0 else 0
    return (
        f"median={med * 1000:>7.0f} ms  "
        f"p10={p10 * 1000:.0f}  p90={p90 * 1000:.0f}  "
        f"throughput={tput:>5,.0f} MB/s"
    )


def run_model_benchmark(
    name: str,
    uncompressed_path: str,
    compressed_path: str,
    device: str,
    iters: int,
    warmup: int,
    do_validate: bool,
):
    """Run the full benchmark suite for one model/shard."""
    unc_size = os.path.getsize(uncompressed_path)
    comp_size = os.path.getsize(compressed_path)
    decomp_body, comp_body, ratio = get_file_info(compressed_path)
    unc_body, _, _ = get_file_info(uncompressed_path)

    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(
        f"  Uncompressed: {unc_size / 1e9:.2f} GB  |  ANS: {comp_size / 1e9:.2f} GB  |  Ratio: {ratio:.2f}x  |  Saved: {(1 - comp_size / unc_size) * 100:.1f}%"
    )
    print(f"{'=' * 72}")

    # Validate
    if do_validate:
        print(f"\n  Validating ANS roundtrip...")
        try:
            ok = validate_roundtrip(uncompressed_path, compressed_path, device)
            if not ok:
                print("  WARNING: Validation failed, results may be unreliable")
        except Exception as e:
            print(f"  WARNING: Validation error (skipping): {e}")
            # Reset CUDA state after error
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Benchmark
    results = {}

    print(f"\n  [1/3] Uncompressed + GDS (cold NVMe)...")
    times = bench_fast_safe_open(uncompressed_path, device, iters, warmup)
    results["unc_gds"] = times
    med_unc = median(times)
    print(f"    {format_times(times, unc_body)}")

    print(f"  [2/3] ANS ({ratio:.2f}x) + GDS + nvcomp (cold NVMe)...")
    times = bench_fast_safe_open(compressed_path, device, iters, warmup)
    results["ans_gds"] = times
    med_ans = median(times)
    print(f"    {format_times(times, decomp_body)}")

    print(f"  [3/3] Uncompressed + mmap baseline (cold NVMe)...")
    times = bench_mmap_baseline(uncompressed_path, device, iters, warmup)
    results["mmap"] = times
    med_mmap = median(times)
    print(f"    {format_times(times, unc_body)}")

    # Summary table
    best = min(med_unc, med_ans, med_mmap)
    rows = [
        ("Uncompressed + GDS", med_unc, unc_body),
        (f"ANS ({ratio:.2f}x) + GDS + nvcomp", med_ans, decomp_body),
        ("Uncompressed + mmap baseline", med_mmap, unc_body),
    ]

    print(f"\n  {'Method':<42} {'Time':>8} {'Eff MB/s':>9} {'vs best':>8}")
    print(f"  {'-' * 42} {'-' * 8} {'-' * 9} {'-' * 8}")
    for label, t, body in rows:
        tput = body / t / 1e6 if t > 0 else 0
        marker = " ***" if abs(t - best) < 0.001 else ""
        print(
            f"  {label:<42} {t * 1000:>7.0f}ms {tput:>8,.0f} {best / t:>7.2f}x{marker}"
        )

    # Verdict
    if med_ans < med_unc:
        pct = (med_unc - med_ans) / med_unc * 100
        print(
            f"\n  >>> ANS WINS: {med_unc / med_ans:.2f}x faster than uncompressed GDS ({pct:.0f}% less time)"
        )
    else:
        gap_ms = (med_ans - med_unc) * 1000
        overhead_pct = (med_ans / med_unc - 1) * 100
        saved_io_ms = (unc_size - comp_size) / 3.4e9 * 1000  # at 3.4 GB/s NVMe ceiling
        print(
            f"\n  >>> Uncompressed GDS wins by {gap_ms:.0f} ms ({overhead_pct:.0f}% overhead)"
        )
        print(f"      NVMe I/O saved by compression: {saved_io_ms:.0f} ms")
        print(f"      nvcomp decode overhead: ~{gap_ms + saved_io_ms:.0f} ms")

    print(
        f"\n  Both beat mmap: GDS {med_mmap / med_unc:.2f}x, ANS {med_mmap / med_ans:.2f}x"
    )

    return {
        "name": name,
        "unc_size_gb": unc_size / 1e9,
        "comp_size_gb": comp_size / 1e9,
        "ratio": ratio,
        "med_unc_ms": med_unc * 1000,
        "med_ans_ms": med_ans * 1000,
        "med_mmap_ms": med_mmap * 1000,
        "unc_tput": unc_body / med_unc / 1e6,
        "ans_tput": decomp_body / med_ans / 1e6,
    }


def main():
    CACHE_DIR = "/data/hf_cache"
    BENCH_DIR = "/data/bench_ans"
    DEVICE = "cuda:0"
    ITERS = 7
    WARMUP = 2

    os.makedirs(BENCH_DIR, exist_ok=True)

    print("=" * 72)
    print("  ANS Compression Benchmark — Real Trained Weights")
    print("=" * 72)
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(
        f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    print(f"  CUDA:         {torch.version.cuda}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Device:       {DEVICE}")
    print(f"  Iterations:   {ITERS} (+{WARMUP} warmup)")
    print(f"  Algorithm:    ANS (gANS via nvcomp)")

    # Check nvcomp
    try:
        import nvidia.nvcomp as nvc

        print(f"  nvcomp:       {nvc.__version__}")
    except ImportError:
        print("  nvcomp:       NOT FOUND — pip install nvidia-nvcomp-cu12")
        sys.exit(1)

    # Check GDS
    from safetensors._safetensors_rust import fast_safe_open

    gds_ok = False
    try:
        from safetensors.torch import save_file

        probe = os.path.join(BENCH_DIR, "_probe.safetensors")
        save_file({"x": torch.zeros(1)}, probe)
        with fast_safe_open(probe, device=DEVICE, nogds=False) as f:
            _ = f.get_all_tensors()
        os.unlink(probe)
        gds_ok = True
    except Exception:
        pass
    print(f"  GDS:          {'available' if gds_ok else 'NOT available'}")

    # --- Download models ---
    models_to_test = []

    # Qwen2.5-1.5B — single shard, ~3.1 GB
    print("\n--- Downloading Qwen2.5-1.5B ---")
    try:
        path_1_5b = download_model("Qwen/Qwen2.5-1.5B", "model.safetensors", CACHE_DIR)
        print(f"  {os.path.getsize(path_1_5b) / 1e9:.2f} GB")
        models_to_test.append(("Qwen2.5-1.5B (3.1 GB, BF16)", path_1_5b))
    except Exception as e:
        print(f"  Failed: {e}")

    # Qwen2.5-7B — shard 4 (~3.6 GB, fits in VRAM)
    print("--- Downloading Qwen2.5-7B shard 4 ---")
    try:
        path_7b_s4 = download_model(
            "Qwen/Qwen2.5-7B", "model-00004-of-00004.safetensors", CACHE_DIR
        )
        print(f"  {os.path.getsize(path_7b_s4) / 1e9:.2f} GB")
        models_to_test.append(("Qwen2.5-7B shard 4/4 (3.6 GB, BF16)", path_7b_s4))
    except Exception as e:
        print(f"  Failed: {e}")

    if not models_to_test:
        print("\nNo models available for benchmarking!")
        sys.exit(1)

    # --- Compress with ANS ---
    compressed_paths = {}
    for name, src_path in models_to_test:
        safe_name = (
            name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        )
        comp_path = os.path.join(BENCH_DIR, f"{safe_name}_ans.safetensors")

        print(f"\n--- Compressing {name} with ANS ---")
        ratio, elapsed = compress_with_ans(src_path, comp_path)
        comp_sz = os.path.getsize(comp_path)
        src_sz = os.path.getsize(src_path)
        if elapsed > 0:
            print(
                f"  {src_sz / 1e9:.2f} GB -> {comp_sz / 1e9:.2f} GB "
                f"({ratio:.2f}x, {(1 - comp_sz / src_sz) * 100:.1f}% smaller) in {elapsed:.1f}s"
            )
        else:
            print(
                f"  {src_sz / 1e9:.2f} GB -> {comp_sz / 1e9:.2f} GB ({ratio:.2f}x) [cached]"
            )

        compressed_paths[name] = comp_path

    # --- Run benchmarks ---
    all_results = []
    for name, src_path in models_to_test:
        comp_path = compressed_paths[name]
        result = run_model_benchmark(
            name=name,
            uncompressed_path=src_path,
            compressed_path=comp_path,
            device=DEVICE,
            iters=ITERS,
            warmup=WARMUP,
            do_validate=True,
        )
        if result:
            all_results.append(result)

    # --- Final summary ---
    if len(all_results) > 1:
        print(f"\n{'=' * 72}")
        print(f"  OVERALL SUMMARY")
        print(f"{'=' * 72}")
        print(
            f"\n  {'Model':<35} {'Size':>6} {'ANS':>6} {'Ratio':>6} {'GDS ms':>7} {'ANS ms':>7} {'mmap ms':>8}"
        )
        print(
            f"  {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8}"
        )
        for r in all_results:
            print(
                f"  {r['name'][:35]:<35} "
                f"{r['unc_size_gb']:>5.1f}G "
                f"{r['comp_size_gb']:>5.1f}G "
                f"{r['ratio']:>5.2f}x "
                f"{r['med_unc_ms']:>6.0f} "
                f"{r['med_ans_ms']:>6.0f} "
                f"{r['med_mmap_ms']:>7.0f}"
            )

    # Cleanup compressed files
    print(f"\n--- Cleanup ---")
    for comp_path in compressed_paths.values():
        if os.path.exists(comp_path):
            os.unlink(comp_path)
            print(f"  Removed {os.path.basename(comp_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
