#!/usr/bin/env python3
"""Benchmark: GDS loading of compressed vs uncompressed safetensors files.

This script creates realistic model weight files (with random data, not zeros)
in both uncompressed and zstd-compressed formats, then benchmarks loading them
to GPU via the safetensors fast path (GDS when available, pread+bounce fallback
otherwise).

Usage:
    python benches/bench_gds_compressed.py [--size SIZE_MB] [--iters N] [--device DEVICE] [--drop-caches]

Requirements:
    - torch with CUDA support
    - safetensors (built with GDS feature for best results)
    - zstandard Python package

The script drops the page cache between iterations when --drop-caches is passed
(requires root or sudo with NOPASSWD for the drop_caches command).
"""

import argparse
import gc
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import zstandard as zstd_lib


def _try_import_fast():
    """Try to import safetensors fast loading path."""
    try:
        from safetensors._safetensors_rust import fast_safe_open

        return fast_safe_open
    except ImportError:
        return None


def _try_import_load_file():
    from safetensors.torch import load_file

    return load_file


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------


def generate_random_tensors(
    total_mb: float,
    dtype: torch.dtype = torch.float16,
    hidden: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Generate a dict of random tensors approximating a transformer model.

    Uses torch.randn for FP types (realistic entropy for compression benchmarks).
    """
    tensors = OrderedDict()
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    total_bytes = int(total_mb * 1024 * 1024)

    # Mimic transformer structure
    layer = 0
    allocated = 0

    # Embeddings (~10% of total)
    embed_elems = min(total_bytes // (10 * bytes_per_elem), 50257 * hidden)
    if embed_elems > 0:
        shape = _factor_shape(embed_elems, hidden)
        tensors["model.embed_tokens.weight"] = torch.randn(shape, dtype=dtype)
        allocated += embed_elems * bytes_per_elem

    # Layers until we fill the budget
    while allocated < total_bytes:
        remaining = total_bytes - allocated

        # Each layer: q,k,v,o projections + 2 MLP + 2 layernorm ≈ 6*hidden^2 + 2*4*hidden^2 + 2*hidden
        layer_proj_bytes = hidden * hidden * bytes_per_elem
        if remaining < layer_proj_bytes:
            # Add a final small tensor to use up remaining bytes
            n_elems = remaining // bytes_per_elem
            if n_elems > 0:
                tensors[f"model.layers.{layer}.residual"] = torch.randn(
                    n_elems, dtype=dtype
                )
                allocated += n_elems * bytes_per_elem
            break

        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            elems = min(hidden * hidden, (total_bytes - allocated) // bytes_per_elem)
            if elems <= 0:
                break
            shape = _factor_shape(elems, hidden)
            tensors[f"model.layers.{layer}.self_attn.{name}.weight"] = torch.randn(
                shape, dtype=dtype
            )
            allocated += elems * bytes_per_elem

        for name in ["gate_proj", "up_proj"]:
            elems = min(hidden * hidden, (total_bytes - allocated) // bytes_per_elem)
            if elems <= 0:
                break
            shape = _factor_shape(elems, hidden)
            tensors[f"model.layers.{layer}.mlp.{name}.weight"] = torch.randn(
                shape, dtype=dtype
            )
            allocated += elems * bytes_per_elem

        for name in ["input_layernorm", "post_attention_layernorm"]:
            elems = min(hidden, (total_bytes - allocated) // bytes_per_elem)
            if elems <= 0:
                break
            tensors[f"model.layers.{layer}.{name}.weight"] = torch.randn(
                elems, dtype=dtype
            )
            allocated += elems * bytes_per_elem

        layer += 1

    actual_mb = allocated / (1024 * 1024)
    return tensors, actual_mb


def _factor_shape(total_elems: int, preferred_dim: int) -> List[int]:
    """Return a 2D shape [rows, cols] with total_elems elements."""
    if total_elems <= preferred_dim:
        return [total_elems]
    cols = min(preferred_dim, total_elems)
    rows = total_elems // cols
    remainder = total_elems - rows * cols
    if remainder > 0:
        # Absorb remainder into last row by reducing cols
        rows = total_elems // preferred_dim
        cols = preferred_dim
        actual = rows * cols
        if actual != total_elems:
            return [total_elems]
    return [rows, cols]


# ---------------------------------------------------------------------------
# File creation
# ---------------------------------------------------------------------------


def save_uncompressed(tensors: Dict[str, torch.Tensor], path: str) -> int:
    """Save tensors as a standard uncompressed safetensors file. Returns file size."""
    from safetensors.torch import save_file

    save_file(tensors, path)
    return os.path.getsize(path)


def save_compressed(
    tensors: Dict[str, torch.Tensor],
    path: str,
    compression: str = "ans",
    level: int = 3,
    chunk_size: int = 16 * 1024 * 1024,
) -> Tuple[int, float]:
    """Save tensors as a compressed safetensors file using safetensors.fast.

    Uses the installed safetensors.fast.save_compressed which supports both
    'zstd' (CPU compressed, RAW format) and 'ans' (nvcomp GPU compressed,
    NVCOMP_NATIVE format — 10x faster decompression).

    Returns (file_size, compression_ratio).
    """
    from safetensors.torch import save_file

    # Save uncompressed first to measure the body size
    uncompressed_path = path + ".tmp_uncompressed"
    save_file(tensors, uncompressed_path)
    uncompressed_size = os.path.getsize(uncompressed_path)
    with open(uncompressed_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
    decompressed_body = uncompressed_size - 8 - hlen
    os.unlink(uncompressed_path)

    # Now save compressed via safetensors.fast
    from safetensors.fast import save_compressed as _save_compressed

    _save_compressed(
        tensors,
        path,
        compression=compression,
        level=level,
        chunk_size=chunk_size,
    )

    file_size = os.path.getsize(path)
    ratio = decompressed_body / (file_size - 8 - hlen) if file_size > 8 + hlen else 1.0
    return file_size, ratio


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def drop_page_cache() -> bool:
    """Attempt to drop the OS page cache. Returns True on success."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return True
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return False


def sync_and_clear_cuda():
    """Synchronize CUDA and clear caches."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def bench_load_fast_safe_open(
    path: str,
    device: str,
    nogds: bool = False,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Load via fast_safe_open (Rust GDS path). Returns (tensors, elapsed_seconds)."""
    fast_safe_open = _try_import_fast()
    if fast_safe_open is None:
        raise RuntimeError("fast_safe_open not available")

    sync_and_clear_cuda()

    start = time.perf_counter()
    with fast_safe_open(path, device=device, nogds=nogds) as f:
        result = f.get_all_tensors()
    if "cuda" in device:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


def bench_load_file(
    path: str,
    device: str,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Load via safetensors.torch.load_file (auto-selects fast path). Returns (tensors, elapsed_seconds)."""
    load_file = _try_import_load_file()

    sync_and_clear_cuda()

    start = time.perf_counter()
    result = load_file(path, device=device)
    if "cuda" in device:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_benchmark(
    path: str,
    device: str,
    method: str,
    iters: int,
    drop_caches: bool,
    warmup: int = 1,
    nogds: bool = False,
) -> List[float]:
    """Run a loading benchmark for multiple iterations. Returns list of elapsed times."""
    times = []

    for i in range(warmup + iters):
        if drop_caches:
            if not drop_page_cache():
                if i == 0:
                    print(
                        "  ⚠ Could not drop page cache (no sudo?). Results may reflect warm cache.",
                        flush=True,
                    )
                    drop_caches = False  # Don't keep trying

        sync_and_clear_cuda()

        if method == "fast_safe_open":
            _, elapsed = bench_load_fast_safe_open(path, device, nogds=nogds)
        elif method == "fast_safe_open_nogds":
            _, elapsed = bench_load_fast_safe_open(path, device, nogds=True)
        elif method == "load_file":
            _, elapsed = bench_load_file(path, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        if i >= warmup:
            times.append(elapsed)

    return times


def percentile(data: List[float], p: float) -> float:
    """Simple percentile calculation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def format_stats(
    times: List[float], file_size_bytes: int, decompressed_bytes: int
) -> str:
    """Format timing statistics."""
    if not times:
        return "no data"

    med = percentile(times, 50)
    p10 = percentile(times, 10)
    p90 = percentile(times, 90)
    mean = sum(times) / len(times)
    throughput = (decompressed_bytes / (1024 * 1024)) / med if med > 0 else 0
    read_throughput = (file_size_bytes / (1024 * 1024)) / med if med > 0 else 0

    return (
        f"median={med * 1000:7.1f} ms  "
        f"mean={mean * 1000:7.1f} ms  "
        f"p10={p10 * 1000:7.1f} ms  "
        f"p90={p90 * 1000:7.1f} ms  "
        f"eff_throughput={throughput:,.0f} MB/s  "
        f"read_throughput={read_throughput:,.0f} MB/s"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_loaded_tensors(
    original: Dict[str, torch.Tensor],
    loaded: Dict[str, torch.Tensor],
    label: str,
) -> bool:
    """Validate that loaded tensors match originals."""
    if set(original.keys()) != set(loaded.keys()):
        print(
            f"  ✗ {label}: key mismatch! original={sorted(original.keys())[:5]}... loaded={sorted(loaded.keys())[:5]}..."
        )
        return False

    mismatches = 0
    for key in original:
        orig = original[key]
        load = loaded[key].cpu()
        if orig.shape != load.shape:
            print(
                f"  ✗ {label}: shape mismatch for '{key}': {orig.shape} vs {load.shape}"
            )
            mismatches += 1
        elif orig.dtype != load.dtype:
            print(
                f"  ✗ {label}: dtype mismatch for '{key}': {orig.dtype} vs {load.dtype}"
            )
            mismatches += 1
        elif not torch.equal(orig, load):
            max_diff = (orig.float() - load.float()).abs().max().item()
            print(f"  ✗ {label}: data mismatch for '{key}', max_diff={max_diff}")
            mismatches += 1

    if mismatches == 0:
        print(f"  ✓ {label}: all {len(original)} tensors validated OK")
        return True
    else:
        print(f"  ✗ {label}: {mismatches} tensor(s) failed validation")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GDS loading: compressed vs uncompressed"
    )
    parser.add_argument(
        "--size", type=float, default=500, help="Model size in MB (default: 500)"
    )
    parser.add_argument(
        "--iters", type=int, default=7, help="Benchmark iterations (default: 7)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations (default: 1)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Target device (default: cuda:0)"
    )
    parser.add_argument(
        "--drop-caches",
        action="store_true",
        help="Drop page cache between iterations (needs sudo)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Tensor dtype (default: float16)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="ans",
        choices=["ans", "zstd"],
        help="Compression algorithm: 'ans' (10x faster GPU decomp) or 'zstd' (default: ans)",
    )
    parser.add_argument(
        "--level", type=int, default=3, help="Zstd compression level (default: 3)"
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=16,
        help="Compression chunk size in MiB (default: 16)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=4096,
        help="Hidden dimension for weight shapes (default: 4096)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate loaded tensors match originals",
    )
    parser.add_argument(
        "--tmpdir",
        type=str,
        default=None,
        help="Directory for temp files (default: system temp)",
    )
    parser.add_argument(
        "--skip-mmap", action="store_true", help="Skip the mmap baseline benchmark"
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("  GDS Compressed vs Uncompressed Safetensors Benchmark")
    print("=" * 80)

    # System info
    print(f"\n{'Device:':<24} {args.device}")
    if torch.cuda.is_available():
        dev_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        print(f"{'GPU:':<24} {torch.cuda.get_device_name(dev_idx)}")
        print(
            f"{'GPU Memory:':<24} {torch.cuda.get_device_properties(dev_idx).total_memory / 1e9:.1f} GB"
        )
    print(f"{'Dtype:':<24} {args.dtype}")
    print(f"{'Target size:':<24} {args.size:.0f} MB")
    print(f"{'Iterations:':<24} {args.iters} (+{args.warmup} warmup)")
    print(f"{'Drop caches:':<24} {args.drop_caches}")
    print(f"{'Compression:':<24} {args.compression.upper()}")
    if args.compression == "zstd":
        print(f"{'Zstd level:':<24} {args.level}")
    print(f"{'Chunk size:':<24} {args.chunk_size_mb} MiB")

    fast_safe_open = _try_import_fast()
    print(
        f"{'fast_safe_open:':<24} {'available' if fast_safe_open else 'NOT available'}"
    )

    # Check GDS availability
    gds_available = False
    if fast_safe_open and torch.cuda.is_available():
        try:
            # Quick probe: open a tiny file to see if GDS initializes
            from safetensors.torch import save_file

            probe_path = os.path.join(tempfile.gettempdir(), "_gds_probe.safetensors")
            save_file({"x": torch.zeros(1)}, probe_path)
            with fast_safe_open(probe_path, device=args.device, nogds=False) as f:
                _ = f.get_all_tensors()
            os.unlink(probe_path)
            gds_available = True
        except Exception:
            pass
    print(
        f"{'GDS:':<24} {'available' if gds_available else 'not available (will use pread+bounce)'}"
    )

    # Generate model
    print(f"\n--- Generating {args.size:.0f} MB random model ({args.dtype}) ---")
    t0 = time.perf_counter()
    tensors, actual_mb = generate_random_tensors(
        args.size, dtype=dtype, hidden=args.hidden
    )
    gen_time = time.perf_counter() - t0
    n_tensors = len(tensors)
    print(f"  Generated {n_tensors} tensors, {actual_mb:.1f} MB in {gen_time:.2f}s")

    # Save files
    tmpdir = args.tmpdir or tempfile.gettempdir()
    uncompressed_path = os.path.join(tmpdir, "bench_uncompressed.safetensors")
    compressed_path = os.path.join(tmpdir, "bench_compressed.safetensors")

    print(f"\n--- Saving uncompressed file ---")
    t0 = time.perf_counter()
    uncompressed_size = save_uncompressed(tensors, uncompressed_path)
    save_time = time.perf_counter() - t0
    print(f"  {uncompressed_path}")
    print(f"  Size: {uncompressed_size / 1e6:.2f} MB, Time: {save_time:.2f}s")

    print(f"\n--- Saving compressed file ({args.compression.upper()}) ---")
    t0 = time.perf_counter()
    compressed_size, ratio = save_compressed(
        tensors,
        compressed_path,
        compression=args.compression,
        level=args.level,
        chunk_size=args.chunk_size_mb * 1024 * 1024,
    )
    compress_time = time.perf_counter() - t0
    print(f"  {compressed_path}")
    print(
        f"  Size: {compressed_size / 1e6:.2f} MB (ratio: {ratio:.2f}x), Time: {compress_time:.2f}s"
    )
    print(f"  Savings: {(1 - compressed_size / uncompressed_size) * 100:.1f}%")

    # Parse header to get decompressed body size
    with open(uncompressed_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
    decompressed_body_bytes = uncompressed_size - 8 - header_len

    # Validate
    if args.validate:
        print(f"\n--- Validating correctness ---")

        if fast_safe_open:
            loaded_unc, _ = bench_load_fast_safe_open(uncompressed_path, args.device)
            validate_loaded_tensors(tensors, loaded_unc, "uncompressed/fast_safe_open")
            del loaded_unc
            sync_and_clear_cuda()

        loaded_lf, _ = bench_load_file(uncompressed_path, args.device)
        validate_loaded_tensors(tensors, loaded_lf, "uncompressed/load_file")
        del loaded_lf
        sync_and_clear_cuda()

        # Compressed files must go through fast_safe_open (mmap path can't handle compressed body)
        if fast_safe_open:
            loaded_comp, _ = bench_load_fast_safe_open(compressed_path, args.device)
            validate_loaded_tensors(tensors, loaded_comp, "compressed/fast_safe_open")
            del loaded_comp
            sync_and_clear_cuda()
        else:
            print("  ⚠ Skipping compressed validation (fast_safe_open not available)")

    # Benchmark
    print(f"\n{'=' * 80}")
    print(
        f"  BENCHMARK RESULTS  ({args.iters} iterations, {'cold NVMe' if args.drop_caches else 'warm cache'})"
    )
    print(f"{'=' * 80}")

    results = {}

    algo = args.compression.upper()

    # 1) Uncompressed via fast_safe_open (GDS when available)
    if fast_safe_open:
        print(
            f"\n[1/5] Uncompressed + fast_safe_open (GDS={'enabled' if gds_available else 'disabled'})..."
        )
        times = run_benchmark(
            uncompressed_path,
            args.device,
            "fast_safe_open",
            args.iters,
            args.drop_caches,
            warmup=args.warmup,
        )
        results["uncompressed_gds"] = times
        print(f"  {format_stats(times, uncompressed_size, decompressed_body_bytes)}")

    # 2) Uncompressed via fast_safe_open with GDS disabled (pread fallback)
    if fast_safe_open:
        print(f"\n[2/5] Uncompressed + fast_safe_open (GDS=disabled, pread+bounce)...")
        times = run_benchmark(
            uncompressed_path,
            args.device,
            "fast_safe_open_nogds",
            args.iters,
            args.drop_caches,
            warmup=args.warmup,
            nogds=True,
        )
        results["uncompressed_pread"] = times
        print(f"  {format_stats(times, uncompressed_size, decompressed_body_bytes)}")

    # 3) Compressed via fast_safe_open (GDS when available + nvcomp GPU decompress)
    if fast_safe_open:
        print(
            f"\n[3/5] Compressed ({algo} {ratio:.2f}x) + fast_safe_open (GDS={'enabled' if gds_available else 'disabled'})..."
        )
        times = run_benchmark(
            compressed_path,
            args.device,
            "fast_safe_open",
            args.iters,
            args.drop_caches,
            warmup=args.warmup,
        )
        results["compressed_gds"] = times
        print(f"  {format_stats(times, compressed_size, decompressed_body_bytes)}")

        # 3b) Compressed via fast_safe_open with GDS disabled
        print(
            f"\n[3b/5] Compressed ({algo} {ratio:.2f}x) + fast_safe_open (GDS=disabled, pread+bounce)..."
        )
        times = run_benchmark(
            compressed_path,
            args.device,
            "fast_safe_open_nogds",
            args.iters,
            args.drop_caches,
            warmup=args.warmup,
            nogds=True,
        )
        results["compressed_pread"] = times
        print(f"  {format_stats(times, compressed_size, decompressed_body_bytes)}")
    else:
        print(f"\n[3/5] Skipped (fast_safe_open not available for compressed files)")

    # 4) Uncompressed via load_file (auto fast path)
    print(f"\n[4/5] Uncompressed + load_file (auto)...")
    times = run_benchmark(
        uncompressed_path,
        args.device,
        "load_file",
        args.iters,
        args.drop_caches,
        warmup=args.warmup,
    )
    results["uncompressed_auto"] = times
    print(f"  {format_stats(times, uncompressed_size, decompressed_body_bytes)}")

    # 5) Mmap baseline
    if not args.skip_mmap:
        print(f"\n[5/5] Uncompressed + mmap baseline (SAFETENSORS_FAST_GPU=0)...")
        old_val = os.environ.get("SAFETENSORS_FAST_GPU")
        os.environ["SAFETENSORS_FAST_GPU"] = "0"
        # Need to reload the module to pick up the env change
        if "safetensors.torch" in sys.modules:
            import importlib

            import safetensors.torch

            importlib.reload(safetensors.torch)
        times = run_benchmark(
            uncompressed_path,
            args.device,
            "load_file",
            args.iters,
            args.drop_caches,
            warmup=args.warmup,
        )
        results["mmap_baseline"] = times
        if old_val is not None:
            os.environ["SAFETENSORS_FAST_GPU"] = old_val
        else:
            os.environ.pop("SAFETENSORS_FAST_GPU", None)
        if "safetensors.torch" in sys.modules:
            import importlib

            import safetensors.torch

            importlib.reload(safetensors.torch)
        print(f"  {format_stats(times, uncompressed_size, decompressed_body_bytes)}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  File sizes:")
    print(f"    Uncompressed: {uncompressed_size / 1e6:>10.2f} MB")
    print(
        f"    Compressed:   {compressed_size / 1e6:>10.2f} MB  ({ratio:.2f}x ratio, {(1 - compressed_size / uncompressed_size) * 100:.1f}% smaller)"
    )

    print(f"\n  {'Method':<48} {'Median':>10} {'Eff MB/s':>10} {'vs best':>10}")
    print(f"  {'-' * 48} {'-' * 10} {'-' * 10} {'-' * 10}")

    best_time = float("inf")
    for label, times in results.items():
        if times:
            med = percentile(times, 50)
            best_time = min(best_time, med)

    for label, times in results.items():
        if not times:
            continue
        med = percentile(times, 50)
        throughput = (decompressed_body_bytes / (1024 * 1024)) / med if med > 0 else 0
        speedup = med / best_time if best_time > 0 else 0

        display_name = {
            "uncompressed_gds": f"Uncompressed + GDS",
            "uncompressed_pread": f"Uncompressed + pread (no GDS)",
            "compressed_gds": f"Compressed ({algo} {ratio:.2f}x) + GDS + nvcomp",
            "compressed_pread": f"Compressed ({algo} {ratio:.2f}x) + pread",
            "uncompressed_auto": f"Uncompressed + auto (load_file)",
            "mmap_baseline": f"Uncompressed + mmap baseline",
        }.get(label, label)

        marker = " ← fastest" if abs(med - best_time) < 1e-6 else ""
        print(
            f"  {display_name:<48} {med * 1000:>8.1f}ms {throughput:>8.0f} {1 / speedup:>9.2f}x{marker}"
        )

    # Speedup of compressed over best uncompressed
    comp_key = "compressed_gds" if "compressed_gds" in results else "compressed_pread"
    if comp_key in results and results[comp_key]:
        comp_med = percentile(results[comp_key], 50)
        uncomp_times = []
        for k in ["uncompressed_gds", "uncompressed_auto"]:
            if k in results and results[k]:
                uncomp_times.extend(results[k])
        if uncomp_times:
            uncomp_med = percentile(uncomp_times, 50)
            if comp_med > 0:
                print(f"\n  Compressed vs best uncompressed: ", end="")
                if comp_med < uncomp_med:
                    print(f"{uncomp_med / comp_med:.2f}x FASTER ✓")
                else:
                    print(f"{comp_med / uncomp_med:.2f}x slower ✗")

    # Cleanup
    print(f"\n--- Cleaning up ---")
    for p in [uncompressed_path, compressed_path]:
        try:
            os.unlink(p)
            print(f"  Removed {p}")
        except OSError:
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
