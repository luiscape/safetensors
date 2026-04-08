#!/usr/bin/env python3
"""Standalone GPU Direct Storage (GDS) throughput benchmark.

This script measures the raw throughput of cuFileRead independently of the
safetensors implementation.  It serves as a **ceiling** measurement: the
maximum read bandwidth that cuFile can deliver on this system for a single
NVMe → GPU transfer.

Usage::

    python bench_gds_raw.py [--size-mb 2048] [--device 0] [--path /data] [--no-drop-cache]

Requirements:
    - libcufile.so  (CUDA GDS / cuFile)
    - libcudart.so  (CUDA runtime)
    - PyTorch        (for GPU memory allocation)
    - nvidia_fs kernel module loaded
    - NVMe storage on a GDS-compatible filesystem (ext4 / xfs with O_DIRECT)
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# cuFile FFI
# ---------------------------------------------------------------------------


class CUfileError(ctypes.Structure):
    """Mirrors ``CUfileError_t`` — two 32-bit error codes."""

    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


class CUfileDescr(ctypes.Structure):
    """Mirrors ``CUfileDescr_t`` on Linux x86-64.

    Layout::

        int32   type          (CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1)
        int32   _pad          (alignment padding for the union)
        int64   handle.fd     (pointer-sized union; fd widened)
        void*   fs_ops        (NULL for local filesystems)
    """

    _fields_ = [
        ("handle_type", ctypes.c_int),
        ("_pad", ctypes.c_int),
        ("handle_fd", ctypes.c_int64),
        ("fs_ops", ctypes.c_void_p),
    ]


CUfileHandle = ctypes.c_void_p  # opaque void*


class CuFile:
    """Thin wrapper around the cuFile (GDS) C API loaded at runtime."""

    def __init__(self) -> None:
        self._lib = ctypes.CDLL("libcufile.so.0")
        self._setup_signatures()
        ret = self._lib.cuFileDriverOpen()
        if ret.err != 0:
            raise RuntimeError(
                f"cuFileDriverOpen failed: err={ret.err}, cu_err={ret.cu_err}"
            )

    # ---- signature setup --------------------------------------------------

    def _setup_signatures(self) -> None:
        L = self._lib

        L.cuFileDriverOpen.restype = CUfileError
        L.cuFileDriverOpen.argtypes = []

        L.cuFileDriverClose.restype = CUfileError
        L.cuFileDriverClose.argtypes = []

        L.cuFileBufRegister.restype = CUfileError
        L.cuFileBufRegister.argtypes = [
            ctypes.c_void_p,  # bufPtr_base
            ctypes.c_size_t,  # length
            ctypes.c_int,  # flags
        ]

        L.cuFileBufDeregister.restype = CUfileError
        L.cuFileBufDeregister.argtypes = [ctypes.c_void_p]

        L.cuFileHandleRegister.restype = CUfileError
        L.cuFileHandleRegister.argtypes = [
            ctypes.POINTER(CUfileHandle),  # *fh (output)
            ctypes.POINTER(CUfileDescr),  # *descr
        ]

        L.cuFileHandleDeregister.restype = None
        L.cuFileHandleDeregister.argtypes = [CUfileHandle]

        L.cuFileRead.restype = ctypes.c_ssize_t
        L.cuFileRead.argtypes = [
            CUfileHandle,  # fh
            ctypes.c_void_p,  # bufPtr_base
            ctypes.c_size_t,  # size
            ctypes.c_int64,  # file_offset
            ctypes.c_int64,  # bufPtr_offset
        ]

        # Optional: cuFileGetVersion
        try:
            L.cuFileGetVersion.restype = CUfileError
            L.cuFileGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        except AttributeError:
            pass

    # ---- high-level helpers -----------------------------------------------

    def version(self) -> Optional[int]:
        ver = ctypes.c_int(0)
        try:
            ret = self._lib.cuFileGetVersion(ctypes.byref(ver))
            if ret.err == 0:
                return ver.value
        except Exception:
            pass
        return None

    def register_buffer(self, dev_ptr: int, size: int) -> None:
        ret = self._lib.cuFileBufRegister(dev_ptr, size, 0)
        if ret.err != 0:
            raise RuntimeError(
                f"cuFileBufRegister failed: err={ret.err}, cu_err={ret.cu_err}"
            )

    def deregister_buffer(self, dev_ptr: int) -> None:
        ret = self._lib.cuFileBufDeregister(dev_ptr)
        if ret.err != 0:
            raise RuntimeError(
                f"cuFileBufDeregister failed: err={ret.err}, cu_err={ret.cu_err}"
            )

    def register_handle(self, fd: int) -> CUfileHandle:
        fh = CUfileHandle()
        descr = CUfileDescr()
        descr.handle_type = 1  # CU_FILE_HANDLE_TYPE_OPAQUE_FD
        descr._pad = 0
        descr.handle_fd = fd
        descr.fs_ops = None
        ret = self._lib.cuFileHandleRegister(ctypes.byref(fh), ctypes.byref(descr))
        if ret.err != 0:
            raise RuntimeError(
                f"cuFileHandleRegister failed: err={ret.err}, cu_err={ret.cu_err}"
            )
        return fh

    def deregister_handle(self, fh: CUfileHandle) -> None:
        self._lib.cuFileHandleDeregister(fh)

    def read(
        self,
        fh: CUfileHandle,
        dev_ptr: int,
        size: int,
        file_offset: int,
        buf_offset: int = 0,
    ) -> int:
        """Issue a single cuFileRead.  Returns bytes read (or raises)."""
        n = self._lib.cuFileRead(fh, dev_ptr, size, file_offset, buf_offset)
        if n < 0:
            raise RuntimeError(f"cuFileRead returned {n} (errno={ctypes.get_errno()})")
        return n

    def close(self) -> None:
        self._lib.cuFileDriverClose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALIGN = 4096  # GDS alignment


def create_test_file(path: str, size_bytes: int) -> str:
    """Write a file of *size_bytes* filled with deterministic data."""
    # Use O_DIRECT to bypass page cache during creation
    fpath = os.path.join(path, "gds_bench_raw.bin")
    print(
        f"  Creating test file: {fpath} ({size_bytes / (1024**2):.0f} MB) ...",
        end=" ",
        flush=True,
    )
    t0 = time.perf_counter()

    # Ensure size is ALIGN-aligned
    assert size_bytes % ALIGN == 0, f"size_bytes ({size_bytes}) must be {ALIGN}-aligned"

    fd = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        chunk = ALIGN * 256  # 1 MB write chunks
        buf = bytearray(chunk)
        # Fill with a pattern so we can verify later
        pattern = b"\xab\xcd\xef\x01"
        for i in range(0, chunk, len(pattern)):
            end = min(i + len(pattern), chunk)
            buf[i:end] = pattern[: end - i]
        # Align the buffer for O_DIRECT
        import mmap as _mmap

        aligned = _mmap.mmap(-1, chunk)
        aligned[:] = bytes(buf)

        written = 0
        while written < size_bytes:
            to_write = min(chunk, size_bytes - written)
            if to_write < chunk:
                small_aligned = _mmap.mmap(-1, to_write)
                small_aligned[:] = bytes(buf[:to_write])
                os.write(fd, small_aligned[:to_write])
                small_aligned.close()
            else:
                os.write(fd, aligned[:to_write])
            written += to_write
    finally:
        os.close(fd)

    elapsed = time.perf_counter() - t0
    tp = size_bytes / (1024**2) / elapsed
    print(f"done ({elapsed:.1f}s, {tp:.0f} MB/s write)")
    return fpath


def drop_caches() -> None:
    os.sync()
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except PermissionError:
        os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1")
    time.sleep(0.3)


def get_nvidia_fs_stats() -> dict[str, str]:
    """Parse /proc/driver/nvidia-fs/stats into a dict."""
    result = {}
    try:
        with open("/proc/driver/nvidia-fs/stats") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    result[key.strip()] = val.strip()
    except FileNotFoundError:
        pass
    return result


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_cufile_read(
    cufile: CuFile,
    fpath: str,
    dev_ptr: int,
    buf_size: int,
    file_size: int,
    chunk_size: int,
    n_iters: int = 3,
    do_drop_cache: bool = True,
    label: str = "",
) -> float:
    """Benchmark cuFileRead and return mean throughput in MB/s."""
    import torch

    fd = os.open(fpath, os.O_RDONLY | os.O_DIRECT)
    fh = cufile.register_handle(fd)
    cufile.register_buffer(dev_ptr, buf_size)

    size_mb = file_size / (1024**2)
    times: list[float] = []

    try:
        for i in range(n_iters):
            if do_drop_cache:
                drop_caches()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Read the whole file in chunk_size pieces
            offset = 0
            while offset < file_size:
                to_read = min(chunk_size, file_size - offset)
                n = cufile.read(fh, dev_ptr, to_read, offset, offset)
                if n == 0:
                    break
                offset += n

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    finally:
        cufile.deregister_buffer(dev_ptr)
        cufile.deregister_handle(fh)
        os.close(fd)

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    tp = size_mb / mean_t
    tag = f" ({label})" if label else ""
    print(
        f"  cuFileRead{tag:30s}  {mean_t * 1000:8.1f} ms ± {std_t * 1000:5.1f} ms"
        f"  {tp:7.0f} MB/s"
    )
    return tp


def bench_pread_bounce(
    fpath: str,
    dev_ptr_unused: int,
    device: str,
    file_size: int,
    bounce_size: int,
    n_iters: int = 3,
    do_drop_cache: bool = True,
) -> float:
    """Benchmark pread + cudaMemcpy bounce-buffer path for comparison."""
    import torch

    size_mb = file_size / (1024**2)
    times: list[float] = []

    gpu_buf = torch.empty(file_size, dtype=torch.uint8, device=device)
    staging = torch.empty(bounce_size, dtype=torch.uint8, pin_memory=True)
    staging_np = staging.numpy()

    for i in range(n_iters):
        if do_drop_cache:
            drop_caches()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        fd = os.open(fpath, os.O_RDONLY)
        offset = 0
        try:
            while offset < file_size:
                to_read = min(bounce_size, file_size - offset)
                view = memoryview(staging_np)[:to_read]
                n = os.pread(fd, to_read, offset)
                staging_np[: len(n)] = bytearray(n)
                gpu_buf[offset : offset + len(n)].copy_(
                    staging[: len(n)], non_blocking=True
                )
                offset += len(n)
        finally:
            os.close(fd)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del gpu_buf, staging

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    tp = size_mb / mean_t
    print(
        f"  pread+cudaMemcpy (bounce)         {mean_t * 1000:8.1f} ms ± {std_t * 1000:5.1f} ms"
        f"  {tp:7.0f} MB/s"
    )
    return tp


def bench_mmap_per_tensor(
    fpath: str,
    device: str,
    file_size: int,
    n_tensors: int,
    n_iters: int = 3,
    do_drop_cache: bool = True,
) -> float:
    """Benchmark the mmap + per-tensor copy path (baseline safetensors)."""
    import mmap as _mmap

    import torch

    size_mb = file_size / (1024**2)
    times: list[float] = []
    tensor_size = file_size // n_tensors

    for i in range(n_iters):
        if do_drop_cache:
            drop_caches()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        fd = os.open(fpath, os.O_RDONLY)
        try:
            mm = _mmap.mmap(fd, 0, access=_mmap.ACCESS_READ)
            # Simulate safetensors: mmap the file, then copy each tensor slice
            tensors = []
            for j in range(n_tensors):
                start = j * tensor_size
                end = start + tensor_size
                data = mm[start:end]
                cpu_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                gpu_t = cpu_t.to(device=device, non_blocking=True)
                tensors.append(gpu_t)
            mm.close()
        finally:
            os.close(fd)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del tensors

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    tp = size_mb / mean_t
    print(
        f"  mmap + per-tensor H2D (×{n_tensors:3d})      {mean_t * 1000:8.1f} ms ± {std_t * 1000:5.1f} ms"
        f"  {tp:7.0f} MB/s"
    )
    return tp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw GDS throughput benchmark")
    parser.add_argument(
        "--size-mb", type=int, default=2048, help="Test file size in MB"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--path", type=str, default="/data", help="Directory on NVMe for test file"
    )
    parser.add_argument(
        "--no-drop-cache", action="store_true", help="Skip dropping page cache"
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="Number of iterations per benchmark"
    )
    args = parser.parse_args()

    import torch

    torch.cuda.init()
    device_str = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)

    size_bytes = args.size_mb * 1024 * 1024
    # Round up to ALIGN
    size_bytes = ((size_bytes + ALIGN - 1) // ALIGN) * ALIGN
    size_mb = size_bytes / (1024**2)

    print("=" * 72)
    print("Raw GDS Throughput Benchmark")
    print("=" * 72)
    print(f"  GPU:        {torch.cuda.get_device_name(args.device)}")
    print(f"  Device:     cuda:{args.device}")
    print(f"  NVMe path:  {args.path}")
    print(f"  File size:  {size_mb:.0f} MB")
    print(f"  Iterations: {args.iters}")
    print(f"  Drop cache: {not args.no_drop_cache}")
    print()

    # ---- Initialize cuFile ------------------------------------------------
    print("Initializing cuFile...")
    cufile = CuFile()
    ver = cufile.version()
    if ver:
        print(f"  cuFile version: {ver // 1000}.{(ver % 1000) // 10}.{ver % 10}")

    # ---- Check GDS support ------------------------------------------------
    try:
        cudart = ctypes.CDLL("libcudart.so")
        val = ctypes.c_int(0)
        cudart.cudaDeviceGetAttribute(
            ctypes.byref(val), 96, args.device
        )  # 96 = GPUDirectRDMASupported
        print(f"  GPUDirect RDMA supported: {bool(val.value)}")
        dver = ctypes.c_int(0)
        cudart.cudaDriverGetVersion(ctypes.byref(dver))
        print(
            f"  CUDA driver version: {dver.value // 1000}.{(dver.value % 1000) // 10}"
        )
    except Exception as e:
        print(f"  CUDA runtime check: {e}")
    print()

    # ---- Print pre-test nvidia-fs stats -----------------------------------
    stats_before = get_nvidia_fs_stats()
    print(f"  nvidia-fs Reads (before): {stats_before.get('Reads', 'N/A')}")
    print(f"  nvidia-fs Ops (before):   {stats_before.get('Ops', 'N/A')}")
    print()

    # ---- Create test file -------------------------------------------------
    bench_dir = os.path.join(args.path, "gds_bench_tmp")
    os.makedirs(bench_dir, exist_ok=True)
    fpath = create_test_file(bench_dir, size_bytes)
    print()

    # ---- Allocate GPU buffer ----------------------------------------------
    gpu_buf = torch.empty(size_bytes, dtype=torch.uint8, device=device_str)
    dev_ptr = gpu_buf.data_ptr()
    print(f"  GPU buffer: 0x{dev_ptr:x}, {size_bytes} bytes")
    print(f"  Buffer aligned to {ALIGN}: {dev_ptr % ALIGN == 0}")
    print()

    do_drop = not args.no_drop_cache

    # ---- Benchmark 1: GDS single large read -------------------------------
    print("-" * 72)
    print("Benchmark 1: cuFileRead — single large transfer")
    print("-" * 72)
    bench_cufile_read(
        cufile,
        fpath,
        dev_ptr,
        size_bytes,
        size_bytes,
        chunk_size=size_bytes,  # one shot
        n_iters=args.iters,
        do_drop_cache=do_drop,
        label="1 chunk",
    )
    print()

    # ---- Benchmark 2: GDS chunked reads (1 GiB chunks) -------------------
    print("-" * 72)
    print("Benchmark 2: cuFileRead — 1 GiB chunks")
    print("-" * 72)
    bench_cufile_read(
        cufile,
        fpath,
        dev_ptr,
        size_bytes,
        size_bytes,
        chunk_size=1 << 30,
        n_iters=args.iters,
        do_drop_cache=do_drop,
        label="1 GiB chunks",
    )
    print()

    # ---- Benchmark 3: GDS chunked reads (16 MiB chunks) ------------------
    print("-" * 72)
    print("Benchmark 3: cuFileRead — 16 MiB chunks")
    print("-" * 72)
    bench_cufile_read(
        cufile,
        fpath,
        dev_ptr,
        size_bytes,
        size_bytes,
        chunk_size=16 * 1024 * 1024,
        n_iters=args.iters,
        do_drop_cache=do_drop,
        label="16 MiB chunks",
    )
    print()

    # ---- Benchmark 4: pread + bounce buffer (for comparison) --------------
    print("-" * 72)
    print("Benchmark 4: pread + pinned bounce buffer + cudaMemcpy")
    print("-" * 72)
    del gpu_buf
    torch.cuda.empty_cache()
    bench_pread_bounce(
        fpath,
        dev_ptr,
        device_str,
        size_bytes,
        bounce_size=16 * 1024 * 1024,
        n_iters=args.iters,
        do_drop_cache=do_drop,
    )
    print()

    # ---- Benchmark 5: mmap + per-tensor copy (safetensors baseline) -------
    print("-" * 72)
    print("Benchmark 5: mmap + per-tensor H2D copy (safetensors baseline)")
    print("-" * 72)
    n_tensors = 100  # simulate ~100 tensors
    bench_mmap_per_tensor(
        fpath,
        device_str,
        size_bytes,
        n_tensors,
        n_iters=args.iters,
        do_drop_cache=do_drop,
    )
    print()

    # ---- Post-test nvidia-fs stats ----------------------------------------
    print("-" * 72)
    print("nvidia-fs stats (after all benchmarks)")
    print("-" * 72)
    stats_after = get_nvidia_fs_stats()
    for key in [
        "Reads",
        "Sparse Reads",
        "Ops",
        "Error",
        "Active Shadow-Buffer (MiB)",
        "Active Process",
    ]:
        val = stats_after.get(key, "N/A")
        print(f"  {key}: {val}")
    print()

    # ---- Cleanup ----------------------------------------------------------
    cufile.close()
    os.unlink(fpath)
    try:
        os.rmdir(bench_dir)
    except OSError:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
