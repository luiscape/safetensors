#!/usr/bin/env python3
"""Test ANS compress→decompress roundtrip via the C shim.

Usage:
    LD_LIBRARY_PATH=/tmp/nvcomp_install/nvidia/libnvcomp/lib64:/usr/local/cuda/lib64 \
    python3.12 benches/test_ans_shim.py
"""

import ctypes as C
import os
import random
import sys
import time


def main():
    # Load libraries
    shim_path = os.environ.get("NVCOMP_SHIM_PATH", "/tmp/libnvcomp_shim.so")
    if not os.path.exists(shim_path):
        print(f"Shim not found at {shim_path}")
        print("Build it with:")
        print("  gcc -shared -fPIC -O2 -o /tmp/libnvcomp_shim.so src/nvcomp_shim.c \\")
        print("      -L/tmp/nvcomp_install/nvidia/libnvcomp/lib64 -lnvcomp \\")
        print("      -Wl,-rpath,/tmp/nvcomp_install/nvidia/libnvcomp/lib64")
        sys.exit(1)

    shim = C.CDLL(shim_path)
    cuda = C.CDLL("libcudart.so")
    cuda.cudaSetDevice(0)

    # ---- helpers ----

    def cuda_malloc(n):
        p = C.c_void_p()
        s = cuda.cudaMalloc(C.byref(p), C.c_size_t(n))
        assert s == 0, f"cudaMalloc({n}) failed: {s}"
        return p

    def h2d(dst, src_bytes, n):
        buf = (C.c_char * n).from_buffer_copy(src_bytes)
        s = cuda.cudaMemcpy(dst, buf, C.c_size_t(n), 1)
        assert s == 0, f"cudaMemcpy H2D failed: {s}"

    def d2h(dst_buf, src, n):
        s = cuda.cudaMemcpy(dst_buf, src, C.c_size_t(n), 2)
        assert s == 0, f"cudaMemcpy D2H failed: {s}"

    def sync():
        cuda.cudaDeviceSynchronize()

    def make_device_array_ptrs(ptrs):
        """Upload a list of c_void_p values as a device array of pointers."""
        n = len(ptrs)
        host_arr = (C.c_void_p * n)(*ptrs)
        dev = cuda_malloc(n * 8)
        h2d(dev, bytes(host_arr), n * 8)
        return dev

    def make_device_array_sizes(sizes):
        """Upload a list of ints as a device array of size_t."""
        n = len(sizes)
        host_arr = (C.c_size_t * n)(*sizes)
        dev = cuda_malloc(n * 8)
        h2d(dev, bytes(host_arr), n * 8)
        return dev

    # ---- opts structs ----

    class CompressOpts(C.Structure):
        _fields_ = [("data", C.c_char * 64)]

    class DecompressOpts(C.Structure):
        _fields_ = [("backend", C.c_int), ("reserved", C.c_char * 60)]

    compress_opts = CompressOpts()
    C.memset(C.byref(compress_opts), 0, 64)

    decompress_opts = DecompressOpts(backend=0, reserved=b"\x00" * 60)

    # ==================================================================
    # Test 1: Small roundtrip (4 MB, single chunk)
    # ==================================================================
    print("=" * 60)
    print("Test 1: ANS roundtrip — 4 MB single chunk")
    print("=" * 60)

    random.seed(42)
    N = 4 * 1024 * 1024
    original = bytes(random.getrandbits(8) for _ in range(N))

    src_gpu = cuda_malloc(N)
    h2d(src_gpu, original, N)

    # Get max output chunk size
    max_out = C.c_size_t(0)
    s = shim.nvcomp_shim_ans_compress_get_max_output_chunk_size(
        C.c_size_t(N), C.byref(compress_opts), C.byref(max_out)
    )
    print(f"  MaxOutputChunkSize: status={s}, max={max_out.value}")
    assert s == 0, f"GetMaxOutputChunkSize failed: {s}"

    # Get compress temp size
    comp_temp_bytes = C.c_size_t(0)
    s = shim.nvcomp_shim_ans_compress_get_temp_size(
        C.c_size_t(1),
        C.c_size_t(N),
        C.byref(compress_opts),
        C.byref(comp_temp_bytes),
        C.c_size_t(N),
    )
    print(f"  CompressGetTempSize: status={s}, temp={comp_temp_bytes.value}")
    assert s == 0, f"CompressGetTempSize failed: {s}"

    comp_gpu = cuda_malloc(max_out.value)
    comp_temp_gpu = cuda_malloc(comp_temp_bytes.value)

    # Build device arrays for 1-chunk compress
    d_uncomp_ptrs = make_device_array_ptrs([src_gpu])
    d_uncomp_sizes = make_device_array_sizes([N])
    d_comp_ptrs = make_device_array_ptrs([comp_gpu])
    d_comp_sizes = make_device_array_sizes([0])
    sync()

    # Compress
    s = shim.nvcomp_shim_ans_compress_async(
        d_uncomp_ptrs,
        d_uncomp_sizes,
        C.c_size_t(N),
        C.c_size_t(1),
        comp_temp_gpu,
        C.c_size_t(comp_temp_bytes.value),
        d_comp_ptrs,
        d_comp_sizes,
        C.byref(compress_opts),
        C.c_void_p(0),
        C.c_void_p(0),
    )
    print(f"  CompressAsync: status={s}")
    assert s == 0, f"CompressAsync failed: {s}"
    sync()

    # Read back compressed size
    actual_comp_size = (C.c_size_t * 1)(0)
    d2h(actual_comp_size, d_comp_sizes, 8)
    comp_bytes = actual_comp_size[0]
    ratio = N / comp_bytes if comp_bytes > 0 else 0
    print(f"  Compressed: {N} -> {comp_bytes} bytes ({ratio:.2f}x)")

    # Get decompress temp size
    decomp_temp_bytes = C.c_size_t(0)
    s = shim.nvcomp_shim_ans_decompress_get_temp_size(
        C.c_size_t(1),
        C.c_size_t(N),
        C.byref(decompress_opts),
        C.byref(decomp_temp_bytes),
        C.c_size_t(N),
    )
    print(f"  DecompGetTempSize: status={s}, temp={decomp_temp_bytes.value}")
    assert s == 0, f"DecompGetTempSize failed: {s}"

    dst_gpu = cuda_malloc(N)
    decomp_temp_gpu = cuda_malloc(decomp_temp_bytes.value)

    # Build device arrays for 1-chunk decompress
    d_dec_comp_ptrs = make_device_array_ptrs([comp_gpu])
    d_dec_comp_sizes = make_device_array_sizes([comp_bytes])
    d_dec_uncomp_ptrs = make_device_array_ptrs([dst_gpu])
    d_dec_buf_sizes = make_device_array_sizes([N])
    sync()

    # Decompress
    s = shim.nvcomp_shim_ans_decompress_async(
        d_dec_comp_ptrs,
        d_dec_comp_sizes,
        d_dec_buf_sizes,
        C.c_void_p(0),  # actual sizes output (NULL)
        C.c_size_t(1),
        decomp_temp_gpu,
        C.c_size_t(decomp_temp_bytes.value),
        d_dec_uncomp_ptrs,
        C.byref(decompress_opts),
        C.c_void_p(0),  # statuses (NULL)
        C.c_void_p(0),  # default stream
    )
    print(f"  DecompressAsync: status={s}")
    assert s == 0, f"DecompressAsync failed: {s}"
    sync()

    # Verify
    result_buf = (C.c_char * N)()
    d2h(result_buf, dst_gpu, N)
    result = bytes(result_buf)
    match = result == original
    print(f"  MATCH: {match}")
    assert match, "Data mismatch in 4MB roundtrip!"

    # ==================================================================
    # Test 2: Multi-chunk roundtrip (128 MB, 16 MB chunks)
    # ==================================================================
    print()
    print("=" * 60)
    print("Test 2: ANS roundtrip — 128 MB in 16 MB chunks")
    print("=" * 60)

    TOTAL = 128 * 1024 * 1024
    CHUNK = 16 * 1024 * 1024
    NUM_CHUNKS = TOTAL // CHUNK

    random.seed(123)
    big_data = bytes(random.getrandbits(8) for _ in range(TOTAL))
    big_gpu = cuda_malloc(TOTAL)
    h2d(big_gpu, big_data, TOTAL)

    # Get max output per chunk
    max_out2 = C.c_size_t(0)
    shim.nvcomp_shim_ans_compress_get_max_output_chunk_size(
        C.c_size_t(CHUNK), C.byref(compress_opts), C.byref(max_out2)
    )

    # Get compress temp (for all chunks at once)
    ct2 = C.c_size_t(0)
    s = shim.nvcomp_shim_ans_compress_get_temp_size(
        C.c_size_t(NUM_CHUNKS),
        C.c_size_t(CHUNK),
        C.byref(compress_opts),
        C.byref(ct2),
        C.c_size_t(TOTAL),
    )
    assert s == 0, f"CompressGetTempSize failed: {s}"

    # Allocate output buffers (one per chunk)
    comp_chunk_gpus = [cuda_malloc(max_out2.value) for _ in range(NUM_CHUNKS)]
    comp_temp2 = cuda_malloc(ct2.value)

    # Build device pointer/size arrays
    uncomp_ptrs_list = [
        C.c_void_p(big_gpu.value + i * CHUNK) for i in range(NUM_CHUNKS)
    ]
    uncomp_sizes_list = [CHUNK] * NUM_CHUNKS

    d_uc_ptrs = make_device_array_ptrs(uncomp_ptrs_list)
    d_uc_sizes = make_device_array_sizes(uncomp_sizes_list)
    d_c_ptrs = make_device_array_ptrs(comp_chunk_gpus)
    d_c_sizes = make_device_array_sizes([0] * NUM_CHUNKS)
    sync()

    # Compress all chunks in one batched call
    s = shim.nvcomp_shim_ans_compress_async(
        d_uc_ptrs,
        d_uc_sizes,
        C.c_size_t(CHUNK),
        C.c_size_t(NUM_CHUNKS),
        comp_temp2,
        C.c_size_t(ct2.value),
        d_c_ptrs,
        d_c_sizes,
        C.byref(compress_opts),
        C.c_void_p(0),
        C.c_void_p(0),
    )
    print(f"  Batched CompressAsync ({NUM_CHUNKS} chunks): status={s}")
    assert s == 0, f"Batched compress failed: {s}"
    sync()

    # Read back per-chunk compressed sizes
    host_comp_sizes = (C.c_size_t * NUM_CHUNKS)()
    d2h(host_comp_sizes, d_c_sizes, NUM_CHUNKS * 8)
    total_comp = sum(host_comp_sizes)
    print(
        f"  Compressed: {TOTAL / 1e6:.0f} MB -> {total_comp / 1e6:.1f} MB "
        f"({TOTAL / total_comp:.2f}x)"
    )
    for i in range(NUM_CHUNKS):
        cs = host_comp_sizes[i]
        print(f"    chunk {i}: {CHUNK} -> {cs} ({CHUNK / cs:.2f}x)")

    # Decompress
    dt2 = C.c_size_t(0)
    s = shim.nvcomp_shim_ans_decompress_get_temp_size(
        C.c_size_t(NUM_CHUNKS),
        C.c_size_t(CHUNK),
        C.byref(decompress_opts),
        C.byref(dt2),
        C.c_size_t(TOTAL),
    )
    assert s == 0

    big_dst = cuda_malloc(TOTAL)
    decomp_temp2 = cuda_malloc(dt2.value)

    dec_uncomp_ptrs_list = [
        C.c_void_p(big_dst.value + i * CHUNK) for i in range(NUM_CHUNKS)
    ]
    dec_comp_sizes_list = [host_comp_sizes[i] for i in range(NUM_CHUNKS)]
    dec_buf_sizes_list = [CHUNK] * NUM_CHUNKS

    d_dc_ptrs = make_device_array_ptrs(comp_chunk_gpus)
    d_dc_sizes = make_device_array_sizes(dec_comp_sizes_list)
    d_du_ptrs = make_device_array_ptrs(dec_uncomp_ptrs_list)
    d_du_bsizes = make_device_array_sizes(dec_buf_sizes_list)
    sync()

    s = shim.nvcomp_shim_ans_decompress_async(
        d_dc_ptrs,
        d_dc_sizes,
        d_du_bsizes,
        C.c_void_p(0),
        C.c_size_t(NUM_CHUNKS),
        decomp_temp2,
        C.c_size_t(dt2.value),
        d_du_ptrs,
        C.byref(decompress_opts),
        C.c_void_p(0),
        C.c_void_p(0),
    )
    print(f"  Batched DecompressAsync ({NUM_CHUNKS} chunks): status={s}")
    assert s == 0, f"Batched decompress failed: {s}"
    sync()

    # Verify
    big_result = (C.c_char * TOTAL)()
    d2h(big_result, big_dst, TOTAL)
    match = bytes(big_result) == big_data
    print(f"  MATCH: {match}")
    assert match, "Data mismatch in 128MB multi-chunk roundtrip!"

    # ==================================================================
    # Test 3: Benchmark decompress throughput
    # ==================================================================
    print()
    print("=" * 60)
    print("Test 3: ANS decompress throughput benchmark")
    print("=" * 60)

    # Warmup
    shim.nvcomp_shim_ans_decompress_async(
        d_dc_ptrs,
        d_dc_sizes,
        d_du_bsizes,
        C.c_void_p(0),
        C.c_size_t(NUM_CHUNKS),
        decomp_temp2,
        C.c_size_t(dt2.value),
        d_du_ptrs,
        C.byref(decompress_opts),
        C.c_void_p(0),
        C.c_void_p(0),
    )
    sync()

    times = []
    for _ in range(20):
        sync()
        t0 = time.perf_counter()
        shim.nvcomp_shim_ans_decompress_async(
            d_dc_ptrs,
            d_dc_sizes,
            d_du_bsizes,
            C.c_void_p(0),
            C.c_size_t(NUM_CHUNKS),
            decomp_temp2,
            C.c_size_t(dt2.value),
            d_du_ptrs,
            C.byref(decompress_opts),
            C.c_void_p(0),
            C.c_void_p(0),
        )
        sync()
        times.append(time.perf_counter() - t0)

    times.sort()
    med = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    tput = TOTAL / med / 1e9
    print(f"  128 MB ANS decompress ({NUM_CHUNKS} chunks):")
    print(f"    median={med * 1000:.2f} ms  p10={p10 * 1000:.2f}  p90={p90 * 1000:.2f}")
    print(f"    throughput={tput:.1f} GB/s")

    # Also benchmark compress
    times_c = []
    for _ in range(20):
        sync()
        t0 = time.perf_counter()
        shim.nvcomp_shim_ans_compress_async(
            d_uc_ptrs,
            d_uc_sizes,
            C.c_size_t(CHUNK),
            C.c_size_t(NUM_CHUNKS),
            comp_temp2,
            C.c_size_t(ct2.value),
            d_c_ptrs,
            d_c_sizes,
            C.byref(compress_opts),
            C.c_void_p(0),
            C.c_void_p(0),
        )
        sync()
        times_c.append(time.perf_counter() - t0)

    times_c.sort()
    med_c = times_c[len(times_c) // 2]
    tput_c = TOTAL / med_c / 1e9
    print(f"  128 MB ANS compress ({NUM_CHUNKS} chunks):")
    print(f"    median={med_c * 1000:.2f} ms  throughput={tput_c:.1f} GB/s")

    print()
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
