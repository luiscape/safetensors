/*
 * nvcomp ABI shim — wraps nvcomp batched ANS and Zstd functions that accept
 * a 64-byte opts struct **by value** on the C ABI.
 *
 * On x86-64 System V ABI, structs larger than 4 eightbytes (32 bytes) are
 * classified as MEMORY and their stack layout is tricky for FFI callers
 * (Rust extern "C", Python ctypes) to get right.  This shim exposes thin
 * wrappers that accept opts **by pointer** and forward the dereferenced
 * struct by value to the real nvcomp symbols.
 *
 * Build (standalone, for testing):
 *   gcc -shared -fPIC -O2 -o libnvcomp_shim.so nvcomp_shim.c \
 *       -L/path/to/nvcomp/lib -lnvcomp -Wl,-rpath,/path/to/nvcomp/lib
 *
 * In practice this is compiled by the Rust build.rs and linked into the
 * safetensors-python wheel.
 */

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------
 * Minimal type declarations mirroring nvcomp headers.
 * We only need the struct layouts and function signatures; the actual
 * enum values are opaque integers to us.
 * ------------------------------------------------------------------ */

typedef int nvcompStatus_t;
typedef void *cudaStream_t;

/* nvcompBatchedANSDecompressOpts_t  (ans.h)
 * nvcompBatchedZstdDecompressOpts_t (zstd.h)
 * Both have identical layout: { int backend; char reserved[60]; }
 * We use a single C type for both since the binary representation is
 * the same and the nvcomp functions only inspect `backend`.            */
typedef struct {
    int   backend;       /* nvcompDecompressBackend_t, 0 = DEFAULT */
    char  reserved[60];  /* must be zeroed                         */
} nvcomp_decompress_opts_t;

/* Same story for compress opts.
 * ANS: { int algo; int type; char reserved[56]; }   — 64 bytes
 * Zstd: { char reserved[64]; }                      — 64 bytes
 * We treat them as opaque 64-byte blobs.              */
typedef struct {
    char data[64];
} nvcomp_compress_opts_t;


/* ------------------------------------------------------------------
 * Extern declarations for the real nvcomp symbols.
 * These are resolved at link time (or via dlopen when the shim .so
 * is loaded with nvcomp on LD_LIBRARY_PATH).
 * ------------------------------------------------------------------ */

/* ---- ANS Compress ---- */
extern nvcompStatus_t nvcompBatchedANSCompressGetTempSizeAsync(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    nvcomp_compress_opts_t compress_opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes);

extern nvcompStatus_t nvcompBatchedANSCompressGetMaxOutputChunkSize(
    size_t max_uncompressed_chunk_bytes,
    nvcomp_compress_opts_t compress_opts,
    size_t *max_compressed_chunk_bytes);

extern nvcompStatus_t nvcompBatchedANSCompressAsync(
    const void *const *device_uncompressed_ptrs,
    const size_t *device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_compressed_ptrs,
    size_t *device_compressed_bytes,
    nvcomp_compress_opts_t compress_opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream);

/* ---- ANS Decompress ---- */
extern nvcompStatus_t nvcompBatchedANSDecompressGetTempSizeAsync(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    nvcomp_decompress_opts_t decompress_opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes);

extern nvcompStatus_t nvcompBatchedANSDecompressAsync(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_uncompressed_ptrs,
    nvcomp_decompress_opts_t decompress_opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream);

extern nvcompStatus_t nvcompBatchedANSGetDecompressSizeAsync(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    size_t *device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream);

/* ---- Zstd Decompress ---- */
extern nvcompStatus_t nvcompBatchedZstdDecompressGetTempSizeAsync(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    nvcomp_decompress_opts_t decompress_opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes);

extern nvcompStatus_t nvcompBatchedZstdDecompressAsync(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_uncompressed_ptrs,
    nvcomp_decompress_opts_t decompress_opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream);


/* ==================================================================
 * SHIM FUNCTIONS — accept opts BY POINTER, forward BY VALUE.
 *
 * Naming convention:  nvcomp_shim_<algorithm>_<operation>
 * ================================================================== */

/* ---- ANS Compress shims ---- */

nvcompStatus_t nvcomp_shim_ans_compress_get_temp_size(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    const nvcomp_compress_opts_t *opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes)
{
    return nvcompBatchedANSCompressGetTempSizeAsync(
        num_chunks, max_uncompressed_chunk_bytes,
        *opts, temp_bytes, max_total_uncompressed_bytes);
}

nvcompStatus_t nvcomp_shim_ans_compress_get_max_output_chunk_size(
    size_t max_uncompressed_chunk_bytes,
    const nvcomp_compress_opts_t *opts,
    size_t *max_compressed_chunk_bytes)
{
    return nvcompBatchedANSCompressGetMaxOutputChunkSize(
        max_uncompressed_chunk_bytes, *opts, max_compressed_chunk_bytes);
}

nvcompStatus_t nvcomp_shim_ans_compress_async(
    const void *const *device_uncompressed_ptrs,
    const size_t *device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_compressed_ptrs,
    size_t *device_compressed_bytes,
    const nvcomp_compress_opts_t *opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream)
{
    return nvcompBatchedANSCompressAsync(
        device_uncompressed_ptrs, device_uncompressed_bytes,
        max_uncompressed_chunk_bytes, num_chunks,
        device_temp_ptr, temp_bytes,
        device_compressed_ptrs, device_compressed_bytes,
        *opts, device_statuses, stream);
}

/* ---- ANS Decompress shims ---- */

nvcompStatus_t nvcomp_shim_ans_decompress_get_temp_size(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    const nvcomp_decompress_opts_t *opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes)
{
    return nvcompBatchedANSDecompressGetTempSizeAsync(
        num_chunks, max_uncompressed_chunk_bytes,
        *opts, temp_bytes, max_total_uncompressed_bytes);
}

nvcompStatus_t nvcomp_shim_ans_decompress_async(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_uncompressed_ptrs,
    const nvcomp_decompress_opts_t *opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream)
{
    return nvcompBatchedANSDecompressAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes,
        num_chunks,
        device_temp_ptr, temp_bytes,
        device_uncompressed_ptrs,
        *opts, device_statuses, stream);
}

nvcompStatus_t nvcomp_shim_ans_get_decompress_size_async(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    size_t *device_uncompressed_bytes,
    size_t batch_size,
    cudaStream_t stream)
{
    return nvcompBatchedANSGetDecompressSizeAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, batch_size, stream);
}

/* ---- Zstd Decompress shims ---- */

nvcompStatus_t nvcomp_shim_zstd_decompress_get_temp_size(
    size_t num_chunks,
    size_t max_uncompressed_chunk_bytes,
    const nvcomp_decompress_opts_t *opts,
    size_t *temp_bytes,
    size_t max_total_uncompressed_bytes)
{
    return nvcompBatchedZstdDecompressGetTempSizeAsync(
        num_chunks, max_uncompressed_chunk_bytes,
        *opts, temp_bytes, max_total_uncompressed_bytes);
}

nvcompStatus_t nvcomp_shim_zstd_decompress_async(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    const size_t *device_uncompressed_buffer_bytes,
    size_t *device_uncompressed_chunk_bytes,
    size_t num_chunks,
    void *device_temp_ptr,
    size_t temp_bytes,
    void *const *device_uncompressed_ptrs,
    const nvcomp_decompress_opts_t *opts,
    nvcompStatus_t *device_statuses,
    cudaStream_t stream)
{
    return nvcompBatchedZstdDecompressAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_buffer_bytes, device_uncompressed_chunk_bytes,
        num_chunks,
        device_temp_ptr, temp_bytes,
        device_uncompressed_ptrs,
        *opts, device_statuses, stream);
}