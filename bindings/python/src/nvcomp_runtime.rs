//! nvcomp runtime FFI via the `libnvcomp_shim.so` C shim library.
//!
//! This module dynamically loads `libnvcomp_shim.so` at runtime to avoid a
//! hard link-time dependency on nvcomp.  The shim wraps nvcomp's batched ANS
//! compress/decompress API, accepting the 64-byte opts structs **by pointer**
//! instead of by value, which sidesteps the x86-64 ABI mismatch that can
//! occur when Rust passes large structs by value through `extern "C"` FFI.
//!
//! The shim function signatures are:
//!
//! ```c
//! // Compress
//! nvcompStatus_t nvcomp_shim_ans_compress_get_temp_size(
//!     size_t num_chunks, size_t max_chunk,
//!     const nvcomp_compress_opts_t *opts,
//!     size_t *temp_bytes, size_t max_total);
//!
//! nvcompStatus_t nvcomp_shim_ans_compress_get_max_output_chunk_size(
//!     size_t max_chunk,
//!     const nvcomp_compress_opts_t *opts,
//!     size_t *max_compressed);
//!
//! nvcompStatus_t nvcomp_shim_ans_compress_async(
//!     const void *const *uncomp_ptrs, const size_t *uncomp_bytes,
//!     size_t max_chunk, size_t num_chunks,
//!     void *temp, size_t temp_bytes,
//!     void *const *comp_ptrs, size_t *comp_bytes,
//!     const nvcomp_compress_opts_t *opts,
//!     nvcompStatus_t *statuses, cudaStream_t stream);
//!
//! // Decompress
//! nvcompStatus_t nvcomp_shim_ans_decompress_get_temp_size(
//!     size_t num_chunks, size_t max_chunk,
//!     const nvcomp_decompress_opts_t *opts,
//!     size_t *temp_bytes, size_t max_total);
//!
//! nvcompStatus_t nvcomp_shim_ans_decompress_async(
//!     const void *const *comp_ptrs, const size_t *comp_bytes,
//!     const size_t *uncomp_buf_bytes, size_t *actual_uncomp_bytes,
//!     size_t num_chunks, void *temp, size_t temp_bytes,
//!     void *const *uncomp_ptrs,
//!     const nvcomp_decompress_opts_t *opts,
//!     nvcompStatus_t *statuses, cudaStream_t stream);
//! ```
//!
//! # Feature gate
//!
//! This module is compiled only when the `nvcomp` feature is enabled.

use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// nvcomp status codes (from shared_types.h)
// ---------------------------------------------------------------------------

/// nvcomp success status.
pub const NVCOMP_SUCCESS: i32 = 0;

// ---------------------------------------------------------------------------
// nvcomp types matching the C API
// ---------------------------------------------------------------------------

/// CUDA stream handle (opaque pointer).
pub type CudaStream = *mut c_void;

/// `nvcompStatus_t` — we use `i32` for FFI since it's a C enum.
pub type NvcompStatus = i32;

/// Opaque compress options struct (64 bytes).
///
/// For the ANS batched API the default options are all-zeroes.
/// The shim receives this by pointer (`const nvcomp_compress_opts_t *`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvcompCompressOpts {
    /// Opaque payload — zeroed means "default ANS options".
    pub data: [u8; 64],
}

impl Default for NvcompCompressOpts {
    fn default() -> Self {
        Self { data: [0u8; 64] }
    }
}

/// Decompress options struct (64 bytes).
///
/// The first 4 bytes are the `nvcompDecompressBackend_t` enum
/// (0 = `NVCOMP_DECOMPRESS_BACKEND_DEFAULT`); the remaining 60 bytes are
/// reserved and must be zeroed.
///
/// The shim receives this by pointer (`const nvcomp_decompress_opts_t *`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvcompDecompressOpts {
    /// Decompression backend.  0 = DEFAULT.
    pub backend: i32,
    /// Reserved bytes, must be zeroed.
    pub _reserved: [u8; 60],
}

impl Default for NvcompDecompressOpts {
    fn default() -> Self {
        Self {
            backend: 0,
            _reserved: [0u8; 60],
        }
    }
}

// ---------------------------------------------------------------------------
// Function pointer type aliases (matching libnvcomp_shim.so signatures)
// ---------------------------------------------------------------------------

// -- Compress ---------------------------------------------------------------

/// `nvcomp_shim_ans_compress_get_temp_size`
type FnCompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_chunk: usize,
    opts: *const NvcompCompressOpts,
    temp_bytes: *mut usize,
    max_total: usize,
) -> NvcompStatus;

/// `nvcomp_shim_ans_compress_get_max_output_chunk_size`
type FnCompressGetMaxOutputChunkSize = unsafe extern "C" fn(
    max_chunk: usize,
    opts: *const NvcompCompressOpts,
    max_compressed: *mut usize,
) -> NvcompStatus;

/// `nvcomp_shim_ans_compress_async`
type FnCompressAsync = unsafe extern "C" fn(
    uncomp_ptrs: *const *const c_void,
    uncomp_bytes: *const usize,
    max_chunk: usize,
    num_chunks: usize,
    temp: *mut c_void,
    temp_bytes: usize,
    comp_ptrs: *const *mut c_void,
    comp_bytes: *mut usize,
    opts: *const NvcompCompressOpts,
    statuses: *mut NvcompStatus,
    stream: CudaStream,
) -> NvcompStatus;

// -- Decompress -------------------------------------------------------------

/// `nvcomp_shim_ans_decompress_get_temp_size`
type FnDecompressGetTempSize = unsafe extern "C" fn(
    num_chunks: usize,
    max_chunk: usize,
    opts: *const NvcompDecompressOpts,
    temp_bytes: *mut usize,
    max_total: usize,
) -> NvcompStatus;

/// `nvcomp_shim_ans_decompress_async`
type FnDecompressAsync = unsafe extern "C" fn(
    comp_ptrs: *const *const c_void,
    comp_bytes: *const usize,
    uncomp_buf_bytes: *const usize,
    actual_uncomp_bytes: *mut usize,
    num_chunks: usize,
    temp: *mut c_void,
    temp_bytes: usize,
    uncomp_ptrs: *const *mut c_void,
    opts: *const NvcompDecompressOpts,
    statuses: *mut NvcompStatus,
    stream: CudaStream,
) -> NvcompStatus;

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

static NVCOMP_RT: OnceLock<Option<NvcompRuntime>> = OnceLock::new();

// ---------------------------------------------------------------------------
// NvcompRuntime
// ---------------------------------------------------------------------------

/// Runtime-loaded nvcomp shim function pointers for batched ANS
/// compression and decompression.
///
/// Loaded via `dlopen("libnvcomp_shim.so")` on first use.  The struct is
/// `Send + Sync` because the underlying nvcomp functions are thread-safe
/// (they operate on explicit CUDA streams).
pub struct NvcompRuntime {
    _lib: libloading::Library,

    // Stored opts — default-initialised once.
    compress_opts: NvcompCompressOpts,
    decompress_opts: NvcompDecompressOpts,

    // Compress function pointers
    ans_compress_get_temp_size: FnCompressGetTempSize,
    ans_compress_get_max_output_chunk_size: FnCompressGetMaxOutputChunkSize,
    ans_compress_async: FnCompressAsync,

    // Decompress function pointers
    ans_decompress_get_temp_size: FnDecompressGetTempSize,
    ans_decompress_async: FnDecompressAsync,
}

unsafe impl Send for NvcompRuntime {}
unsafe impl Sync for NvcompRuntime {}

impl NvcompRuntime {
    /// Attempt to load `libnvcomp_shim.so` and resolve all required symbols.
    ///
    /// # Safety
    ///
    /// Loads shared-library symbols at runtime.  The resolved function
    /// pointers are assumed to match the shim ABI as declared above.
    pub unsafe fn load() -> Result<Self, String> {
        let lib = libloading::Library::new("libnvcomp_shim.so").map_err(|e| {
            format!(
                "Failed to load libnvcomp_shim.so — is the nvcomp shim installed? \
                 Make sure libnvcomp_shim.so is on LD_LIBRARY_PATH.  Error: {e}"
            )
        })?;

        macro_rules! load_sym {
            ($lib:expr, $name:literal, $ty:ty) => {{
                let sym: libloading::Symbol<$ty> = $lib.get($name).map_err(|e| {
                    format!(
                        "Failed to load symbol {}: {e}",
                        String::from_utf8_lossy($name)
                    )
                })?;
                std::mem::transmute(*sym)
            }};
        }

        // -- compress symbols ------------------------------------------------
        let ans_compress_get_temp_size: FnCompressGetTempSize = load_sym!(
            lib,
            b"nvcomp_shim_ans_compress_get_temp_size\0",
            FnCompressGetTempSize
        );

        let ans_compress_get_max_output_chunk_size: FnCompressGetMaxOutputChunkSize = load_sym!(
            lib,
            b"nvcomp_shim_ans_compress_get_max_output_chunk_size\0",
            FnCompressGetMaxOutputChunkSize
        );

        let ans_compress_async: FnCompressAsync =
            load_sym!(lib, b"nvcomp_shim_ans_compress_async\0", FnCompressAsync);

        // -- decompress symbols ----------------------------------------------
        let ans_decompress_get_temp_size: FnDecompressGetTempSize = load_sym!(
            lib,
            b"nvcomp_shim_ans_decompress_get_temp_size\0",
            FnDecompressGetTempSize
        );

        let ans_decompress_async: FnDecompressAsync = load_sym!(
            lib,
            b"nvcomp_shim_ans_decompress_async\0",
            FnDecompressAsync
        );

        Ok(Self {
            _lib: lib,
            compress_opts: NvcompCompressOpts::default(),
            decompress_opts: NvcompDecompressOpts::default(),
            ans_compress_get_temp_size,
            ans_compress_get_max_output_chunk_size,
            ans_compress_async,
            ans_decompress_get_temp_size,
            ans_decompress_async,
        })
    }

    /// Returns the process-wide [`NvcompRuntime`] singleton, or `None` if
    /// `libnvcomp_shim.so` could not be loaded.
    ///
    /// The first call triggers `dlopen` and symbol resolution.  Subsequent
    /// calls return the cached result in O(1).
    pub fn get() -> Option<&'static NvcompRuntime> {
        NVCOMP_RT
            .get_or_init(|| match unsafe { NvcompRuntime::load() } {
                Ok(rt) => Some(rt),
                Err(_e) => {
                    #[cfg(debug_assertions)]
                    eprintln!("[nvcomp_runtime] Failed to load nvcomp shim: {_e}");
                    None
                }
            })
            .as_ref()
    }

    /// Returns `true` if the nvcomp shim library was successfully loaded.
    #[allow(dead_code)]
    pub fn is_available() -> bool {
        Self::get().is_some()
    }

    // -----------------------------------------------------------------------
    // Compress API
    // -----------------------------------------------------------------------

    /// Query the temporary workspace size needed for batched ANS compression.
    ///
    /// Wraps `nvcomp_shim_ans_compress_get_temp_size`.
    ///
    /// # Arguments
    ///
    /// * `num_chunks`  — Number of independently compressed chunks.
    /// * `max_chunk`   — Maximum uncompressed size of any single chunk.
    /// * `max_total`   — Total uncompressed size across all chunks.
    ///
    /// # Returns
    ///
    /// The required temporary buffer size in bytes.
    pub fn ans_compress_get_temp_size(
        &self,
        num_chunks: usize,
        max_chunk: usize,
        max_total: usize,
    ) -> Result<usize, String> {
        let mut temp_bytes: usize = 0;
        let status = unsafe {
            (self.ans_compress_get_temp_size)(
                num_chunks,
                max_chunk,
                &self.compress_opts,
                &mut temp_bytes,
                max_total,
            )
        };
        if status != NVCOMP_SUCCESS {
            return Err(format!(
                "nvcomp_shim_ans_compress_get_temp_size failed: {} (status {status})",
                status_to_string(status),
            ));
        }
        Ok(temp_bytes)
    }

    /// Query the worst-case compressed output size for a single chunk.
    ///
    /// Wraps `nvcomp_shim_ans_compress_get_max_output_chunk_size`.
    ///
    /// # Arguments
    ///
    /// * `max_chunk` — Maximum uncompressed chunk size.
    ///
    /// # Returns
    ///
    /// The maximum possible compressed chunk size in bytes.
    pub fn ans_compress_get_max_output_chunk_size(
        &self,
        max_chunk: usize,
    ) -> Result<usize, String> {
        let mut max_compressed: usize = 0;
        let status = unsafe {
            (self.ans_compress_get_max_output_chunk_size)(
                max_chunk,
                &self.compress_opts,
                &mut max_compressed,
            )
        };
        if status != NVCOMP_SUCCESS {
            return Err(format!(
                "nvcomp_shim_ans_compress_get_max_output_chunk_size failed: {} (status {status})",
                status_to_string(status),
            ));
        }
        Ok(max_compressed)
    }

    /// Perform batched ANS compression asynchronously on a CUDA stream.
    ///
    /// Wraps `nvcomp_shim_ans_compress_async`.
    ///
    /// # Safety
    ///
    /// All device pointers must be valid CUDA device memory.  The arrays of
    /// pointers and sizes must have exactly `num_chunks` elements and reside
    /// in device-accessible memory.  The temporary buffer must be at least
    /// `temp_bytes` bytes (as returned by [`ans_compress_get_temp_size`]).
    ///
    /// `stream` must be a valid CUDA stream handle (or null for the default
    /// stream).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ans_compress_async(
        &self,
        uncomp_ptrs: *const *const c_void,
        uncomp_bytes: *const usize,
        max_chunk: usize,
        num_chunks: usize,
        temp: *mut c_void,
        temp_bytes: usize,
        comp_ptrs: *const *mut c_void,
        comp_bytes: *mut usize,
        statuses: *mut NvcompStatus,
        stream: CudaStream,
    ) -> Result<(), String> {
        let status = (self.ans_compress_async)(
            uncomp_ptrs,
            uncomp_bytes,
            max_chunk,
            num_chunks,
            temp,
            temp_bytes,
            comp_ptrs,
            comp_bytes,
            &self.compress_opts,
            statuses,
            stream,
        );
        if status != NVCOMP_SUCCESS {
            return Err(format!(
                "nvcomp_shim_ans_compress_async failed: {} (status {status})",
                status_to_string(status),
            ));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Decompress API
    // -----------------------------------------------------------------------

    /// Query the temporary workspace size needed for batched ANS decompression.
    ///
    /// Wraps `nvcomp_shim_ans_decompress_get_temp_size`.
    ///
    /// # Arguments
    ///
    /// * `num_chunks`  — Number of independently compressed chunks.
    /// * `max_chunk`   — Maximum uncompressed size of any single chunk.
    /// * `max_total`   — Total uncompressed size across all chunks.
    ///
    /// # Returns
    ///
    /// The required temporary buffer size in bytes.
    pub fn ans_decompress_get_temp_size(
        &self,
        num_chunks: usize,
        max_chunk: usize,
        max_total: usize,
    ) -> Result<usize, String> {
        let mut temp_bytes: usize = 0;
        let status = unsafe {
            (self.ans_decompress_get_temp_size)(
                num_chunks,
                max_chunk,
                &self.decompress_opts,
                &mut temp_bytes,
                max_total,
            )
        };
        if status != NVCOMP_SUCCESS {
            return Err(format!(
                "nvcomp_shim_ans_decompress_get_temp_size failed: {} (status {status})",
                status_to_string(status),
            ));
        }
        Ok(temp_bytes)
    }

    /// Perform batched ANS decompression asynchronously on a CUDA stream.
    ///
    /// Wraps `nvcomp_shim_ans_decompress_async`.
    ///
    /// # Safety
    ///
    /// All device pointers must be valid CUDA device memory.  The arrays of
    /// pointers and sizes must have exactly `num_chunks` elements and reside
    /// in device-accessible memory.  The temporary buffer must be at least
    /// `temp_bytes` bytes (as returned by [`ans_decompress_get_temp_size`]).
    ///
    /// `stream` must be a valid CUDA stream handle (or null for the default
    /// stream).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn ans_decompress_async(
        &self,
        comp_ptrs: *const *const c_void,
        comp_bytes: *const usize,
        uncomp_buf_bytes: *const usize,
        actual_uncomp_bytes: *mut usize,
        num_chunks: usize,
        temp: *mut c_void,
        temp_bytes: usize,
        uncomp_ptrs: *const *mut c_void,
        statuses: *mut NvcompStatus,
        stream: CudaStream,
    ) -> Result<(), String> {
        let status = (self.ans_decompress_async)(
            comp_ptrs,
            comp_bytes,
            uncomp_buf_bytes,
            actual_uncomp_bytes,
            num_chunks,
            temp,
            temp_bytes,
            uncomp_ptrs,
            &self.decompress_opts,
            statuses,
            stream,
        );
        if status != NVCOMP_SUCCESS {
            return Err(format!(
                "nvcomp_shim_ans_decompress_async failed: {} (status {status})",
                status_to_string(status),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: format an nvcomp status code to a human-readable string
// ---------------------------------------------------------------------------

/// Convert an nvcomp status code to a human-readable description.
#[allow(dead_code)]
pub fn status_to_string(status: NvcompStatus) -> &'static str {
    match status {
        0 => "nvcompSuccess",
        10 => "nvcompErrorInvalidValue",
        11 => "nvcompErrorNotSupported",
        12 => "nvcompErrorCannotDecompress",
        13 => "nvcompErrorCannotVerifyChecksums",
        14 => "nvcompErrorCudaError",
        15 => "nvcompErrorInternal",
        16 => "nvcompErrorAlignment",
        17 => "nvcompErrorBadChecksum",
        18 => "nvcompErrorCannotQuery",
        _ => "nvcompErrorUnknown",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singleton_consistency() {
        let first = NvcompRuntime::get();
        let second = NvcompRuntime::get();
        match (first, second) {
            (Some(a), Some(b)) => {
                assert!(std::ptr::eq(a, b));
            }
            (None, None) => {
                // nvcomp shim not available — expected in most CI environments.
            }
            _ => panic!("NvcompRuntime singleton returned inconsistent results"),
        }
    }

    #[test]
    fn test_is_available_matches_get() {
        let available = NvcompRuntime::is_available();
        let get_result = NvcompRuntime::get();
        assert_eq!(available, get_result.is_some());
    }

    #[test]
    fn test_compress_opts_default_is_zeroed() {
        let opts = NvcompCompressOpts::default();
        assert!(opts.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_decompress_opts_default_is_zeroed() {
        let opts = NvcompDecompressOpts::default();
        assert_eq!(opts.backend, 0);
        assert!(opts._reserved.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_compress_opts_size() {
        assert_eq!(
            std::mem::size_of::<NvcompCompressOpts>(),
            64,
            "NvcompCompressOpts must be exactly 64 bytes"
        );
    }

    #[test]
    fn test_decompress_opts_size() {
        assert_eq!(
            std::mem::size_of::<NvcompDecompressOpts>(),
            64,
            "NvcompDecompressOpts must be exactly 64 bytes"
        );
    }
}
