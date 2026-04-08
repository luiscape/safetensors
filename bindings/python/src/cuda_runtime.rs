//! Direct CUDA runtime FFI loaded at runtime via `dlopen`.
//!
//! This module provides a thin wrapper around a handful of CUDA runtime API
//! functions that are resolved **at runtime** through `libloading` (i.e.
//! `dlopen` / `dlsym`).  This avoids a hard link-time dependency on
//! `libcudart.so` and — critically — allows Rust worker threads to perform
//! `cudaMemcpy`, `cudaHostAlloc`, etc. **without holding the Python GIL**.
//!
//! The [`CudaRuntime`] handle is cached as a process-wide singleton via
//! [`std::sync::OnceLock`] so that the (potentially expensive) library load
//! and symbol resolution happens at most once.
//!
//! # Feature gate
//!
//! This entire module is compiled only when the `gds` feature is enabled,
//! because it depends on `libloading` which is an optional dependency gated
//! behind that feature.

use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA constants
// ---------------------------------------------------------------------------

/// `cudaMemcpyHostToDevice` enum value from `cuda_runtime_api.h`.
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

/// `cudaHostAllocDefault` flag — no special behaviour.
const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

/// Cached CUDA runtime handle — loaded once per process.
///
/// The outer `Option` is `None` when `libcudart.so` could not be loaded (or
/// symbol resolution failed).  Once the `OnceLock` is initialised the value
/// is immutable for the lifetime of the process.
static CUDA_RT: OnceLock<Option<CudaRuntime>> = OnceLock::new();

// ---------------------------------------------------------------------------
// CudaRuntime
// ---------------------------------------------------------------------------

/// A collection of CUDA runtime function pointers loaded via `dlopen`.
///
/// Instances of this struct are **`Send + Sync`** because the underlying
/// function pointers are plain `extern "C"` symbols whose thread-safety is
/// guaranteed by the CUDA runtime itself.
#[allow(dead_code)]
pub struct CudaRuntime {
    /// We keep the library handle alive so that the loaded symbols remain
    /// valid for the lifetime of the process.
    _lib: libloading::Library,

    // -- function pointers --------------------------------------------------
    cuda_malloc: unsafe extern "C" fn(dev_ptr: *mut *mut c_void, size: usize) -> i32,
    cuda_free: unsafe extern "C" fn(dev_ptr: *mut c_void) -> i32,
    cuda_memcpy:
        unsafe extern "C" fn(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32,
    cuda_host_alloc: unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32,
    cuda_free_host: unsafe extern "C" fn(ptr: *mut c_void) -> i32,
    cuda_set_device: unsafe extern "C" fn(device: i32) -> i32,
    cuda_device_synchronize: unsafe extern "C" fn() -> i32,
}

// Safety: the CUDA runtime is internally thread-safe.  All function pointers
// are plain `extern "C"` addresses that do not capture mutable state.
unsafe impl Send for CudaRuntime {}
unsafe impl Sync for CudaRuntime {}

impl CudaRuntime {
    // -- construction -------------------------------------------------------

    /// Attempts to load `libcudart.so` and resolve all required symbols.
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string if the library cannot be opened
    /// or any symbol cannot be found.
    ///
    /// # Safety
    ///
    /// This function loads shared-library symbols at runtime.  The resolved
    /// function pointers are assumed to match the CUDA 11+ runtime ABI.
    pub unsafe fn load() -> Result<Self, String> {
        let lib = libloading::Library::new("libcudart.so").map_err(|e| {
            format!("Failed to load libcudart.so — is the CUDA runtime installed? Error: {e}")
        })?;

        macro_rules! load_sym {
            ($lib:expr, $name:literal, $ty:ty) => {{
                let sym: libloading::Symbol<$ty> = $lib
                    .get($name)
                    .map_err(|e| format!("Failed to load symbol {}: {e}", stringify!($name)))?;
                *sym
            }};
        }

        let cuda_malloc = load_sym!(
            lib,
            b"cudaMalloc\0",
            unsafe extern "C" fn(*mut *mut c_void, usize) -> i32
        );
        let cuda_free = load_sym!(lib, b"cudaFree\0", unsafe extern "C" fn(*mut c_void) -> i32);
        let cuda_memcpy = load_sym!(
            lib,
            b"cudaMemcpy\0",
            unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32
        );
        let cuda_host_alloc = load_sym!(
            lib,
            b"cudaHostAlloc\0",
            unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> i32
        );
        let cuda_free_host = load_sym!(
            lib,
            b"cudaFreeHost\0",
            unsafe extern "C" fn(*mut c_void) -> i32
        );
        let cuda_set_device = load_sym!(lib, b"cudaSetDevice\0", unsafe extern "C" fn(i32) -> i32);
        let cuda_device_synchronize = load_sym!(
            lib,
            b"cudaDeviceSynchronize\0",
            unsafe extern "C" fn() -> i32
        );

        Ok(Self {
            _lib: lib,
            cuda_malloc: std::mem::transmute(cuda_malloc),
            cuda_free: std::mem::transmute(cuda_free),
            cuda_memcpy: std::mem::transmute(cuda_memcpy),
            cuda_host_alloc: std::mem::transmute(cuda_host_alloc),
            cuda_free_host: std::mem::transmute(cuda_free_host),
            cuda_set_device: std::mem::transmute(cuda_set_device),
            cuda_device_synchronize: std::mem::transmute(cuda_device_synchronize),
        })
    }

    // -- singleton accessor -------------------------------------------------

    /// Returns the process-wide [`CudaRuntime`] singleton, or `None` if the
    /// CUDA runtime library could not be loaded.
    ///
    /// The first call triggers `dlopen("libcudart.so")` and symbol resolution.
    /// Subsequent calls return the cached result in O(1).
    pub fn get() -> Option<&'static CudaRuntime> {
        CUDA_RT
            .get_or_init(|| {
                // Safety: we are loading well-known CUDA runtime symbols whose
                // ABI has been stable since CUDA 11.
                match unsafe { CudaRuntime::load() } {
                    Ok(rt) => Some(rt),
                    Err(_e) => {
                        // In debug builds surface the error for diagnostics.
                        #[cfg(debug_assertions)]
                        eprintln!("[cuda_runtime] Failed to load CUDA runtime: {_e}");
                        None
                    }
                }
            })
            .as_ref()
    }

    // -- public API ---------------------------------------------------------

    /// Allocates `size` bytes of device (GPU) memory via `cudaMalloc`.
    ///
    /// # Safety
    ///
    /// The caller must eventually free the returned pointer with [`free`](Self::free).
    #[allow(dead_code)]
    pub unsafe fn malloc(&self, size: usize) -> Result<*mut c_void, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = (self.cuda_malloc)(&mut ptr, size);
        if status != 0 {
            return Err(format!(
                "cudaMalloc({size} bytes) failed with error code {status}"
            ));
        }
        if ptr.is_null() {
            return Err("cudaMalloc returned a null pointer".into());
        }
        Ok(ptr)
    }

    /// Frees device memory previously allocated with [`malloc`](Self::malloc).
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior successful call to
    /// [`malloc`](Self::malloc) and must not have been freed already.
    #[allow(dead_code)]
    pub unsafe fn free(&self, ptr: *mut c_void) {
        let _status = (self.cuda_free)(ptr);
        // We intentionally ignore the return code in the free path to avoid
        // panicking during cleanup.
    }

    /// Copies `size` bytes from host memory (`src`) to device memory (`dst`)
    /// using `cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)`.
    ///
    /// # Safety
    ///
    /// * `dst` must be a valid CUDA device pointer with at least `size` bytes.
    /// * `src` must be a valid host pointer with at least `size` readable bytes.
    pub unsafe fn memcpy_h2d(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> Result<(), String> {
        let status = (self.cuda_memcpy)(dst, src, size, CUDA_MEMCPY_HOST_TO_DEVICE);
        if status != 0 {
            return Err(format!(
                "cudaMemcpy(H2D, {size} bytes) failed with error code {status}"
            ));
        }
        Ok(())
    }

    /// Allocates `size` bytes of **page-locked** (pinned) host memory via
    /// `cudaHostAlloc`.
    ///
    /// Pinned memory enables higher-bandwidth DMA transfers between host and
    /// device.
    ///
    /// # Safety
    ///
    /// The caller must eventually free the returned pointer with
    /// [`host_free`](Self::host_free).
    pub unsafe fn host_alloc(&self, size: usize) -> Result<*mut c_void, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let status = (self.cuda_host_alloc)(&mut ptr, size, CUDA_HOST_ALLOC_DEFAULT);
        if status != 0 {
            return Err(format!(
                "cudaHostAlloc({size} bytes) failed with error code {status}"
            ));
        }
        if ptr.is_null() {
            return Err("cudaHostAlloc returned a null pointer".into());
        }
        Ok(ptr)
    }

    /// Frees pinned host memory previously allocated with
    /// [`host_alloc`](Self::host_alloc).
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior successful call to
    /// [`host_alloc`](Self::host_alloc) and must not have been freed already.
    pub unsafe fn host_free(&self, ptr: *mut c_void) {
        let _status = (self.cuda_free_host)(ptr);
    }

    /// Sets the current CUDA device for the calling thread.
    ///
    /// # Safety
    ///
    /// `device_id` must be a valid device ordinal (0-based).
    pub unsafe fn set_device(&self, device_id: i32) -> Result<(), String> {
        let status = (self.cuda_set_device)(device_id);
        if status != 0 {
            return Err(format!(
                "cudaSetDevice({device_id}) failed with error code {status}"
            ));
        }
        Ok(())
    }

    /// Blocks the calling thread until all preceding CUDA operations on the
    /// current device have completed.
    ///
    /// # Safety
    ///
    /// The caller should have previously called [`set_device`](Self::set_device)
    /// to select the appropriate device.
    pub unsafe fn synchronize(&self) -> Result<(), String> {
        let status = (self.cuda_device_synchronize)();
        if status != 0 {
            return Err(format!(
                "cudaDeviceSynchronize() failed with error code {status}"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the singleton accessor is consistent across calls.
    #[test]
    fn test_singleton_consistency() {
        // On machines without CUDA this will return None — that's fine.
        let first = CudaRuntime::get();
        let second = CudaRuntime::get();
        match (first, second) {
            (Some(a), Some(b)) => {
                // Both references must point to the same static allocation.
                assert!(std::ptr::eq(a, b));
            }
            (None, None) => {
                // CUDA not available — expected in CI / non-GPU environments.
            }
            _ => panic!("Singleton returned inconsistent results"),
        }
    }
}
