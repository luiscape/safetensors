//! GPU loading optimizations for safetensors.
//!
//! This module provides high-performance GPU loading capabilities including:
//! - CUDA stream pools for parallel async transfers
//! - Pinned memory pools for faster DMA transfers
//! - Double-buffered loading pipelines
//! - Optional GPUDirect Storage support
//!
//! # Example
//!
//! ```ignore
//! use safetensors::gpu::{GpuLoader, GpuLoaderConfig};
//!
//! let config = GpuLoaderConfig::builder()
//!     .num_streams(4)
//!     .pinned_memory_mb(512)
//!     .build();
//!
//! let loader = GpuLoader::new(config)?;
//! let tensors = loader.load_file("model.safetensors")?;
//! ```

pub mod cuda;
mod error;
mod loader;
mod memory;
mod stream;

#[cfg(feature = "io-uring")]
pub mod io_uring;

#[cfg(feature = "io-uring")]
mod loader_uring;

// Provide AlignedBuffer even without io-uring feature for general use
#[cfg(not(feature = "io-uring"))]
mod aligned_buffer {
    //! Aligned buffer implementation for non-io_uring builds.

    #![allow(missing_docs)]

    use super::error::{GpuError, GpuResult};

    /// A buffer aligned for direct I/O operations.
    pub struct AlignedBuffer {
        ptr: *mut u8,
        len: usize,
        capacity: usize,
        alignment: usize,
    }

    impl AlignedBuffer {
        /// Create a new aligned buffer with the given capacity and alignment.
        pub fn new(capacity: usize, alignment: usize) -> GpuResult<Self> {
            if capacity == 0 {
                return Err(GpuError::InvalidConfig {
                    message: "cannot allocate zero-sized buffer".into(),
                });
            }
            let aligned_cap = (capacity + alignment - 1) & !(alignment - 1);
            let layout = std::alloc::Layout::from_size_align(aligned_cap, alignment).map_err(|_| {
                GpuError::InvalidConfig {
                    message: format!("invalid layout: size={}, align={}", aligned_cap, alignment),
                }
            })?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(GpuError::PinnedAllocationFailed {
                    requested_bytes: aligned_cap,
                    reason: Some("aligned allocation failed".into()),
                });
            }
            Ok(Self { ptr, len: 0, capacity: aligned_cap, alignment })
        }

        /// Create a buffer with default 4KB alignment.
        pub fn with_capacity(capacity: usize) -> GpuResult<Self> {
            Self::new(capacity, 4096)
        }

        #[inline] pub fn capacity(&self) -> usize { self.capacity }
        #[inline] pub fn len(&self) -> usize { self.len }
        #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
        #[inline] pub fn alignment(&self) -> usize { self.alignment }
        #[inline] pub fn as_ptr(&self) -> *const u8 { self.ptr }
        #[inline] pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr }
        #[inline] pub fn as_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
        #[inline] pub fn as_mut_slice(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } }
        #[inline] pub unsafe fn set_len(&mut self, len: usize) { debug_assert!(len <= self.capacity); self.len = len; }
        #[inline] pub fn clear(&mut self) { self.len = 0; }
        #[inline] pub fn is_aligned(&self) -> bool { (self.ptr as usize) % self.alignment == 0 }
    }

    impl Drop for AlignedBuffer {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                let layout = std::alloc::Layout::from_size_align(self.capacity, self.alignment).unwrap();
                unsafe { std::alloc::dealloc(self.ptr, layout); }
            }
        }
    }

    unsafe impl Send for AlignedBuffer {}
    unsafe impl Sync for AlignedBuffer {}
}

pub use cuda::{CudaDevice, CudaEvent, CudaStream};
pub use error::{GpuError, GpuResult};
pub use loader::{GpuLoader, GpuLoaderConfig, GpuLoaderConfigBuilder, GpuTensor, LoadHandle};
pub use memory::{PinnedBuffer, PinnedMemoryPool};
pub use stream::CudaStreamPool;

#[cfg(feature = "io-uring")]
pub use io_uring::{
    AlignedBuffer, IoUringConfig, IoUringConfigBuilder, IoUringReader, ReadCompletion,
    ReadRequest, TensorFileReader,
};

#[cfg(feature = "io-uring")]
pub use loader_uring::{UringLoader, UringLoaderConfig, UringLoaderConfigBuilder};

#[cfg(not(feature = "io-uring"))]
pub use aligned_buffer::AlignedBuffer;

/// Re-export for convenience
pub mod low_level {
    pub use super::cuda::*;
    pub use super::memory::*;
    pub use super::stream::*;
}
