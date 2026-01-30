//! Low-level CUDA bindings and safe wrappers.
//!
//! This module provides safe Rust wrappers around CUDA runtime API functions.
//! When the `cuda` feature is not enabled, it provides mock implementations
//! for testing and development purposes.

#![allow(missing_docs)]

use super::error::{GpuError, GpuResult};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

static CUDA_INITIALIZED: AtomicBool = AtomicBool::new(false);

// FFI bindings for CUDA (when cuda feature is enabled)
#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    pub type CudaStream = *mut c_void;
    pub type CudaEvent = *mut c_void;
    pub const CUDA_SUCCESS: i32 = 0;
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    pub const CUDA_HOST_ALLOC_PORTABLE: u32 = 1;
    pub const CUDA_HOST_ALLOC_WRITE_COMBINED: u32 = 4;
    pub const CUDA_EVENT_DISABLE_TIMING: u32 = 2;
    pub const CUDA_STREAM_DEFAULT: u32 = 0;

    #[link(name = "cudart")]
    extern "C" {
        pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
        pub fn cudaSetDevice(device: i32) -> i32;
        pub fn cudaGetDevice(device: *mut i32) -> i32;
        pub fn cudaDeviceSynchronize() -> i32;
        pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(ptr: *mut c_void) -> i32;
        pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
        pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
        pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: CudaStream) -> i32;
        pub fn cudaStreamCreateWithPriority(stream: *mut CudaStream, flags: u32, priority: i32) -> i32;
        pub fn cudaStreamDestroy(stream: CudaStream) -> i32;
        pub fn cudaStreamSynchronize(stream: CudaStream) -> i32;
        pub fn cudaStreamQuery(stream: CudaStream) -> i32;
        pub fn cudaEventCreateWithFlags(event: *mut CudaEvent, flags: u32) -> i32;
        pub fn cudaEventDestroy(event: CudaEvent) -> i32;
        pub fn cudaEventRecord(event: CudaEvent, stream: CudaStream) -> i32;
        pub fn cudaEventSynchronize(event: CudaEvent) -> i32;
        pub fn cudaEventQuery(event: CudaEvent) -> i32;
        pub fn cudaGetErrorString(error: i32) -> *const i8;
    }
}

// Mock FFI for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
mod ffi {
    use std::ffi::c_void;
    pub type CudaStream = *mut c_void;
    pub type CudaEvent = *mut c_void;
    pub const CUDA_SUCCESS: i32 = 0;
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
    pub const CUDA_HOST_ALLOC_PORTABLE: u32 = 1;
    pub const CUDA_HOST_ALLOC_WRITE_COMBINED: u32 = 4;
    pub const CUDA_EVENT_DISABLE_TIMING: u32 = 2;
    pub const CUDA_STREAM_DEFAULT: u32 = 0;
}

#[allow(dead_code)]
fn cuda_error(code: i32, context: &str) -> GpuError {
    #[cfg(feature = "cuda")]
    let message = {
        let msg_ptr = unsafe { ffi::cudaGetErrorString(code) };
        if msg_ptr.is_null() {
            format!("{}: unknown error", context)
        } else {
            let c_str = unsafe { std::ffi::CStr::from_ptr(msg_ptr) };
            format!("{}: {}", context, c_str.to_string_lossy())
        }
    };
    #[cfg(not(feature = "cuda"))]
    let message = format!("{}: error code {}", context, code);
    GpuError::CudaError { code, message }
}

#[inline]
#[allow(dead_code)]
fn check_cuda(result: i32, context: &str) -> GpuResult<()> {
    if result == ffi::CUDA_SUCCESS { Ok(()) } else { Err(cuda_error(result, context)) }
}

/// Represents a CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CudaDevice { id: i32 }

impl CudaDevice {
    /// Create a device handle for the given device ID.
    pub fn new(id: i32) -> GpuResult<Self> {
        let count = Self::count()?;
        if id < 0 || id >= count {
            return Err(GpuError::InvalidDevice { device_id: id, num_devices: count });
        }
        Ok(Self { id })
    }

    #[inline]
    pub fn id(&self) -> i32 { self.id }

    pub fn count() -> GpuResult<i32> {
        #[cfg(feature = "cuda")]
        { let mut count = 0; check_cuda(unsafe { ffi::cudaGetDeviceCount(&mut count) }, "cudaGetDeviceCount")?; Ok(count) }
        #[cfg(not(feature = "cuda"))]
        { Ok(1) }
    }

    pub fn current() -> GpuResult<Self> {
        #[cfg(feature = "cuda")]
        { let mut id = 0; check_cuda(unsafe { ffi::cudaGetDevice(&mut id) }, "cudaGetDevice")?; Ok(Self { id }) }
        #[cfg(not(feature = "cuda"))]
        { Ok(Self { id: 0 }) }
    }

    pub fn set_current(&self) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaSetDevice(self.id) }, "cudaSetDevice") }
        #[cfg(not(feature = "cuda"))]
        { Ok(()) }
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        self.set_current()?;
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaDeviceSynchronize() }, "cudaDeviceSynchronize") }
        #[cfg(not(feature = "cuda"))]
        { Ok(()) }
    }

    pub fn alloc(&self, size: usize) -> GpuResult<DevicePtr> {
        self.set_current()?;
        #[cfg(feature = "cuda")]
        {
            let mut ptr: *mut c_void = ptr::null_mut();
            check_cuda(unsafe { ffi::cudaMalloc(&mut ptr, size) }, "cudaMalloc")?;
            Ok(DevicePtr { ptr, size, device_id: self.id })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let layout = std::alloc::Layout::from_size_align(size, 256)
                .map_err(|_| GpuError::GpuAllocationFailed { requested_bytes: size, device_id: self.id })?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() { return Err(GpuError::GpuAllocationFailed { requested_bytes: size, device_id: self.id }); }
            Ok(DevicePtr { ptr: ptr as *mut c_void, size, device_id: self.id })
        }
    }
}

/// A pointer to memory allocated on a CUDA device.
#[derive(Debug)]
pub struct DevicePtr { ptr: *mut c_void, size: usize, device_id: i32 }

impl DevicePtr {
    #[inline] pub fn as_ptr(&self) -> *mut c_void { self.ptr }
    #[inline] pub fn size(&self) -> usize { self.size }
    #[inline] pub fn device_id(&self) -> i32 { self.device_id }

    pub fn copy_from_host(&self, src: &[u8]) -> GpuResult<()> {
        if src.len() > self.size { return Err(GpuError::BufferSizeMismatch { expected: self.size, actual: src.len() }); }
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaMemcpy(self.ptr, src.as_ptr() as *const c_void, src.len(), ffi::CUDA_MEMCPY_HOST_TO_DEVICE) }, "cudaMemcpy (H2D)") }
        #[cfg(not(feature = "cuda"))]
        { unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.ptr as *mut u8, src.len()); } Ok(()) }
    }

    pub fn copy_from_host_async(&self, src: &[u8], stream: &CudaStream) -> GpuResult<()> {
        if src.len() > self.size { return Err(GpuError::BufferSizeMismatch { expected: self.size, actual: src.len() }); }
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaMemcpyAsync(self.ptr, src.as_ptr() as *const c_void, src.len(), ffi::CUDA_MEMCPY_HOST_TO_DEVICE, stream.raw()) }, "cudaMemcpyAsync (H2D)") }
        #[cfg(not(feature = "cuda"))]
        { unsafe { ptr::copy_nonoverlapping(src.as_ptr(), self.ptr as *mut u8, src.len()); } let _ = stream; Ok(()) }
    }

    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() > self.size { return Err(GpuError::BufferSizeMismatch { expected: self.size, actual: dst.len() }); }
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaMemcpy(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len(), ffi::CUDA_MEMCPY_DEVICE_TO_HOST) }, "cudaMemcpy (D2H)") }
        #[cfg(not(feature = "cuda"))]
        { unsafe { ptr::copy_nonoverlapping(self.ptr as *const u8, dst.as_mut_ptr(), dst.len()); } Ok(()) }
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(feature = "cuda")]
            { unsafe { let _ = ffi::cudaFree(self.ptr); } }
            #[cfg(not(feature = "cuda"))]
            { let layout = std::alloc::Layout::from_size_align(self.size, 256).unwrap(); unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout); } }
        }
    }
}

unsafe impl Send for DevicePtr {}
unsafe impl Sync for DevicePtr {}

/// Stream priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamPriority { High, #[default] Normal, Low }

/// A CUDA stream for asynchronous operations.
pub struct CudaStream { stream: ffi::CudaStream, device_id: i32 }

impl CudaStream {
    pub fn new() -> GpuResult<Self> { Self::with_priority(StreamPriority::Normal) }

    pub fn with_priority(priority: StreamPriority) -> GpuResult<Self> {
        let device = CudaDevice::current()?;
        #[cfg(feature = "cuda")]
        {
            let mut stream: ffi::CudaStream = ptr::null_mut();
            let p = match priority { StreamPriority::High => -1, StreamPriority::Normal => 0, StreamPriority::Low => 0 };
            check_cuda(unsafe { ffi::cudaStreamCreateWithPriority(&mut stream, ffi::CUDA_STREAM_DEFAULT, p) }, "cudaStreamCreateWithPriority")?;
            Ok(Self { stream, device_id: device.id() })
        }
        #[cfg(not(feature = "cuda"))]
        { let _ = priority; Ok(Self { stream: 1 as ffi::CudaStream, device_id: device.id() }) }
    }

    #[inline] pub fn raw(&self) -> ffi::CudaStream { self.stream }
    #[inline] pub fn device_id(&self) -> i32 { self.device_id }

    pub fn synchronize(&self) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaStreamSynchronize(self.stream) }, "cudaStreamSynchronize") }
        #[cfg(not(feature = "cuda"))]
        { Ok(()) }
    }

    pub fn is_complete(&self) -> GpuResult<bool> {
        #[cfg(feature = "cuda")]
        { let r = unsafe { ffi::cudaStreamQuery(self.stream) }; if r == ffi::CUDA_SUCCESS { Ok(true) } else if r == 600 { Ok(false) } else { Err(cuda_error(r, "cudaStreamQuery")) } }
        #[cfg(not(feature = "cuda"))]
        { Ok(true) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        { if !self.stream.is_null() { unsafe { let _ = ffi::cudaStreamDestroy(self.stream); } } }
    }
}

unsafe impl Send for CudaStream {}

/// Event creation flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventFlags { pub blocking_sync: bool, pub disable_timing: bool }

/// A CUDA event for synchronization.
pub struct CudaEvent { event: ffi::CudaEvent }

impl CudaEvent {
    pub fn new() -> GpuResult<Self> { Self::with_flags(EventFlags::default()) }
    pub fn new_sync() -> GpuResult<Self> { Self::with_flags(EventFlags { disable_timing: true, ..Default::default() }) }

    pub fn with_flags(flags: EventFlags) -> GpuResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let mut event: ffi::CudaEvent = ptr::null_mut();
            let mut f = 0u32;
            if flags.disable_timing { f |= ffi::CUDA_EVENT_DISABLE_TIMING; }
            check_cuda(unsafe { ffi::cudaEventCreateWithFlags(&mut event, f) }, "cudaEventCreateWithFlags")?;
            Ok(Self { event })
        }
        #[cfg(not(feature = "cuda"))]
        { let _ = flags; Ok(Self { event: 1 as ffi::CudaEvent }) }
    }

    #[inline] pub fn raw(&self) -> ffi::CudaEvent { self.event }

    pub fn record(&self, stream: &CudaStream) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaEventRecord(self.event, stream.raw()) }, "cudaEventRecord") }
        #[cfg(not(feature = "cuda"))]
        { let _ = stream; Ok(()) }
    }

    pub fn synchronize(&self) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        { check_cuda(unsafe { ffi::cudaEventSynchronize(self.event) }, "cudaEventSynchronize") }
        #[cfg(not(feature = "cuda"))]
        { Ok(()) }
    }

    pub fn is_complete(&self) -> GpuResult<bool> {
        #[cfg(feature = "cuda")]
        { let r = unsafe { ffi::cudaEventQuery(self.event) }; if r == ffi::CUDA_SUCCESS { Ok(true) } else if r == 600 { Ok(false) } else { Err(cuda_error(r, "cudaEventQuery")) } }
        #[cfg(not(feature = "cuda"))]
        { Ok(true) }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        { if !self.event.is_null() { unsafe { let _ = ffi::cudaEventDestroy(self.event); } } }
    }
}

unsafe impl Send for CudaEvent {}

/// Flags for pinned memory allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostAllocFlags { pub portable: bool, pub mapped: bool, pub write_combined: bool }

/// Allocate pinned (page-locked) host memory.
pub fn host_alloc(size: usize, flags: HostAllocFlags) -> GpuResult<*mut u8> {
    #[cfg(feature = "cuda")]
    {
        let mut ptr: *mut c_void = ptr::null_mut();
        let mut f = 0u32;
        if flags.portable { f |= ffi::CUDA_HOST_ALLOC_PORTABLE; }
        if flags.write_combined { f |= ffi::CUDA_HOST_ALLOC_WRITE_COMBINED; }
        check_cuda(unsafe { ffi::cudaHostAlloc(&mut ptr, size, f) }, "cudaHostAlloc")?;
        Ok(ptr as *mut u8)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = flags;
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|_| GpuError::PinnedAllocationFailed { requested_bytes: size, reason: Some("invalid layout".into()) })?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() { return Err(GpuError::PinnedAllocationFailed { requested_bytes: size, reason: Some("allocation failed".into()) }); }
        Ok(ptr)
    }
}

/// Free pinned host memory.
pub fn host_free(ptr: *mut u8, size: usize) -> GpuResult<()> {
    #[cfg(feature = "cuda")]
    { let _ = size; check_cuda(unsafe { ffi::cudaFreeHost(ptr as *mut c_void) }, "cudaFreeHost") }
    #[cfg(not(feature = "cuda"))]
    { let layout = std::alloc::Layout::from_size_align(size, 4096).unwrap(); unsafe { std::alloc::dealloc(ptr, layout); } Ok(()) }
}

/// Copy data from host to device asynchronously.
pub fn memcpy_h2d_async(dst: *mut c_void, src: *const u8, size: usize, stream: &CudaStream) -> GpuResult<()> {
    #[cfg(feature = "cuda")]
    { check_cuda(unsafe { ffi::cudaMemcpyAsync(dst, src as *const c_void, size, ffi::CUDA_MEMCPY_HOST_TO_DEVICE, stream.raw()) }, "cudaMemcpyAsync (H2D)") }
    #[cfg(not(feature = "cuda"))]
    { unsafe { ptr::copy_nonoverlapping(src, dst as *mut u8, size); } let _ = stream; Ok(()) }
}

pub fn init() -> GpuResult<()> {
    if CUDA_INITIALIZED.load(Ordering::Acquire) { return Ok(()); }
    #[cfg(feature = "cuda")]
    { let mut count = 0; let r = unsafe { ffi::cudaGetDeviceCount(&mut count) }; if r != ffi::CUDA_SUCCESS { return Err(GpuError::CudaNotAvailable { reason: format!("error {}", r) }); } if count == 0 { return Err(GpuError::CudaNotAvailable { reason: "no devices".into() }); } }
    CUDA_INITIALIZED.store(true, Ordering::Release);
    Ok(())
}

pub fn is_available() -> bool { init().is_ok() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_device_count() { assert!(CudaDevice::count().unwrap() >= 0); }
    #[test] fn test_device_new() { assert_eq!(CudaDevice::new(0).unwrap().id(), 0); }
    #[test] fn test_device_invalid() { assert!(CudaDevice::new(9999).is_err()); }
    #[test] fn test_stream_create() { assert!(CudaStream::new().unwrap().is_complete().unwrap()); }
    #[test] fn test_event_create() { assert!(CudaEvent::new().unwrap().is_complete().unwrap()); }
    #[test] fn test_device_alloc_free() { let d = CudaDevice::new(0).unwrap(); let p = d.alloc(1024).unwrap(); assert!(!p.as_ptr().is_null()); }
    #[test] fn test_host_alloc_free() { let p = host_alloc(4096, HostAllocFlags::default()).unwrap(); assert!(!p.is_null()); host_free(p, 4096).unwrap(); }
    #[test] fn test_device_copy() { let d = CudaDevice::new(0).unwrap(); let p = d.alloc(1024).unwrap(); let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect(); p.copy_from_host(&src).unwrap(); let mut dst = vec![0u8; 1024]; p.copy_to_host(&mut dst).unwrap(); assert_eq!(src, dst); }
}
