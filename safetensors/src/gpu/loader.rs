//! High-level GPU tensor loader with async support.
//!
//! This module provides the main API for loading safetensors directly to GPU memory
//! with support for async loading, double-buffered transfers, and progress tracking.

#![allow(missing_docs)]

use super::cuda::{memcpy_h2d_async, CudaDevice, CudaStream, DevicePtr};
use super::error::{GpuError, GpuResult};
use super::memory::{PinnedMemoryPool, PinnedPoolConfig};
use super::stream::{CudaStreamPool, StreamPoolConfig};
use crate::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A tensor stored in GPU memory.
#[derive(Debug)]
pub struct GpuTensor {
    ptr: DevicePtr,
    dtype: Dtype,
    shape: Vec<usize>,
    device_id: i32,
}

impl GpuTensor {
    /// Create a new GPU tensor (allocates GPU memory).
    pub fn new(device: &CudaDevice, dtype: Dtype, shape: Vec<usize>) -> GpuResult<Self> {
        let n_elements: usize = shape.iter().product();
        let nbits = n_elements * dtype.bitsize();
        let size = (nbits + 7) / 8;
        let ptr = device.alloc(size)?;
        Ok(Self { ptr, dtype, shape, device_id: device.id() })
    }

    #[inline] pub fn as_ptr(&self) -> *mut c_void { self.ptr.as_ptr() }
    #[inline] pub fn dtype(&self) -> Dtype { self.dtype }
    #[inline] pub fn shape(&self) -> &[usize] { &self.shape }
    #[inline] pub fn device_id(&self) -> i32 { self.device_id }
    #[inline] pub fn size_bytes(&self) -> usize { self.ptr.size() }

    pub fn copy_from_host(&self, data: &[u8]) -> GpuResult<()> { self.ptr.copy_from_host(data) }
    pub fn copy_from_host_async(&self, data: &[u8], stream: &CudaStream) -> GpuResult<()> { self.ptr.copy_from_host_async(data, stream) }
    pub fn copy_to_host(&self, dst: &mut [u8]) -> GpuResult<()> { self.ptr.copy_to_host(dst) }
}

/// Configuration for the GPU tensor loader.
#[derive(Debug, Clone)]
pub struct GpuLoaderConfig {
    pub num_streams: usize,
    pub max_pinned_memory: usize,
    pub pinned_size_classes: Vec<usize>,
    pub device_id: i32,
    pub double_buffer: bool,
    pub prefetch_size: usize,
    /// Maximum chunk size for large tensor transfers (default: 256MB)
    pub max_chunk_size: usize,
}

impl Default for GpuLoaderConfig {
    fn default() -> Self {
        Self {
            num_streams: 4,
            max_pinned_memory: 512 << 20,
            pinned_size_classes: vec![1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20],
            device_id: 0,
            double_buffer: true,
            prefetch_size: 64 << 20,
            max_chunk_size: 256 << 20, // 256MB chunks for large tensors
        }
    }
}

impl GpuLoaderConfig {
    pub fn builder() -> GpuLoaderConfigBuilder { GpuLoaderConfigBuilder::default() }
}

/// Builder for GpuLoaderConfig.
#[derive(Debug, Default)]
pub struct GpuLoaderConfigBuilder { config: GpuLoaderConfig }

impl GpuLoaderConfigBuilder {
    pub fn num_streams(mut self, count: usize) -> Self { self.config.num_streams = count; self }
    pub fn max_pinned_memory(mut self, bytes: usize) -> Self { self.config.max_pinned_memory = bytes; self }
    pub fn max_pinned_memory_mb(mut self, mb: usize) -> Self { self.config.max_pinned_memory = mb << 20; self }
    pub fn device_id(mut self, id: i32) -> Self { self.config.device_id = id; self }
    pub fn double_buffer(mut self, enabled: bool) -> Self { self.config.double_buffer = enabled; self }
    pub fn prefetch_size(mut self, bytes: usize) -> Self { self.config.prefetch_size = bytes; self }
    pub fn prefetch_size_mb(mut self, mb: usize) -> Self { self.config.prefetch_size = mb << 20; self }
    pub fn pinned_size_classes(mut self, classes: Vec<usize>) -> Self { self.config.pinned_size_classes = classes; self }
    pub fn max_chunk_size(mut self, bytes: usize) -> Self { self.config.max_chunk_size = bytes; self }
    pub fn max_chunk_size_mb(mut self, mb: usize) -> Self { self.config.max_chunk_size = mb << 20; self }
    pub fn build(self) -> GpuLoaderConfig { self.config }
}

/// Progress information for an ongoing load operation.
#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub total_tensors: usize,
    pub loaded_tensors: usize,
    pub total_bytes: usize,
    pub transferred_bytes: usize,
    pub current_tensor: Option<String>,
}

impl LoadProgress {
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            if self.total_tensors == 0 { 1.0 } else { self.loaded_tensors as f64 / self.total_tensors as f64 }
        } else { self.transferred_bytes as f64 / self.total_bytes as f64 }
    }
    pub fn percentage(&self) -> f64 { self.fraction() * 100.0 }
}

/// Handle for an async load operation.
pub struct LoadHandle { state: Arc<LoadState> }

struct LoadState {
    complete: AtomicBool,
    cancelled: AtomicBool,
    loaded_count: AtomicUsize,
    transferred_bytes: AtomicUsize,
    total_tensors: usize,
    total_bytes: usize,
}

impl LoadHandle {
    fn new(total_tensors: usize, total_bytes: usize) -> Self {
        Self { state: Arc::new(LoadState {
            complete: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            loaded_count: AtomicUsize::new(0),
            transferred_bytes: AtomicUsize::new(0),
            total_tensors,
            total_bytes,
        })}
    }

    pub fn is_complete(&self) -> bool { self.state.complete.load(Ordering::Acquire) }
    pub fn is_cancelled(&self) -> bool { self.state.cancelled.load(Ordering::Acquire) }

    pub fn progress(&self) -> LoadProgress {
        LoadProgress {
            total_tensors: self.state.total_tensors,
            loaded_tensors: self.state.loaded_count.load(Ordering::Relaxed),
            total_bytes: self.state.total_bytes,
            transferred_bytes: self.state.transferred_bytes.load(Ordering::Relaxed),
            current_tensor: None,
        }
    }

    pub fn progress_fraction(&self) -> f64 { self.progress().fraction() }
    pub fn cancel(&self) { self.state.cancelled.store(true, Ordering::Release); }
    fn mark_complete(&self) { self.state.complete.store(true, Ordering::Release); }
    fn add_loaded(&self, bytes: usize) { self.state.loaded_count.fetch_add(1, Ordering::Relaxed); self.state.transferred_bytes.fetch_add(bytes, Ordering::Relaxed); }
    fn should_cancel(&self) -> bool { self.state.cancelled.load(Ordering::Relaxed) }
}

/// High-performance GPU tensor loader.
pub struct GpuLoader {
    stream_pool: CudaStreamPool,
    pinned_pool: PinnedMemoryPool,
    config: GpuLoaderConfig,
    device: CudaDevice,
}

impl GpuLoader {
    /// Create a new GPU loader with the given configuration.
    pub fn new(config: GpuLoaderConfig) -> GpuResult<Self> {
        let device = CudaDevice::new(config.device_id)?;
        device.set_current()?;

        let stream_config = StreamPoolConfig { num_streams: config.num_streams, device_id: config.device_id, ..Default::default() };
        let stream_pool = CudaStreamPool::with_config(stream_config)?;

        let pinned_config = PinnedPoolConfig { max_memory: config.max_pinned_memory, size_classes: config.pinned_size_classes.clone(), portable: true, ..Default::default() };
        let pinned_pool = PinnedMemoryPool::with_config(pinned_config)?;

        Ok(Self { stream_pool, pinned_pool, config, device })
    }

    /// Create a new GPU loader with default configuration.
    pub fn with_defaults() -> GpuResult<Self> { Self::new(GpuLoaderConfig::default()) }

    pub fn config(&self) -> &GpuLoaderConfig { &self.config }
    pub fn device(&self) -> &CudaDevice { &self.device }

    /// Load all tensors from a safetensor file to GPU memory.
    pub fn load_file<P: AsRef<Path>>(&self, path: P) -> GpuResult<HashMap<String, GpuTensor>> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.load_from_buffer(&mmap)
    }

    /// Load tensors from an in-memory buffer.
    pub fn load_from_buffer(&self, buffer: &[u8]) -> GpuResult<HashMap<String, GpuTensor>> {
        let safetensors = SafeTensors::deserialize(buffer)?;
        self.load_safetensors(&safetensors)
    }

    /// Load tensors from a parsed SafeTensors structure.
    pub fn load_safetensors(&self, safetensors: &SafeTensors) -> GpuResult<HashMap<String, GpuTensor>> {
        self.device.set_current()?;
        let mut tensors = HashMap::new();

        if self.config.double_buffer && self.config.num_streams >= 2 {
            self.load_double_buffered(safetensors, &mut tensors)?;
        } else {
            self.load_sequential(safetensors, &mut tensors)?;
        }

        self.stream_pool.synchronize_all()?;
        Ok(tensors)
    }

    /// Load specific tensors by name.
    pub fn load_tensors(&self, safetensors: &SafeTensors, names: &[&str]) -> GpuResult<HashMap<String, GpuTensor>> {
        self.device.set_current()?;
        let mut tensors = HashMap::with_capacity(names.len());

        for name in names {
            let view = safetensors.tensor(name)?;
            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let data = view.data();

            // Use chunked transfer for large tensors
            if data.len() > self.config.max_chunk_size {
                self.transfer_chunked(&gpu_tensor, data)?;
            } else {
                let stream = self.stream_pool.next_stream();
                let mut pinned = self.pinned_pool.acquire(data.len())?;
                pinned.copy_from_slice(data)?;
                memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), data.len(), stream)?;
                self.pinned_pool.release(pinned);
            }
            tensors.insert(name.to_string(), gpu_tensor);
        }

        self.stream_pool.synchronize_all()?;
        Ok(tensors)
    }

    /// Transfer large tensor data in chunks using double-buffered async copies.
    fn transfer_chunked(&self, gpu_tensor: &GpuTensor, data: &[u8]) -> GpuResult<()> {
        let chunk_size = self.config.max_chunk_size;
        let total_size = data.len();

        // Acquire two buffers for double-buffering
        let mut buffer_a = self.pinned_pool.acquire(chunk_size)?;
        let mut buffer_b = self.pinned_pool.acquire(chunk_size)?;

        let stream_a = self.stream_pool.next_stream();
        let stream_b = self.stream_pool.next_stream();

        let mut current_buffer = &mut buffer_a;
        let mut current_stream = stream_a;
        let mut other_buffer = &mut buffer_b;
        let mut other_stream = stream_b;

        let mut offset = 0usize;
        let gpu_ptr = gpu_tensor.as_ptr() as usize;

        // Process first chunk
        if offset < total_size {
            let end = (offset + chunk_size).min(total_size);
            let chunk_data = &data[offset..end];
            current_buffer.copy_from_slice(chunk_data)?;
            let dst_ptr = (gpu_ptr + offset) as *mut c_void;
            memcpy_h2d_async(dst_ptr, current_buffer.as_ptr(), chunk_data.len(), current_stream)?;
            offset = end;
        }

        // Process remaining chunks with double-buffering
        while offset < total_size {
            std::mem::swap(&mut current_buffer, &mut other_buffer);
            std::mem::swap(&mut current_stream, &mut other_stream);

            let end = (offset + chunk_size).min(total_size);
            let chunk_data = &data[offset..end];

            // Wait for current stream before reusing its buffer
            current_stream.synchronize()?;

            current_buffer.copy_from_slice(chunk_data)?;
            let dst_ptr = (gpu_ptr + offset) as *mut c_void;
            memcpy_h2d_async(dst_ptr, current_buffer.as_ptr(), chunk_data.len(), current_stream)?;

            offset = end;
        }

        // Wait for all transfers to complete
        stream_a.synchronize()?;
        stream_b.synchronize()?;

        self.pinned_pool.release(buffer_a);
        self.pinned_pool.release(buffer_b);

        Ok(())
    }

    fn load_sequential(&self, safetensors: &SafeTensors, tensors: &mut HashMap<String, GpuTensor>) -> GpuResult<()> {
        for (name, view) in safetensors.iter() {
            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let data = view.data();

            // Use chunked transfer for large tensors
            if data.len() > self.config.max_chunk_size {
                self.transfer_chunked(&gpu_tensor, data)?;
            } else {
                let stream = self.stream_pool.next_stream();
                let mut pinned = self.pinned_pool.acquire(data.len())?;
                pinned.copy_from_slice(data)?;
                memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), data.len(), stream)?;
                self.pinned_pool.release(pinned);
            }
            tensors.insert(name.to_string(), gpu_tensor);
        }
        Ok(())
    }

    fn load_double_buffered(&self, safetensors: &SafeTensors, tensors: &mut HashMap<String, GpuTensor>) -> GpuResult<()> {
        let tensor_list: Vec<_> = safetensors.iter().collect();
        if tensor_list.is_empty() { return Ok(()); }

        // Cap buffer size at max_chunk_size for memory efficiency
        let max_size = tensor_list.iter()
            .map(|(_, v)| v.data().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_chunk_size);
        let mut buffer_a = self.pinned_pool.acquire(max_size)?;
        let mut buffer_b = self.pinned_pool.acquire(max_size)?;

        let stream_a = self.stream_pool.next_stream();
        let stream_b = self.stream_pool.next_stream();

        let mut current_buffer = &mut buffer_a;
        let mut current_stream = stream_a;
        let mut other_buffer = &mut buffer_b;
        let mut other_stream = stream_b;

        if let Some((name, view)) = tensor_list.first() {
            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let data = view.data();

            if data.len() > self.config.max_chunk_size {
                // Release buffers temporarily and use chunked transfer
                self.pinned_pool.release(buffer_a);
                self.pinned_pool.release(buffer_b);
                self.transfer_chunked(&gpu_tensor, data)?;
                // Re-acquire buffers for remaining tensors
                buffer_a = self.pinned_pool.acquire(max_size)?;
                buffer_b = self.pinned_pool.acquire(max_size)?;
                current_buffer = &mut buffer_a;
                other_buffer = &mut buffer_b;
            } else {
                current_buffer.copy_from_slice(data)?;
                memcpy_h2d_async(gpu_tensor.as_ptr(), current_buffer.as_ptr(), data.len(), current_stream)?;
            }
            tensors.insert(name.to_string(), gpu_tensor);
        }

        for (name, view) in tensor_list.iter().skip(1) {
            std::mem::swap(&mut current_buffer, &mut other_buffer);
            std::mem::swap(&mut current_stream, &mut other_stream);

            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let data = view.data();

            if data.len() > self.config.max_chunk_size {
                // Use chunked transfer for large tensors
                current_stream.synchronize()?;
                self.transfer_chunked(&gpu_tensor, data)?;
            } else {
                current_stream.synchronize()?;
                current_buffer.copy_from_slice(data)?;
                memcpy_h2d_async(gpu_tensor.as_ptr(), current_buffer.as_ptr(), data.len(), current_stream)?;
            }
            tensors.insert(name.to_string(), gpu_tensor);
        }

        self.pinned_pool.release(buffer_a);
        self.pinned_pool.release(buffer_b);
        Ok(())
    }

    /// Load file with progress tracking.
    pub fn load_file_with_progress<P: AsRef<Path>>(&self, path: P) -> GpuResult<(HashMap<String, GpuTensor>, LoadHandle)> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;

        let total_tensors = safetensors.len();
        let total_bytes: usize = safetensors.iter().map(|(_, v)| v.data().len()).sum();
        let handle = LoadHandle::new(total_tensors, total_bytes);

        let mut tensors = HashMap::with_capacity(total_tensors);
        self.device.set_current()?;

        for (name, view) in safetensors.iter() {
            if handle.should_cancel() { return Err(GpuError::Cancelled); }

            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let stream = self.stream_pool.next_stream();
            let mut pinned = self.pinned_pool.acquire(view.data().len())?;
            pinned.copy_from_slice(view.data())?;
            memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), view.data().len(), stream)?;
            self.pinned_pool.release(pinned);
            handle.add_loaded(view.data().len());
            tensors.insert(name.to_string(), gpu_tensor);
        }

        self.stream_pool.synchronize_all()?;
        handle.mark_complete();
        Ok((tensors, handle))
    }

    /// Load multiple safetensor files (for sharded models).
    pub fn load_sharded<P: AsRef<Path>>(&self, paths: &[P]) -> GpuResult<HashMap<String, GpuTensor>> {
        let mut all_tensors = HashMap::new();
        for path in paths { all_tensors.extend(self.load_file(path)?); }
        Ok(all_tensors)
    }

    pub fn stats(&self) -> LoaderStats {
        LoaderStats {
            num_streams: self.stream_pool.len(),
            pinned_memory_allocated: self.pinned_pool.allocated(),
            pinned_memory_max: self.pinned_pool.max_memory(),
            free_pinned_buffers: self.pinned_pool.free_buffer_count(),
        }
    }

    pub fn synchronize(&self) -> GpuResult<()> { self.stream_pool.synchronize_all() }
    pub fn release_caches(&self) { self.pinned_pool.clear(); }
}

/// Statistics about loader resource usage.
#[derive(Debug, Clone)]
pub struct LoaderStats {
    pub num_streams: usize,
    pub pinned_memory_allocated: usize,
    pub pinned_memory_max: usize,
    pub free_pinned_buffers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_config_builder() { let c = GpuLoaderConfig::builder().num_streams(8).max_pinned_memory_mb(256).device_id(0).double_buffer(true).build(); assert_eq!(c.num_streams, 8); assert_eq!(c.max_pinned_memory, 256 << 20); assert!(c.double_buffer); }
    #[test] fn test_progress() { let p = LoadProgress { total_tensors: 100, loaded_tensors: 50, total_bytes: 1000, transferred_bytes: 500, current_tensor: None }; assert_eq!(p.fraction(), 0.5); assert_eq!(p.percentage(), 50.0); }
    #[test] fn test_progress_empty() { let p = LoadProgress { total_tensors: 0, loaded_tensors: 0, total_bytes: 0, transferred_bytes: 0, current_tensor: None }; assert_eq!(p.fraction(), 1.0); }
    #[test] fn test_load_handle() { let h = LoadHandle::new(10, 1000); assert!(!h.is_complete()); assert!(!h.is_cancelled()); h.add_loaded(100); let p = h.progress(); assert_eq!(p.loaded_tensors, 1); assert_eq!(p.transferred_bytes, 100); h.cancel(); assert!(h.is_cancelled()); h.mark_complete(); assert!(h.is_complete()); }
    #[test] fn test_loader_stats() { let s = LoaderStats { num_streams: 4, pinned_memory_allocated: 1 << 20, pinned_memory_max: 512 << 20, free_pinned_buffers: 2 }; assert_eq!(s.num_streams, 4); }
    #[test] fn test_loader_new() { let c = GpuLoaderConfig::builder().num_streams(2).max_pinned_memory_mb(64).build(); assert!(GpuLoader::new(c).is_ok()); }
    #[test] fn test_gpu_tensor_new() { let d = CudaDevice::new(0).unwrap(); let t = GpuTensor::new(&d, Dtype::F32, vec![2, 3, 4]).unwrap(); assert_eq!(t.dtype(), Dtype::F32); assert_eq!(t.shape(), &[2, 3, 4]); assert_eq!(t.device_id(), 0); assert_eq!(t.size_bytes(), 2 * 3 * 4 * 4); }
}
