//! Enhanced GPU loader with io_uring support for high-performance I/O.
//!
//! This module provides an optimized tensor loader that uses io_uring on Linux
//! for maximum I/O throughput. It combines:
//! - io_uring for async file reads with minimal syscall overhead
//! - Pinned memory for fast DMA transfers
//! - CUDA streams for parallel GPU uploads
//! - Pipelining to overlap I/O, CPU processing, and GPU transfers

#![allow(missing_docs)]

use super::cuda::{memcpy_h2d_async, CudaDevice, CudaStream};
use super::error::{GpuError, GpuResult};
use super::loader::{GpuLoaderConfig, GpuTensor, LoadProgress, LoaderStats};
use super::memory::{PinnedMemoryPool, PinnedPoolConfig};
use super::stream::{CudaStreamPool, StreamPoolConfig};
use crate::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(feature = "io-uring")]
use super::io_uring::{IoUringConfig, TensorFileReader};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the io_uring-enhanced loader.
#[derive(Debug, Clone)]
pub struct UringLoaderConfig {
    /// Base GPU loader configuration.
    pub gpu: GpuLoaderConfig,
    /// io_uring configuration.
    #[cfg(feature = "io-uring")]
    pub io_uring: IoUringConfig,
    /// Number of concurrent I/O operations.
    pub io_concurrency: usize,
    /// Enable pipelining (overlap I/O with GPU transfers).
    pub pipeline: bool,
    /// Number of pipeline stages.
    pub pipeline_stages: usize,
}

impl Default for UringLoaderConfig {
    fn default() -> Self {
        Self {
            gpu: GpuLoaderConfig::default(),
            #[cfg(feature = "io-uring")]
            io_uring: IoUringConfig::default(),
            io_concurrency: 8,
            pipeline: true,
            pipeline_stages: 3,
        }
    }
}

impl UringLoaderConfig {
    /// Create a new configuration builder.
    pub fn builder() -> UringLoaderConfigBuilder {
        UringLoaderConfigBuilder::default()
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        Self {
            gpu: GpuLoaderConfig {
                num_streams: 8,
                max_pinned_memory: 1 << 30, // 1GB
                double_buffer: true,
                ..Default::default()
            },
            #[cfg(feature = "io-uring")]
            io_uring: IoUringConfig::high_performance(),
            io_concurrency: 16,
            pipeline: true,
            pipeline_stages: 4,
        }
    }
}

/// Builder for UringLoaderConfig.
#[derive(Debug, Default)]
pub struct UringLoaderConfigBuilder {
    config: UringLoaderConfig,
}

impl UringLoaderConfigBuilder {
    /// Set the number of CUDA streams.
    pub fn num_streams(mut self, count: usize) -> Self {
        self.config.gpu.num_streams = count;
        self
    }

    /// Set the maximum pinned memory in bytes.
    pub fn max_pinned_memory(mut self, bytes: usize) -> Self {
        self.config.gpu.max_pinned_memory = bytes;
        self
    }

    /// Set the maximum pinned memory in megabytes.
    pub fn max_pinned_memory_mb(mut self, mb: usize) -> Self {
        self.config.gpu.max_pinned_memory = mb << 20;
        self
    }

    /// Set the device ID.
    pub fn device_id(mut self, id: i32) -> Self {
        self.config.gpu.device_id = id;
        self
    }

    /// Enable or disable double buffering.
    pub fn double_buffer(mut self, enabled: bool) -> Self {
        self.config.gpu.double_buffer = enabled;
        self
    }

    /// Set the I/O concurrency level.
    pub fn io_concurrency(mut self, count: usize) -> Self {
        self.config.io_concurrency = count;
        self
    }

    /// Enable or disable pipelining.
    pub fn pipeline(mut self, enabled: bool) -> Self {
        self.config.pipeline = enabled;
        self
    }

    /// Set the number of pipeline stages.
    pub fn pipeline_stages(mut self, stages: usize) -> Self {
        self.config.pipeline_stages = stages;
        self
    }

    /// Set the io_uring queue depth.
    #[cfg(feature = "io-uring")]
    pub fn queue_depth(mut self, depth: u32) -> Self {
        self.config.io_uring.queue_depth = depth;
        self
    }

    /// Enable or disable SQPOLL for io_uring.
    #[cfg(feature = "io-uring")]
    pub fn sq_poll(mut self, enabled: bool) -> Self {
        self.config.io_uring.sq_poll = enabled;
        self
    }

    /// Enable or disable direct I/O.
    #[cfg(feature = "io-uring")]
    pub fn direct_io(mut self, enabled: bool) -> Self {
        self.config.io_uring.direct_io = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> UringLoaderConfig {
        self.config
    }
}

// ============================================================================
// Pipeline Stage
// ============================================================================

/// A stage in the loading pipeline.
struct PipelineStage {
    /// Tensor name being processed.
    name: String,
    /// Data buffer (from I/O).
    data: Vec<u8>,
    /// GPU tensor (allocated).
    tensor: Option<GpuTensor>,
    /// Stream being used.
    stream_idx: usize,
    /// Whether GPU transfer is in flight.
    transfer_pending: bool,
}

// ============================================================================
// UringLoader - Main loader with io_uring support
// ============================================================================

/// High-performance GPU loader with io_uring support.
///
/// This loader uses io_uring on Linux for async file I/O, combined with
/// CUDA streams and pinned memory for optimal GPU loading performance.
///
/// # Example
///
/// ```ignore
/// use safetensors::gpu::{UringLoader, UringLoaderConfig};
///
/// let config = UringLoaderConfig::high_performance();
/// let loader = UringLoader::new(config)?;
/// let tensors = loader.load_file("model.safetensors")?;
/// ```
pub struct UringLoader {
    stream_pool: CudaStreamPool,
    pinned_pool: PinnedMemoryPool,
    config: UringLoaderConfig,
    device: CudaDevice,
}

impl UringLoader {
    /// Create a new loader with the given configuration.
    pub fn new(config: UringLoaderConfig) -> GpuResult<Self> {
        let device = CudaDevice::new(config.gpu.device_id)?;
        device.set_current()?;

        let stream_config = StreamPoolConfig {
            num_streams: config.gpu.num_streams,
            device_id: config.gpu.device_id,
            ..Default::default()
        };
        let stream_pool = CudaStreamPool::with_config(stream_config)?;

        let pinned_config = PinnedPoolConfig {
            max_memory: config.gpu.max_pinned_memory,
            size_classes: config.gpu.pinned_size_classes.clone(),
            portable: true,
            ..Default::default()
        };
        let pinned_pool = PinnedMemoryPool::with_config(pinned_config)?;

        Ok(Self {
            stream_pool,
            pinned_pool,
            config,
            device,
        })
    }

    /// Create a new loader with default configuration.
    pub fn with_defaults() -> GpuResult<Self> {
        Self::new(UringLoaderConfig::default())
    }

    /// Create a new loader optimized for high performance.
    pub fn high_performance() -> GpuResult<Self> {
        Self::new(UringLoaderConfig::high_performance())
    }

    /// Get the configuration.
    pub fn config(&self) -> &UringLoaderConfig {
        &self.config
    }

    /// Get the target device.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Load all tensors from a safetensor file.
    #[cfg(feature = "io-uring")]
    pub fn load_file<P: AsRef<Path>>(&self, path: P) -> GpuResult<HashMap<String, GpuTensor>> {
        use super::io_uring::is_available;

        if is_available() && self.config.pipeline {
            self.load_file_pipelined(path)
        } else {
            self.load_file_standard(path)
        }
    }

    /// Load all tensors from a safetensor file (non-io_uring version).
    #[cfg(not(feature = "io-uring"))]
    pub fn load_file<P: AsRef<Path>>(&self, path: P) -> GpuResult<HashMap<String, GpuTensor>> {
        self.load_file_standard(path)
    }

    /// Standard loading path (uses mmap).
    fn load_file_standard<P: AsRef<Path>>(&self, path: P) -> GpuResult<HashMap<String, GpuTensor>> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        self.load_from_buffer(&mmap)
    }

    /// Pipelined loading with io_uring.
    #[cfg(feature = "io-uring")]
    fn load_file_pipelined<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> GpuResult<HashMap<String, GpuTensor>> {
        use super::io_uring::{ReadRequest, TensorFileReader};

        // First, read and parse the header using mmap (fast for small data)
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let (header_size, metadata) = SafeTensors::read_metadata(&mmap)?;
        let data_offset = 8 + header_size; // 8 bytes for header size + header

        // Get tensor info sorted by offset for sequential reading
        let mut tensor_infos: Vec<_> = metadata
            .tensors()
            .into_iter()
            .map(|(name, info)| {
                let (start, end) = info.data_offsets;
                (name, info.dtype, info.shape.clone(), start, end)
            })
            .collect();
        tensor_infos.sort_by_key(|(_, _, _, start, _)| *start);

        drop(mmap);
        drop(file);

        // Create io_uring reader
        let mut reader = TensorFileReader::new(path, self.config.io_uring.clone())?;

        self.device.set_current()?;
        let mut tensors = HashMap::with_capacity(tensor_infos.len());

        if self.config.pipeline && self.config.pipeline_stages > 1 {
            // Pipelined loading: overlap I/O with GPU transfers
            self.load_pipelined_impl(&mut reader, &tensor_infos, data_offset, &mut tensors)?;
        } else {
            // Batched I/O but sequential GPU uploads
            self.load_batched_impl(&mut reader, &tensor_infos, data_offset, &mut tensors)?;
        }

        self.stream_pool.synchronize_all()?;
        Ok(tensors)
    }

    /// Pipelined implementation: overlap I/O reads with GPU transfers.
    #[cfg(feature = "io-uring")]
    fn load_pipelined_impl(
        &self,
        reader: &mut TensorFileReader,
        tensor_infos: &[(String, Dtype, Vec<usize>, usize, usize)],
        data_offset: usize,
        tensors: &mut HashMap<String, GpuTensor>,
    ) -> GpuResult<()> {
        let num_stages = self.config.pipeline_stages.min(tensor_infos.len());
        let mut stages: Vec<Option<PipelineStage>> = (0..num_stages).map(|_| None).collect();
        let mut next_tensor_idx = 0;
        let mut stream_idx = 0;

        // Fill initial pipeline stages with I/O requests
        for stage_idx in 0..num_stages {
            if next_tensor_idx >= tensor_infos.len() {
                break;
            }

            let (name, dtype, shape, start, end) = &tensor_infos[next_tensor_idx];
            let offset = data_offset + start;
            let length = end - start;

            // Read data
            let data = reader.read_region(offset as u64, length)?;

            // Allocate GPU tensor
            let gpu_tensor = GpuTensor::new(&self.device, *dtype, shape.clone())?;

            stages[stage_idx] = Some(PipelineStage {
                name: name.clone(),
                data,
                tensor: Some(gpu_tensor),
                stream_idx,
                transfer_pending: false,
            });

            next_tensor_idx += 1;
            stream_idx = (stream_idx + 1) % self.stream_pool.len();
        }

        // Process pipeline
        loop {
            let mut any_active = false;

            for stage_idx in 0..num_stages {
                if let Some(stage) = &mut stages[stage_idx] {
                    any_active = true;

                    if !stage.transfer_pending {
                        // Start GPU transfer
                        let stream = self.stream_pool.stream(stage.stream_idx).unwrap();
                        let mut pinned = self.pinned_pool.acquire(stage.data.len())?;
                        pinned.copy_from_slice(&stage.data)?;

                        if let Some(ref tensor) = stage.tensor {
                            memcpy_h2d_async(
                                tensor.as_ptr(),
                                pinned.as_ptr(),
                                stage.data.len(),
                                stream,
                            )?;
                        }

                        self.pinned_pool.release(pinned);
                        stage.transfer_pending = true;
                    }

                    // Check if transfer is complete
                    let stream = self.stream_pool.stream(stage.stream_idx).unwrap();
                    if stream.is_complete()? {
                        // Move tensor to result
                        let completed_stage = stages[stage_idx].take().unwrap();
                        if let Some(tensor) = completed_stage.tensor {
                            tensors.insert(completed_stage.name, tensor);
                        }

                        // Load next tensor into this stage
                        if next_tensor_idx < tensor_infos.len() {
                            let (name, dtype, shape, start, end) = &tensor_infos[next_tensor_idx];
                            let offset = data_offset + start;
                            let length = end - start;

                            let data = reader.read_region(offset as u64, length)?;
                            let gpu_tensor = GpuTensor::new(&self.device, *dtype, shape.clone())?;

                            stages[stage_idx] = Some(PipelineStage {
                                name: name.clone(),
                                data,
                                tensor: Some(gpu_tensor),
                                stream_idx: stage.stream_idx,
                                transfer_pending: false,
                            });

                            next_tensor_idx += 1;
                        }
                    }
                }
            }

            if !any_active && next_tensor_idx >= tensor_infos.len() {
                break;
            }
        }

        Ok(())
    }

    /// Batched I/O implementation: read all data first, then upload.
    #[cfg(feature = "io-uring")]
    fn load_batched_impl(
        &self,
        reader: &mut TensorFileReader,
        tensor_infos: &[(String, Dtype, Vec<usize>, usize, usize)],
        data_offset: usize,
        tensors: &mut HashMap<String, GpuTensor>,
    ) -> GpuResult<()> {
        // Prepare regions to read
        let regions: Vec<_> = tensor_infos
            .iter()
            .map(|(_, _, _, start, end)| ((data_offset + start) as u64, end - start))
            .collect();

        // Read all data in parallel
        let data_buffers = reader.read_regions(&regions)?;

        // Upload to GPU
        for ((name, dtype, shape, _, _), data) in tensor_infos.iter().zip(data_buffers) {
            let gpu_tensor = GpuTensor::new(&self.device, *dtype, shape.clone())?;
            let stream = self.stream_pool.next_stream();
            let mut pinned = self.pinned_pool.acquire(data.len())?;

            pinned.copy_from_slice(&data)?;
            memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), data.len(), stream)?;

            self.pinned_pool.release(pinned);
            tensors.insert(name.clone(), gpu_tensor);
        }

        Ok(())
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

        for (name, view) in safetensors.iter() {
            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let stream = self.stream_pool.next_stream();
            let mut pinned = self.pinned_pool.acquire(view.data().len())?;

            pinned.copy_from_slice(view.data())?;
            memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), view.data().len(), stream)?;

            self.pinned_pool.release(pinned);
            tensors.insert(name.to_string(), gpu_tensor);
        }

        self.stream_pool.synchronize_all()?;
        Ok(tensors)
    }

    /// Load specific tensors by name.
    pub fn load_tensors(
        &self,
        safetensors: &SafeTensors,
        names: &[&str],
    ) -> GpuResult<HashMap<String, GpuTensor>> {
        self.device.set_current()?;
        let mut tensors = HashMap::with_capacity(names.len());

        for name in names {
            let view = safetensors.tensor(name)?;
            let gpu_tensor = GpuTensor::new(&self.device, view.dtype(), view.shape().to_vec())?;
            let stream = self.stream_pool.next_stream();
            let mut pinned = self.pinned_pool.acquire(view.data().len())?;

            pinned.copy_from_slice(view.data())?;
            memcpy_h2d_async(gpu_tensor.as_ptr(), pinned.as_ptr(), view.data().len(), stream)?;

            self.pinned_pool.release(pinned);
            tensors.insert(name.to_string(), gpu_tensor);
        }

        self.stream_pool.synchronize_all()?;
        Ok(tensors)
    }

    /// Load multiple safetensor files (for sharded models).
    pub fn load_sharded<P: AsRef<Path>>(&self, paths: &[P]) -> GpuResult<HashMap<String, GpuTensor>> {
        let mut all_tensors = HashMap::new();
        for path in paths {
            all_tensors.extend(self.load_file(path)?);
        }
        Ok(all_tensors)
    }

    /// Get statistics about current resource usage.
    pub fn stats(&self) -> LoaderStats {
        LoaderStats {
            num_streams: self.stream_pool.len(),
            pinned_memory_allocated: self.pinned_pool.allocated(),
            pinned_memory_max: self.pinned_pool.max_memory(),
            free_pinned_buffers: self.pinned_pool.free_buffer_count(),
        }
    }

    /// Synchronize all pending operations.
    pub fn synchronize(&self) -> GpuResult<()> {
        self.stream_pool.synchronize_all()
    }

    /// Release cached resources.
    pub fn release_caches(&self) {
        self.pinned_pool.clear();
    }

    /// Check if io_uring is being used.
    #[cfg(feature = "io-uring")]
    pub fn uses_io_uring(&self) -> bool {
        super::io_uring::is_available()
    }

    /// Check if io_uring is being used (always false without feature).
    #[cfg(not(feature = "io-uring"))]
    pub fn uses_io_uring(&self) -> bool {
        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = UringLoaderConfig::builder()
            .num_streams(8)
            .max_pinned_memory_mb(256)
            .device_id(0)
            .io_concurrency(16)
            .pipeline(true)
            .pipeline_stages(4)
            .build();

        assert_eq!(config.gpu.num_streams, 8);
        assert_eq!(config.gpu.max_pinned_memory, 256 << 20);
        assert_eq!(config.io_concurrency, 16);
        assert!(config.pipeline);
        assert_eq!(config.pipeline_stages, 4);
    }

    #[test]
    fn test_high_performance_config() {
        let config = UringLoaderConfig::high_performance();
        assert!(config.gpu.num_streams >= 4);
        assert!(config.pipeline);
        assert!(config.io_concurrency >= 8);
    }

    #[test]
    fn test_loader_new() {
        let config = UringLoaderConfig::builder()
            .num_streams(2)
            .max_pinned_memory_mb(64)
            .build();

        let loader = UringLoader::new(config);
        assert!(loader.is_ok());
    }

    #[test]
    fn test_loader_stats() {
        let loader = UringLoader::with_defaults().unwrap();
        let stats = loader.stats();
        assert!(stats.num_streams > 0);
    }

    #[test]
    fn test_uses_io_uring() {
        let loader = UringLoader::with_defaults().unwrap();
        // Just verify it doesn't panic
        let _ = loader.uses_io_uring();
    }
}
