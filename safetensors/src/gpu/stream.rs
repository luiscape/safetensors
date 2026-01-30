//! CUDA stream pool for managing multiple async streams.
//!
//! This module provides a pool of CUDA streams for parallel async operations.
//! Using multiple streams allows overlapping memory transfers and kernel execution.

#![allow(missing_docs)]

use super::cuda::{CudaDevice, CudaEvent, CudaStream, EventFlags, StreamPriority};
use super::error::{GpuError, GpuResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for the stream pool.
#[derive(Debug, Clone)]
pub struct StreamPoolConfig {
    pub num_streams: usize,
    pub priority: StreamPriority,
    pub device_id: i32,
}

impl Default for StreamPoolConfig {
    fn default() -> Self {
        Self { num_streams: 4, priority: StreamPriority::Normal, device_id: 0 }
    }
}

impl StreamPoolConfig {
    pub fn builder() -> StreamPoolConfigBuilder { StreamPoolConfigBuilder::default() }
}

/// Builder for StreamPoolConfig.
#[derive(Debug, Default)]
pub struct StreamPoolConfigBuilder { config: StreamPoolConfig }

impl StreamPoolConfigBuilder {
    pub fn num_streams(mut self, count: usize) -> Self { self.config.num_streams = count; self }
    pub fn priority(mut self, priority: StreamPriority) -> Self { self.config.priority = priority; self }
    pub fn device_id(mut self, device_id: i32) -> Self { self.config.device_id = device_id; self }
    pub fn build(self) -> StreamPoolConfig { self.config }
}

/// A pool of CUDA streams for parallel async operations.
pub struct CudaStreamPool {
    streams: Vec<CudaStream>,
    events: Vec<CudaEvent>,
    current: AtomicUsize,
    device_id: i32,
}

impl CudaStreamPool {
    pub fn new(num_streams: usize) -> GpuResult<Self> {
        Self::with_config(StreamPoolConfig { num_streams, ..Default::default() })
    }

    pub fn new_high_priority(num_streams: usize) -> GpuResult<Self> {
        Self::with_config(StreamPoolConfig { num_streams, priority: StreamPriority::High, ..Default::default() })
    }

    pub fn with_config(config: StreamPoolConfig) -> GpuResult<Self> {
        if config.num_streams == 0 {
            return Err(GpuError::InvalidConfig { message: "stream pool must have at least one stream".into() });
        }

        let device = CudaDevice::new(config.device_id)?;
        device.set_current()?;

        let mut streams = Vec::with_capacity(config.num_streams);
        for _ in 0..config.num_streams {
            streams.push(CudaStream::with_priority(config.priority)?);
        }

        let mut events = Vec::with_capacity(config.num_streams);
        for _ in 0..config.num_streams {
            events.push(CudaEvent::with_flags(EventFlags { disable_timing: true, ..Default::default() })?);
        }

        Ok(Self { streams, events, current: AtomicUsize::new(0), device_id: config.device_id })
    }

    #[inline] pub fn len(&self) -> usize { self.streams.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.streams.is_empty() }
    #[inline] pub fn device_id(&self) -> i32 { self.device_id }

    /// Get the next stream in round-robin fashion.
    pub fn next_stream(&self) -> &CudaStream {
        let idx = self.current.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        &self.streams[idx]
    }

    /// Get stream and event at the given index.
    pub fn stream_with_event(&self, index: usize) -> Option<(&CudaStream, &CudaEvent)> {
        if index < self.streams.len() { Some((&self.streams[index], &self.events[index])) } else { None }
    }

    /// Get the next stream and its associated event in round-robin fashion.
    pub fn next_stream_with_event(&self) -> (&CudaStream, &CudaEvent) {
        let idx = self.current.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        (&self.streams[idx], &self.events[idx])
    }

    pub fn stream(&self, index: usize) -> Option<&CudaStream> { self.streams.get(index) }
    pub fn event(&self, index: usize) -> Option<&CudaEvent> { self.events.get(index) }
    pub fn iter(&self) -> impl Iterator<Item = &CudaStream> { self.streams.iter() }
    pub fn iter_with_events(&self) -> impl Iterator<Item = (&CudaStream, &CudaEvent)> { self.streams.iter().zip(self.events.iter()) }

    pub fn synchronize(&self, index: usize) -> GpuResult<()> {
        self.streams.get(index).ok_or_else(|| GpuError::InvalidConfig {
            message: format!("stream index {} out of bounds", index),
        })?.synchronize()
    }

    /// Synchronize all streams in the pool.
    pub fn synchronize_all(&self) -> GpuResult<()> {
        for stream in &self.streams { stream.synchronize()?; }
        Ok(())
    }

    /// Check if all streams have completed their operations.
    pub fn all_complete(&self) -> GpuResult<bool> {
        for stream in &self.streams { if !stream.is_complete()? { return Ok(false); } }
        Ok(true)
    }

    /// Record events on all streams.
    pub fn record_all_events(&self) -> GpuResult<()> {
        for (stream, event) in self.streams.iter().zip(self.events.iter()) { event.record(stream)?; }
        Ok(())
    }

    /// Wait for all recorded events to complete.
    pub fn wait_all_events(&self) -> GpuResult<()> {
        for event in &self.events { event.synchronize()?; }
        Ok(())
    }

    /// Reset the round-robin counter.
    pub fn reset_counter(&self) { self.current.store(0, Ordering::Relaxed); }
}

/// A thread-safe, reference-counted stream pool.
#[derive(Clone)]
pub struct SharedStreamPool { inner: Arc<CudaStreamPool> }

impl SharedStreamPool {
    pub fn new(num_streams: usize) -> GpuResult<Self> { Ok(Self { inner: Arc::new(CudaStreamPool::new(num_streams)?) }) }
    pub fn with_config(config: StreamPoolConfig) -> GpuResult<Self> { Ok(Self { inner: Arc::new(CudaStreamPool::with_config(config)?) }) }
    pub fn ref_count(&self) -> usize { Arc::strong_count(&self.inner) }
}

impl std::ops::Deref for SharedStreamPool {
    type Target = CudaStreamPool;
    fn deref(&self) -> &Self::Target { &self.inner }
}

/// Stream pools for multiple GPU devices.
pub struct MultiDeviceStreamPool { pools: Vec<CudaStreamPool> }

impl MultiDeviceStreamPool {
    pub fn new(streams_per_device: usize) -> GpuResult<Self> {
        let num_devices = CudaDevice::count()?;
        Self::for_devices(streams_per_device, (0..num_devices).collect())
    }

    pub fn for_devices(streams_per_device: usize, device_ids: Vec<i32>) -> GpuResult<Self> {
        let mut pools = Vec::with_capacity(device_ids.len());
        for device_id in device_ids {
            pools.push(CudaStreamPool::with_config(StreamPoolConfig { num_streams: streams_per_device, device_id, ..Default::default() })?);
        }
        Ok(Self { pools })
    }

    #[inline] pub fn num_devices(&self) -> usize { self.pools.len() }
    pub fn pool(&self, device_id: i32) -> Option<&CudaStreamPool> { self.pools.iter().find(|p| p.device_id() == device_id) }
    pub fn pool_by_index(&self, index: usize) -> Option<&CudaStreamPool> { self.pools.get(index) }
    pub fn iter(&self) -> impl Iterator<Item = &CudaStreamPool> { self.pools.iter() }

    pub fn synchronize_all(&self) -> GpuResult<()> {
        for pool in &self.pools { pool.synchronize_all()?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_config_builder() { let c = StreamPoolConfig::builder().num_streams(8).priority(StreamPriority::High).build(); assert_eq!(c.num_streams, 8); assert_eq!(c.priority, StreamPriority::High); }
    #[test] fn test_pool_new() { let pool = CudaStreamPool::new(4).unwrap(); assert_eq!(pool.len(), 4); assert!(!pool.is_empty()); }
    #[test] fn test_pool_zero_streams() { assert!(CudaStreamPool::new(0).is_err()); }
    #[test] fn test_pool_round_robin() { let pool = CudaStreamPool::new(3).unwrap(); let s0 = pool.next_stream() as *const _; let s1 = pool.next_stream() as *const _; let s2 = pool.next_stream() as *const _; let s3 = pool.next_stream() as *const _; assert_ne!(s0, s1); assert_ne!(s1, s2); assert_eq!(s0, s3); }
    #[test] fn test_pool_synchronize_all() { CudaStreamPool::new(4).unwrap().synchronize_all().unwrap(); }
    #[test] fn test_pool_all_complete() { assert!(CudaStreamPool::new(2).unwrap().all_complete().unwrap()); }
    #[test] fn test_pool_reset_counter() { let pool = CudaStreamPool::new(4).unwrap(); let _ = pool.next_stream(); let _ = pool.next_stream(); pool.reset_counter(); let s0 = pool.next_stream() as *const _; assert_eq!(s0, pool.stream(0).unwrap() as *const _); }
    #[test] fn test_pool_iter() { assert_eq!(CudaStreamPool::new(3).unwrap().iter().count(), 3); }
    #[test] fn test_shared_pool() { let pool = SharedStreamPool::new(4).unwrap(); assert_eq!(pool.len(), 4); let pool2 = pool.clone(); assert_eq!(pool.ref_count(), 2); drop(pool2); assert_eq!(pool.ref_count(), 1); }
    #[test] fn test_record_events() { let pool = CudaStreamPool::new(2).unwrap(); pool.record_all_events().unwrap(); pool.wait_all_events().unwrap(); }
    #[test] fn test_stream_with_event() { let pool = CudaStreamPool::new(2).unwrap(); let (stream, event) = pool.stream_with_event(0).unwrap(); event.record(stream).unwrap(); event.synchronize().unwrap(); assert!(pool.stream_with_event(10).is_none()); }
}
