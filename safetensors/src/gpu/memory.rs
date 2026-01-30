//! Pinned (page-locked) memory management for fast GPU transfers.
//!
//! This module provides a memory pool for pinned memory buffers, which enable
//! faster DMA transfers between CPU and GPU.

#![allow(missing_docs)]

use super::cuda::{host_alloc, host_free, HostAllocFlags};
use super::error::{GpuError, GpuResult};
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A buffer of pinned (page-locked) memory for fast GPU transfers.
pub struct PinnedBuffer {
    ptr: NonNull<u8>,
    capacity: usize,
    len: usize,
}

impl PinnedBuffer {
    /// Allocate a new pinned buffer with the given capacity.
    pub fn new(capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::InvalidConfig { message: "cannot allocate zero-sized pinned buffer".into() });
        }
        let ptr = host_alloc(capacity, HostAllocFlags::default())?;
        let ptr = NonNull::new(ptr).ok_or_else(|| GpuError::PinnedAllocationFailed {
            requested_bytes: capacity, reason: Some("allocation returned null".into()),
        })?;
        Ok(Self { ptr, capacity, len: 0 })
    }

    /// Allocate a new pinned buffer with portable flag.
    pub fn new_portable(capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::InvalidConfig { message: "cannot allocate zero-sized pinned buffer".into() });
        }
        let ptr = host_alloc(capacity, HostAllocFlags { portable: true, ..Default::default() })?;
        let ptr = NonNull::new(ptr).ok_or_else(|| GpuError::PinnedAllocationFailed {
            requested_bytes: capacity, reason: Some("allocation returned null".into()),
        })?;
        Ok(Self { ptr, capacity, len: 0 })
    }

    /// Allocate a new pinned buffer optimized for CPU-write, GPU-read patterns.
    pub fn new_write_combined(capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::InvalidConfig { message: "cannot allocate zero-sized pinned buffer".into() });
        }
        let ptr = host_alloc(capacity, HostAllocFlags { write_combined: true, portable: true, ..Default::default() })?;
        let ptr = NonNull::new(ptr).ok_or_else(|| GpuError::PinnedAllocationFailed {
            requested_bytes: capacity, reason: Some("allocation returned null".into()),
        })?;
        Ok(Self { ptr, capacity, len: 0 })
    }

    #[inline] pub fn capacity(&self) -> usize { self.capacity }
    #[inline] pub fn len(&self) -> usize { self.len }
    #[inline] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline] pub fn as_ptr(&self) -> *const u8 { self.ptr.as_ptr() }
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut u8 { self.ptr.as_ptr() }
    #[inline] pub fn as_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    #[inline] pub fn as_mut_slice(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }
    #[inline] pub fn clear(&mut self) { self.len = 0; }

    /// Copy data from a slice into the buffer.
    pub fn copy_from_slice(&mut self, src: &[u8]) -> GpuResult<()> {
        if src.len() > self.capacity {
            return Err(GpuError::BufferSizeMismatch { expected: self.capacity, actual: src.len() });
        }
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.as_ptr(), src.len()); self.len = src.len(); }
        Ok(())
    }

    /// Copy data from the buffer to a destination slice.
    pub fn copy_to_slice(&self, dst: &mut [u8]) -> GpuResult<()> {
        if dst.len() < self.len {
            return Err(GpuError::BufferSizeMismatch { expected: self.len, actual: dst.len() });
        }
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), dst.as_mut_ptr(), self.len); }
        Ok(())
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) { let _ = host_free(self.ptr.as_ptr(), self.capacity); }
}

impl Deref for PinnedBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target { self.as_slice() }
}

impl DerefMut for PinnedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut_slice() }
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

/// Configuration for the pinned memory pool.
#[derive(Debug, Clone)]
pub struct PinnedPoolConfig {
    pub max_memory: usize,
    pub size_classes: Vec<usize>,
    pub initial_buffers_per_class: usize,
    pub write_combined: bool,
    pub portable: bool,
}

impl Default for PinnedPoolConfig {
    fn default() -> Self {
        Self {
            max_memory: 1 << 30,
            size_classes: vec![1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20],
            initial_buffers_per_class: 0,
            write_combined: false,
            portable: true,
        }
    }
}

impl PinnedPoolConfig {
    pub fn builder() -> PinnedPoolConfigBuilder { PinnedPoolConfigBuilder::default() }
}

/// Builder for PinnedPoolConfig.
#[derive(Debug, Default)]
pub struct PinnedPoolConfigBuilder { config: PinnedPoolConfig }

impl PinnedPoolConfigBuilder {
    pub fn max_memory(mut self, bytes: usize) -> Self { self.config.max_memory = bytes; self }
    pub fn max_memory_mb(mut self, mb: usize) -> Self { self.config.max_memory = mb << 20; self }
    pub fn size_classes(mut self, classes: Vec<usize>) -> Self { self.config.size_classes = classes; self }
    pub fn initial_buffers_per_class(mut self, count: usize) -> Self { self.config.initial_buffers_per_class = count; self }
    pub fn write_combined(mut self, enabled: bool) -> Self { self.config.write_combined = enabled; self }
    pub fn portable(mut self, enabled: bool) -> Self { self.config.portable = enabled; self }
    pub fn build(self) -> PinnedPoolConfig { self.config }
}

/// A pool of pinned memory buffers for reuse.
pub struct PinnedMemoryPool {
    pools: Vec<Mutex<VecDeque<PinnedBuffer>>>,
    size_classes: Vec<usize>,
    allocated: AtomicUsize,
    max_memory: usize,
    write_combined: bool,
    portable: bool,
}

impl PinnedMemoryPool {
    pub fn new() -> GpuResult<Self> { Self::with_config(PinnedPoolConfig::default()) }
    pub fn with_max_memory(max_memory: usize) -> GpuResult<Self> { Self::with_config(PinnedPoolConfig { max_memory, ..Default::default() }) }

    pub fn with_config(config: PinnedPoolConfig) -> GpuResult<Self> {
        let mut size_classes = config.size_classes;
        size_classes.sort_unstable();
        size_classes.dedup();
        if size_classes.is_empty() {
            return Err(GpuError::InvalidConfig { message: "at least one size class is required".into() });
        }
        let pools: Vec<_> = size_classes.iter().map(|_| Mutex::new(VecDeque::new())).collect();
        let pool = Self { pools, size_classes, allocated: AtomicUsize::new(0), max_memory: config.max_memory, write_combined: config.write_combined, portable: config.portable };

        if config.initial_buffers_per_class > 0 {
            for (i, &size) in pool.size_classes.iter().enumerate() {
                for _ in 0..config.initial_buffers_per_class {
                    if pool.allocated.load(Ordering::Relaxed) + size > pool.max_memory { break; }
                    if let Ok(buffer) = pool.allocate_buffer(size) { pool.pools[i].lock().unwrap().push_back(buffer); }
                }
            }
        }
        Ok(pool)
    }

    pub fn allocated(&self) -> usize { self.allocated.load(Ordering::Relaxed) }
    pub fn max_memory(&self) -> usize { self.max_memory }
    pub fn free_buffer_count(&self) -> usize { self.pools.iter().map(|p| p.lock().unwrap().len()).sum() }

    fn size_class_index(&self, size: usize) -> Option<usize> { self.size_classes.iter().position(|&class| class >= size) }

    fn allocate_buffer(&self, capacity: usize) -> GpuResult<PinnedBuffer> {
        let buffer = if self.write_combined { PinnedBuffer::new_write_combined(capacity)? }
                     else if self.portable { PinnedBuffer::new_portable(capacity)? }
                     else { PinnedBuffer::new(capacity)? };
        self.allocated.fetch_add(capacity, Ordering::Relaxed);
        Ok(buffer)
    }

    /// Acquire a buffer of at least the given size.
    pub fn acquire(&self, min_size: usize) -> GpuResult<PinnedBuffer> {
        let class_idx = self.size_class_index(min_size).ok_or_else(|| GpuError::InvalidConfig {
            message: format!("requested size {} exceeds largest size class {}", min_size, self.size_classes.last().unwrap_or(&0)),
        })?;
        let capacity = self.size_classes[class_idx];

        { let mut pool = self.pools[class_idx].lock().unwrap(); if let Some(mut buffer) = pool.pop_front() { buffer.clear(); return Ok(buffer); } }

        let current = self.allocated.load(Ordering::Relaxed);
        if current + capacity > self.max_memory {
            return Err(GpuError::OutOfPinnedMemory { requested_bytes: capacity, allocated_bytes: current, max_bytes: self.max_memory });
        }
        self.allocate_buffer(capacity)
    }

    /// Release a buffer back to the pool for reuse.
    pub fn release(&self, mut buffer: PinnedBuffer) {
        buffer.clear();
        if let Some(class_idx) = self.size_class_index(buffer.capacity()) {
            if self.size_classes[class_idx] == buffer.capacity() {
                self.pools[class_idx].lock().unwrap().push_back(buffer);
                return;
            }
        }
        self.allocated.fetch_sub(buffer.capacity(), Ordering::Relaxed);
    }

    /// Clear all pooled buffers.
    pub fn clear(&self) {
        for pool in &self.pools {
            for buffer in pool.lock().unwrap().drain(..) { self.allocated.fetch_sub(buffer.capacity(), Ordering::Relaxed); }
        }
    }

    /// Shrink the pool by releasing buffers until allocated memory is below target.
    pub fn shrink_to(&self, target_bytes: usize) {
        while self.allocated.load(Ordering::Relaxed) > target_bytes {
            let mut released = false;
            for pool in self.pools.iter().rev() {
                if let Some(buffer) = pool.lock().unwrap().pop_back() {
                    self.allocated.fetch_sub(buffer.capacity(), Ordering::Relaxed);
                    released = true;
                    break;
                }
            }
            if !released { break; }
        }
    }
}

impl Default for PinnedMemoryPool {
    fn default() -> Self { Self::new().expect("failed to create default pinned memory pool") }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) { self.clear(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_pinned_buffer_new() { let b = PinnedBuffer::new(4096).unwrap(); assert_eq!(b.capacity(), 4096); assert!(b.is_empty()); }
    #[test] fn test_pinned_buffer_zero_size() { assert!(PinnedBuffer::new(0).is_err()); }
    #[test] fn test_pinned_buffer_copy() { let mut b = PinnedBuffer::new(1024).unwrap(); let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect(); b.copy_from_slice(&data).unwrap(); assert_eq!(b.len(), 512); assert_eq!(b.as_slice(), &data[..]); }
    #[test] fn test_pinned_buffer_overflow() { let mut b = PinnedBuffer::new(256).unwrap(); assert!(b.copy_from_slice(&vec![0u8; 512]).is_err()); }
    #[test] fn test_pool_config_builder() { let c = PinnedPoolConfig::builder().max_memory_mb(512).write_combined(true).build(); assert_eq!(c.max_memory, 512 << 20); assert!(c.write_combined); }
    #[test] fn test_pool_acquire_release() { let pool = PinnedMemoryPool::with_config(PinnedPoolConfig { max_memory: 10 << 20, size_classes: vec![1 << 20], ..Default::default() }).unwrap(); let b = pool.acquire(512 << 10).unwrap(); assert_eq!(b.capacity(), 1 << 20); pool.release(b); assert_eq!(pool.free_buffer_count(), 1); }
    #[test] fn test_pool_memory_limit() { let pool = PinnedMemoryPool::with_config(PinnedPoolConfig { max_memory: 2 << 20, size_classes: vec![1 << 20], ..Default::default() }).unwrap(); let b1 = pool.acquire(1).unwrap(); let b2 = pool.acquire(1).unwrap(); assert!(matches!(pool.acquire(1), Err(GpuError::OutOfPinnedMemory { .. }))); pool.release(b1); pool.release(b2); }
    #[test] fn test_pool_clear() { let pool = PinnedMemoryPool::with_config(PinnedPoolConfig { max_memory: 10 << 20, size_classes: vec![1 << 20], initial_buffers_per_class: 2, ..Default::default() }).unwrap(); assert!(pool.allocated() > 0); pool.clear(); assert_eq!(pool.allocated(), 0); }
}
