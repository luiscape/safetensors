//! io_uring-based async I/O for high-performance file reading on Linux.
//!
//! This module provides io_uring integration for reading safetensor files with
//! minimal syscall overhead and optimal throughput. It supports:
//! - Batched read submissions
//! - Kernel-side polling (SQPOLL) for zero-syscall operation
//! - Direct I/O bypass of page cache
//! - Registered buffers for zero-copy operations
//!
//! # Platform Support
//!
//! This module is only available on Linux with kernel 5.1+. On other platforms,
//! it provides stub implementations that fall back to standard I/O.

#![allow(missing_docs)]

use super::error::{GpuError, GpuResult};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for io_uring operations.
#[derive(Debug, Clone)]
pub struct IoUringConfig {
    /// Queue depth (number of entries in submission/completion queues).
    pub queue_depth: u32,
    /// Enable kernel-side polling (SQPOLL) for reduced latency.
    pub sq_poll: bool,
    /// SQPOLL idle timeout in milliseconds (0 = never idle).
    pub sq_poll_idle_ms: u32,
    /// Enable direct I/O (bypass page cache).
    pub direct_io: bool,
    /// Buffer alignment for direct I/O (typically 512 or 4096).
    pub alignment: usize,
    /// Number of registered buffers (0 = no buffer registration).
    pub registered_buffers: usize,
    /// Size of each registered buffer.
    pub buffer_size: usize,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            queue_depth: 256,
            sq_poll: false,
            sq_poll_idle_ms: 1000,
            direct_io: false,
            alignment: 4096,
            registered_buffers: 0,
            buffer_size: 1 << 20, // 1MB
        }
    }
}

impl IoUringConfig {
    /// Create a new configuration builder.
    pub fn builder() -> IoUringConfigBuilder {
        IoUringConfigBuilder::default()
    }

    /// Create a high-performance configuration with SQPOLL and direct I/O.
    pub fn high_performance() -> Self {
        Self {
            queue_depth: 512,
            sq_poll: true,
            sq_poll_idle_ms: 2000,
            direct_io: true,
            alignment: 4096,
            registered_buffers: 32,
            buffer_size: 4 << 20, // 4MB
        }
    }
}

/// Builder for IoUringConfig.
#[derive(Debug, Default)]
pub struct IoUringConfigBuilder {
    config: IoUringConfig,
}

impl IoUringConfigBuilder {
    pub fn queue_depth(mut self, depth: u32) -> Self {
        self.config.queue_depth = depth;
        self
    }

    pub fn sq_poll(mut self, enabled: bool) -> Self {
        self.config.sq_poll = enabled;
        self
    }

    pub fn sq_poll_idle_ms(mut self, ms: u32) -> Self {
        self.config.sq_poll_idle_ms = ms;
        self
    }

    pub fn direct_io(mut self, enabled: bool) -> Self {
        self.config.direct_io = enabled;
        self
    }

    pub fn alignment(mut self, align: usize) -> Self {
        self.config.alignment = align;
        self
    }

    pub fn registered_buffers(mut self, count: usize, size: usize) -> Self {
        self.config.registered_buffers = count;
        self.config.buffer_size = size;
        self
    }

    pub fn build(self) -> IoUringConfig {
        self.config
    }
}

// ============================================================================
// Aligned Buffer
// ============================================================================

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

        // Round up capacity to alignment
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

        Ok(Self {
            ptr,
            len: 0,
            capacity: aligned_cap,
            alignment,
        })
    }

    /// Create a buffer with default 4KB alignment.
    pub fn with_capacity(capacity: usize) -> GpuResult<Self> {
        Self::new(capacity, 4096)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Set the length of valid data in the buffer.
    ///
    /// # Safety
    /// The caller must ensure that data up to `len` is initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.capacity);
        self.len = len;
    }

    /// Clear the buffer (set length to 0).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Check if the buffer is properly aligned for direct I/O.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr as usize) % self.alignment == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout =
                std::alloc::Layout::from_size_align(self.capacity, self.alignment).unwrap();
            unsafe {
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for AlignedBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

// ============================================================================
// Read Request / Completion
// ============================================================================

/// A pending read request.
#[derive(Debug, Clone)]
pub struct ReadRequest {
    /// File offset to read from.
    pub offset: u64,
    /// Number of bytes to read.
    pub length: usize,
    /// User-provided context/tag for this request.
    pub user_data: u64,
}

/// Result of a completed read operation.
#[derive(Debug)]
pub struct ReadCompletion {
    /// User-provided context/tag from the request.
    pub user_data: u64,
    /// Number of bytes actually read (may be less than requested).
    pub bytes_read: usize,
    /// Error code (0 = success, negative = errno).
    pub result: i32,
    /// Buffer containing the read data.
    pub buffer: AlignedBuffer,
}

impl ReadCompletion {
    /// Check if the read was successful.
    #[inline]
    pub fn is_success(&self) -> bool {
        self.result >= 0
    }

    /// Get the error if the read failed.
    pub fn error(&self) -> Option<std::io::Error> {
        if self.result < 0 {
            Some(std::io::Error::from_raw_os_error(-self.result))
        } else {
            None
        }
    }
}

// ============================================================================
// io_uring implementation (Linux only)
// ============================================================================

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod linux {
    use super::*;
    use io_uring::{opcode, types, IoUring, Probe};
    use std::os::unix::fs::OpenOptionsExt;

    /// Check if io_uring is available on this system.
    pub fn is_available() -> bool {
        // Try to create a minimal ring to check availability
        IoUring::builder().build(2).is_ok()
    }

    /// Check which io_uring operations are supported.
    pub fn probe_ops() -> Option<Probe> {
        let ring = IoUring::builder().build(2).ok()?;
        let mut probe = Probe::new();
        ring.submitter().register_probe(&mut probe).ok()?;
        Some(probe)
    }

    /// io_uring-based async file reader.
    pub struct IoUringReader {
        ring: IoUring,
        file: File,
        file_size: u64,
        config: IoUringConfig,
        pending: HashMap<u64, PendingRead>,
        next_id: AtomicU64,
        registered_buffers: Vec<AlignedBuffer>,
    }

    struct PendingRead {
        buffer: Option<AlignedBuffer>,
        buffer_index: Option<u16>,
        offset: u64,
        length: usize,
    }

    impl IoUringReader {
        /// Create a new io_uring reader for the given file.
        pub fn new(path: impl AsRef<Path>, config: IoUringConfig) -> GpuResult<Self> {
            // Open file with appropriate flags
            let mut open_options = std::fs::OpenOptions::new();
            open_options.read(true);

            if config.direct_io {
                // O_DIRECT for bypassing page cache
                open_options.custom_flags(libc::O_DIRECT);
            }

            let file = open_options.open(path.as_ref()).map_err(GpuError::IoError)?;
            let file_size = file.metadata().map_err(GpuError::IoError)?.len();

            // Build io_uring instance
            let mut builder = IoUring::builder();

            if config.sq_poll {
                builder.setup_sqpoll(config.sq_poll_idle_ms);
            }

            let ring = builder
                .build(config.queue_depth)
                .map_err(|e| GpuError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            let mut reader = Self {
                ring,
                file,
                file_size,
                config,
                pending: HashMap::new(),
                next_id: AtomicU64::new(1),
                registered_buffers: Vec::new(),
            };

            // Register buffers if configured
            if config.registered_buffers > 0 {
                reader.register_buffers()?;
            }

            Ok(reader)
        }

        /// Register fixed buffers with the kernel for zero-copy I/O.
        fn register_buffers(&mut self) -> GpuResult<()> {
            let mut buffers = Vec::with_capacity(self.config.registered_buffers);
            let mut iovecs = Vec::with_capacity(self.config.registered_buffers);

            for _ in 0..self.config.registered_buffers {
                let buffer = AlignedBuffer::new(self.config.buffer_size, self.config.alignment)?;
                iovecs.push(libc::iovec {
                    iov_base: buffer.as_ptr() as *mut _,
                    iov_len: buffer.capacity(),
                });
                buffers.push(buffer);
            }

            // Register with kernel
            self.ring
                .submitter()
                .register_buffers(&iovecs)
                .map_err(|e| GpuError::IoError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

            self.registered_buffers = buffers;
            Ok(())
        }

        /// Get the file size.
        #[inline]
        pub fn file_size(&self) -> u64 {
            self.file_size
        }

        /// Submit a single read request.
        pub fn submit_read(&mut self, request: ReadRequest) -> GpuResult<u64> {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);

            // Align offset and length for direct I/O
            let (aligned_offset, aligned_length) = if self.config.direct_io {
                let align = self.config.alignment as u64;
                let start = request.offset & !(align - 1);
                let end = (request.offset + request.length as u64 + align - 1) & !(align - 1);
                (start, (end - start) as usize)
            } else {
                (request.offset, request.length)
            };

            // Allocate buffer
            let buffer = AlignedBuffer::new(aligned_length, self.config.alignment)?;

            // Build read operation
            let read_op = opcode::Read::new(
                types::Fd(self.file.as_raw_fd()),
                buffer.as_ptr() as *mut u8,
                aligned_length as u32,
            )
            .offset(aligned_offset)
            .build()
            .user_data(id);

            // Submit to ring
            unsafe {
                self.ring
                    .submission()
                    .push(&read_op)
                    .map_err(|_| GpuError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "submission queue full",
                    )))?;
            }

            self.ring.submit().map_err(GpuError::IoError)?;

            // Track pending request
            self.pending.insert(
                id,
                PendingRead {
                    buffer: Some(buffer),
                    buffer_index: None,
                    offset: request.offset,
                    length: request.length,
                },
            );

            Ok(id)
        }

        /// Submit multiple read requests in a batch.
        pub fn submit_batch(&mut self, requests: &[ReadRequest]) -> GpuResult<Vec<u64>> {
            let mut ids = Vec::with_capacity(requests.len());

            for request in requests {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);

                let (aligned_offset, aligned_length) = if self.config.direct_io {
                    let align = self.config.alignment as u64;
                    let start = request.offset & !(align - 1);
                    let end = (request.offset + request.length as u64 + align - 1) & !(align - 1);
                    (start, (end - start) as usize)
                } else {
                    (request.offset, request.length)
                };

                let buffer = AlignedBuffer::new(aligned_length, self.config.alignment)?;

                let read_op = opcode::Read::new(
                    types::Fd(self.file.as_raw_fd()),
                    buffer.as_ptr() as *mut u8,
                    aligned_length as u32,
                )
                .offset(aligned_offset)
                .build()
                .user_data(id);

                unsafe {
                    self.ring
                        .submission()
                        .push(&read_op)
                        .map_err(|_| GpuError::IoError(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "submission queue full",
                        )))?;
                }

                self.pending.insert(
                    id,
                    PendingRead {
                        buffer: Some(buffer),
                        buffer_index: None,
                        offset: request.offset,
                        length: request.length,
                    },
                );

                ids.push(id);
            }

            // Submit all at once
            self.ring.submit().map_err(GpuError::IoError)?;

            Ok(ids)
        }

        /// Wait for and retrieve a specific completion.
        pub fn wait_for(&mut self, id: u64) -> GpuResult<ReadCompletion> {
            loop {
                // Check completion queue
                self.ring.submit_and_wait(1).map_err(GpuError::IoError)?;

                for cqe in self.ring.completion() {
                    let cqe_id = cqe.user_data();
                    let result = cqe.result();

                    if let Some(mut pending) = self.pending.remove(&cqe_id) {
                        let mut buffer = pending.buffer.take().unwrap();

                        if result >= 0 {
                            unsafe {
                                buffer.set_len(result as usize);
                            }
                        }

                        let completion = ReadCompletion {
                            user_data: cqe_id,
                            bytes_read: if result >= 0 { result as usize } else { 0 },
                            result,
                            buffer,
                        };

                        if cqe_id == id {
                            return Ok(completion);
                        }
                        // For other completions, we'd need to store them somewhere
                        // For now, just continue if it's not the one we're looking for
                    }
                }
            }
        }

        /// Poll for any available completions without blocking.
        pub fn poll_completions(&mut self) -> Vec<ReadCompletion> {
            let mut completions = Vec::new();

            // Non-blocking check
            let _ = self.ring.submit();

            for cqe in self.ring.completion() {
                let id = cqe.user_data();
                let result = cqe.result();

                if let Some(mut pending) = self.pending.remove(&id) {
                    let mut buffer = pending.buffer.take().unwrap();

                    if result >= 0 {
                        unsafe {
                            buffer.set_len(result as usize);
                        }
                    }

                    completions.push(ReadCompletion {
                        user_data: id,
                        bytes_read: if result >= 0 { result as usize } else { 0 },
                        result,
                        buffer,
                    });
                }
            }

            completions
        }

        /// Wait for all pending operations to complete.
        pub fn wait_all(&mut self) -> GpuResult<Vec<ReadCompletion>> {
            let mut completions = Vec::new();

            while !self.pending.is_empty() {
                self.ring.submit_and_wait(1).map_err(GpuError::IoError)?;

                for cqe in self.ring.completion() {
                    let id = cqe.user_data();
                    let result = cqe.result();

                    if let Some(mut pending) = self.pending.remove(&id) {
                        let mut buffer = pending.buffer.take().unwrap();

                        if result >= 0 {
                            unsafe {
                                buffer.set_len(result as usize);
                            }
                        }

                        completions.push(ReadCompletion {
                            user_data: id,
                            bytes_read: if result >= 0 { result as usize } else { 0 },
                            result,
                            buffer,
                        });
                    }
                }
            }

            Ok(completions)
        }

        /// Get the number of pending operations.
        #[inline]
        pub fn pending_count(&self) -> usize {
            self.pending.len()
        }
    }

    impl Drop for IoUringReader {
        fn drop(&mut self) {
            // Unregister buffers if registered
            if !self.registered_buffers.is_empty() {
                let _ = self.ring.submitter().unregister_buffers();
            }
        }
    }
}

// ============================================================================
// Fallback implementation (non-Linux or no io-uring feature)
// ============================================================================

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
mod fallback {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    /// Check if io_uring is available (always false on non-Linux).
    pub fn is_available() -> bool {
        false
    }

    /// Fallback reader using standard I/O.
    pub struct IoUringReader {
        file: File,
        file_size: u64,
        config: IoUringConfig,
        next_id: AtomicU64,
    }

    impl IoUringReader {
        /// Create a new reader (falls back to standard I/O).
        pub fn new(path: impl AsRef<Path>, config: IoUringConfig) -> GpuResult<Self> {
            let file = File::open(path.as_ref()).map_err(GpuError::IoError)?;
            let file_size = file.metadata().map_err(GpuError::IoError)?.len();

            Ok(Self {
                file,
                file_size,
                config,
                next_id: AtomicU64::new(1),
            })
        }

        #[inline]
        pub fn file_size(&self) -> u64 {
            self.file_size
        }

        /// Submit a read request (synchronous fallback).
        pub fn submit_read(&mut self, request: ReadRequest) -> GpuResult<u64> {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            Ok(id)
        }

        /// Submit multiple read requests.
        pub fn submit_batch(&mut self, requests: &[ReadRequest]) -> GpuResult<Vec<u64>> {
            let mut ids = Vec::with_capacity(requests.len());
            for _ in requests {
                ids.push(self.next_id.fetch_add(1, Ordering::Relaxed));
            }
            Ok(ids)
        }

        /// Perform a synchronous read (fallback).
        pub fn read_sync(&mut self, offset: u64, length: usize) -> GpuResult<AlignedBuffer> {
            let mut buffer = AlignedBuffer::new(length, self.config.alignment)?;

            self.file
                .seek(SeekFrom::Start(offset))
                .map_err(GpuError::IoError)?;

            let slice = unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr(), length) };
            let bytes_read = self.file.read(slice).map_err(GpuError::IoError)?;

            unsafe {
                buffer.set_len(bytes_read);
            }

            Ok(buffer)
        }

        /// Wait for a completion (synchronous fallback - reads immediately).
        pub fn wait_for(&mut self, _id: u64) -> GpuResult<ReadCompletion> {
            Err(GpuError::InvalidConfig {
                message: "io_uring not available, use read_sync instead".into(),
            })
        }

        /// Poll for completions (always empty in fallback mode).
        pub fn poll_completions(&mut self) -> Vec<ReadCompletion> {
            Vec::new()
        }

        /// Wait for all (no-op in fallback mode).
        pub fn wait_all(&mut self) -> GpuResult<Vec<ReadCompletion>> {
            Ok(Vec::new())
        }

        #[inline]
        pub fn pending_count(&self) -> usize {
            0
        }
    }
}

// ============================================================================
// Public API
// ============================================================================

#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub use linux::{is_available, IoUringReader};

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
pub use fallback::{is_available, IoUringReader};

// ============================================================================
// High-level tensor loading with io_uring
// ============================================================================

/// Read regions from a file using io_uring for optimal performance.
pub struct TensorFileReader {
    reader: IoUringReader,
    config: IoUringConfig,
}

impl TensorFileReader {
    /// Create a new tensor file reader.
    pub fn new(path: impl AsRef<Path>, config: IoUringConfig) -> GpuResult<Self> {
        let reader = IoUringReader::new(path, config.clone())?;
        Ok(Self { reader, config })
    }

    /// Create with default configuration.
    pub fn open(path: impl AsRef<Path>) -> GpuResult<Self> {
        Self::new(path, IoUringConfig::default())
    }

    /// Get the file size.
    #[inline]
    pub fn file_size(&self) -> u64 {
        self.reader.file_size()
    }

    /// Read a single region from the file.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn read_region(&mut self, offset: u64, length: usize) -> GpuResult<Vec<u8>> {
        let request = ReadRequest {
            offset,
            length,
            user_data: 0,
        };

        let id = self.reader.submit_read(request)?;
        let completion = self.reader.wait_for(id)?;

        if completion.is_success() {
            Ok(completion.buffer.as_slice().to_vec())
        } else {
            Err(GpuError::IoError(completion.error().unwrap()))
        }
    }

    /// Read a single region from the file (fallback).
    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    pub fn read_region(&mut self, offset: u64, length: usize) -> GpuResult<Vec<u8>> {
        let buffer = self.reader.read_sync(offset, length)?;
        Ok(buffer.as_slice().to_vec())
    }

    /// Read multiple regions in parallel.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn read_regions(&mut self, regions: &[(u64, usize)]) -> GpuResult<Vec<Vec<u8>>> {
        let requests: Vec<_> = regions
            .iter()
            .enumerate()
            .map(|(i, &(offset, length))| ReadRequest {
                offset,
                length,
                user_data: i as u64,
            })
            .collect();

        let _ids = self.reader.submit_batch(&requests)?;
        let completions = self.reader.wait_all()?;

        // Sort by user_data to maintain order
        let mut results: Vec<Option<Vec<u8>>> = vec![None; regions.len()];
        for completion in completions {
            if completion.is_success() {
                let idx = completion.user_data as usize;
                results[idx] = Some(completion.buffer.as_slice().to_vec());
            } else {
                return Err(GpuError::IoError(completion.error().unwrap()));
            }
        }

        results
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| GpuError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                "incomplete reads",
            )))
    }

    /// Read multiple regions (fallback).
    #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
    pub fn read_regions(&mut self, regions: &[(u64, usize)]) -> GpuResult<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(regions.len());
        for &(offset, length) in regions {
            let buffer = self.reader.read_sync(offset, length)?;
            results.push(buffer.as_slice().to_vec());
        }
        Ok(results)
    }

    /// Read tensor data regions for a safetensor file.
    ///
    /// This reads multiple tensor data regions efficiently using io_uring's
    /// batched submission capability.
    pub fn read_tensor_regions(
        &mut self,
        header_size: usize,
        tensor_offsets: &[(String, usize, usize)], // (name, start, end)
    ) -> GpuResult<HashMap<String, Vec<u8>>> {
        let regions: Vec<_> = tensor_offsets
            .iter()
            .map(|(_, start, end)| ((header_size + start) as u64, end - start))
            .collect();

        let data_buffers = self.read_regions(&regions)?;

        let mut result = HashMap::with_capacity(tensor_offsets.len());
        for ((name, _, _), data) in tensor_offsets.iter().zip(data_buffers) {
            result.insert(name.clone(), data);
        }

        Ok(result)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_file(size: usize) -> (tempfile::NamedTempFile, Vec<u8>) {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        file.write_all(&data).unwrap();
        file.flush().unwrap();
        (file, data)
    }

    #[test]
    fn test_aligned_buffer_new() {
        let buffer = AlignedBuffer::new(4096, 4096).unwrap();
        assert_eq!(buffer.capacity(), 4096);
        assert!(buffer.is_empty());
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_zero_size() {
        assert!(AlignedBuffer::new(0, 4096).is_err());
    }

    #[test]
    fn test_aligned_buffer_alignment() {
        let buffer = AlignedBuffer::new(1000, 512).unwrap();
        // Capacity should be rounded up to alignment
        assert!(buffer.capacity() >= 1000);
        assert_eq!(buffer.capacity() % 512, 0);
        assert!(buffer.is_aligned());
    }

    #[test]
    fn test_aligned_buffer_set_len() {
        let mut buffer = AlignedBuffer::new(4096, 4096).unwrap();
        assert_eq!(buffer.len(), 0);
        unsafe {
            buffer.set_len(100);
        }
        assert_eq!(buffer.len(), 100);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_config_builder() {
        let config = IoUringConfig::builder()
            .queue_depth(512)
            .sq_poll(true)
            .sq_poll_idle_ms(2000)
            .direct_io(true)
            .alignment(4096)
            .registered_buffers(16, 1 << 20)
            .build();

        assert_eq!(config.queue_depth, 512);
        assert!(config.sq_poll);
        assert_eq!(config.sq_poll_idle_ms, 2000);
        assert!(config.direct_io);
        assert_eq!(config.alignment, 4096);
        assert_eq!(config.registered_buffers, 16);
        assert_eq!(config.buffer_size, 1 << 20);
    }

    #[test]
    fn test_high_performance_config() {
        let config = IoUringConfig::high_performance();
        assert!(config.sq_poll);
        assert!(config.direct_io);
        assert!(config.queue_depth >= 256);
    }

    #[test]
    fn test_is_available() {
        // Just check it doesn't panic
        let _ = is_available();
    }

    #[test]
    fn test_tensor_file_reader() {
        let (file, data) = create_test_file(8192);
        let mut reader = TensorFileReader::open(file.path()).unwrap();

        assert_eq!(reader.file_size(), 8192);

        // Read a region
        let region = reader.read_region(0, 1024).unwrap();
        assert_eq!(region.len(), 1024);
        assert_eq!(&region[..], &data[..1024]);
    }

    #[test]
    fn test_tensor_file_reader_multiple_regions() {
        let (file, data) = create_test_file(16384);
        let mut reader = TensorFileReader::open(file.path()).unwrap();

        let regions = vec![(0, 1024), (4096, 2048), (8192, 512)];
        let results = reader.read_regions(&regions).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(&results[0][..], &data[0..1024]);
        assert_eq!(&results[1][..], &data[4096..6144]);
        assert_eq!(&results[2][..], &data[8192..8704]);
    }

    #[test]
    fn test_read_completion_success() {
        let buffer = AlignedBuffer::new(1024, 4096).unwrap();
        let completion = ReadCompletion {
            user_data: 42,
            bytes_read: 1024,
            result: 1024,
            buffer,
        };

        assert!(completion.is_success());
        assert!(completion.error().is_none());
    }

    #[test]
    fn test_read_completion_error() {
        let buffer = AlignedBuffer::new(1024, 4096).unwrap();
        let completion = ReadCompletion {
            user_data: 42,
            bytes_read: 0,
            result: -2, // ENOENT
            buffer,
        };

        assert!(!completion.is_success());
        assert!(completion.error().is_some());
    }
}
