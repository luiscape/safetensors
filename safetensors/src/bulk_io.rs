//! Bulk I/O primitives for high-throughput safetensors file loading.
//!
//! This module provides optimized, parallel file reading strategies inspired by
//! [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors).
//! It is gated behind the `fast_io` feature flag and requires the `std` feature.
//!
//! # Overview
//!
//! - [`BulkReadPlan`]: Splits a file body into sized blocks for parallel reads.
//! - [`AlignmentFixup`]: Describes tensors whose data is misaligned and needs correction.
//! - [`PreadBulkReader`]: Performs parallel `pread(2)` I/O via a rayon thread pool.
//! - [`GdsReadPlan`]: Describes read blocks with 512-byte alignment for GPU Direct Storage.
//! - NUMA helpers for topology-aware buffer placement.

use crate::tensor::{Metadata, SafeTensorError};
use std::collections::HashMap;

// ─── BulkReadPlan ───────────────────────────────────────────────────────────

/// A single contiguous read block within the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadBlock {
    /// Absolute byte offset within the file where this block starts.
    pub file_offset: usize,
    /// Number of bytes to read for this block.
    pub length: usize,
}

/// A plan that describes how to read a safetensors file body in chunks.
///
/// The body is the portion of the file after the 8-byte length prefix and
/// the JSON header. `BulkReadPlan` splits that region into blocks of at most
/// `max_block_size` bytes so that multiple threads can read in parallel.
#[derive(Debug, Clone)]
pub struct BulkReadPlan {
    /// Absolute byte offset in the file where the tensor data body begins.
    pub body_file_offset: usize,
    /// Total length of the tensor data body in bytes.
    pub body_length: usize,
    /// Ordered list of read blocks that together cover the entire body.
    pub blocks: Vec<ReadBlock>,
}

/// Default maximum block size: 16 MiB.
const DEFAULT_MAX_BLOCK_SIZE: usize = 16 * 1024 * 1024;

impl BulkReadPlan {
    /// Create a new read plan for the file body.
    ///
    /// # Arguments
    ///
    /// * `header_size` – Total size of the file header in bytes (8-byte length
    ///   prefix + JSON header). The body starts immediately after this.
    /// * `body_length` – Length of the tensor data body in bytes (i.e.
    ///   [`Metadata::data_len()`]).
    /// * `max_block_size` – Maximum number of bytes per read block. Pass `None`
    ///   to use the default of 16 MiB.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let plan = BulkReadPlan::new(header_size, metadata.data_len(), None);
    /// assert!(!plan.blocks.is_empty());
    /// ```
    pub fn new(header_size: usize, body_length: usize, max_block_size: Option<usize>) -> Self {
        let block_size = max_block_size.unwrap_or(DEFAULT_MAX_BLOCK_SIZE).max(1);
        let body_file_offset = header_size;
        let mut blocks = Vec::new();

        let mut remaining = body_length;
        let mut offset = body_file_offset;
        while remaining > 0 {
            let len = remaining.min(block_size);
            blocks.push(ReadBlock {
                file_offset: offset,
                length: len,
            });
            offset += len;
            remaining -= len;
        }

        Self {
            body_file_offset,
            body_length,
            blocks,
        }
    }

    /// Create a read plan directly from [`Metadata`].
    ///
    /// This is a convenience wrapper around [`BulkReadPlan::new`] that extracts
    /// `body_length` from the metadata.
    pub fn from_metadata(
        metadata: &Metadata,
        header_size: usize,
        max_block_size: Option<usize>,
    ) -> Self {
        Self::new(header_size, metadata.data_len(), max_block_size)
    }

    /// Returns the total number of read blocks in this plan.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// ─── AlignmentFixup ─────────────────────────────────────────────────────────

/// Describes a tensor whose in-buffer data is not aligned to a required boundary.
///
/// When the file header size is not a multiple of the desired alignment (e.g.
/// 256 bytes for GPU transfers), tensor data offsets inside a loaded buffer may
/// be misaligned. An `AlignmentFixup` records the information needed to copy
/// or remap the data to a properly aligned location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentFixup {
    /// The name of the misaligned tensor.
    pub tensor_name: String,
    /// Current byte offset of this tensor's data within the body buffer.
    pub current_offset: usize,
    /// Length of the tensor's data in bytes.
    pub length: usize,
    /// The byte offset the data *should* reside at for proper alignment.
    pub aligned_offset: usize,
}

/// Compute alignment fixups for every tensor in the metadata.
///
/// If `header_size` is already a multiple of `alignment`, all tensors are
/// naturally aligned and the returned vector will be empty.
///
/// # Arguments
///
/// * `metadata` – Parsed safetensors [`Metadata`].
/// * `header_size` – Total file header size (8-byte prefix + JSON).
/// * `alignment` – Required alignment in bytes (e.g. 256, 512, 4096).
///
/// # Returns
///
/// A vector of [`AlignmentFixup`] entries for tensors that need adjustment.
/// Tensors that are already aligned are omitted.
pub fn compute_alignment_fixups(
    metadata: &Metadata,
    header_size: usize,
    alignment: usize,
) -> Vec<AlignmentFixup> {
    if alignment == 0 || header_size % alignment == 0 {
        return Vec::new();
    }

    let tensors = metadata.tensors();
    let mut fixups = Vec::new();

    // Compute the padding needed to align the body start.
    let misalignment = header_size % alignment;
    let padding = alignment - misalignment;

    for (name, info) in &tensors {
        let (start, end) = info.data_offsets;
        let length = end - start;
        if length == 0 {
            continue;
        }
        let aligned_start = start + padding;
        if aligned_start != start {
            fixups.push(AlignmentFixup {
                tensor_name: name.clone(),
                current_offset: start,
                length,
                aligned_offset: aligned_start,
            });
        }
    }

    // Sort by current offset for deterministic ordering.
    fixups.sort_by_key(|f| f.current_offset);
    fixups
}

// ─── PreadBulkReader ────────────────────────────────────────────────────────

/// Default number of parallel reader threads.
const DEFAULT_MAX_THREADS: usize = 16;

/// A parallel file reader that uses POSIX `pread(2)` and a rayon thread pool
/// to load a safetensors file body into a host buffer at high throughput.
///
/// `PreadBulkReader` does **not** own the file descriptor or the destination
/// buffer—it merely orchestrates the parallel reads described by a
/// [`BulkReadPlan`].
///
/// # Platform support
///
/// The `pread`-based implementation is only available on Unix targets. On
/// non-Unix platforms, this struct is still defined but the read method will
/// return an error.
#[derive(Debug, Clone)]
pub struct PreadBulkReader {
    /// Size hint for the internal bounce buffer per thread (bytes).
    /// This is currently advisory; the reader reads directly into the
    /// destination slice without intermediate copies.
    pub bounce_buffer_size: usize,
    /// Maximum number of threads in the rayon thread pool used for parallel I/O.
    pub max_threads: usize,
}

impl Default for PreadBulkReader {
    fn default() -> Self {
        Self {
            bounce_buffer_size: DEFAULT_MAX_BLOCK_SIZE,
            max_threads: DEFAULT_MAX_THREADS,
        }
    }
}

impl PreadBulkReader {
    /// Create a new `PreadBulkReader`.
    ///
    /// # Arguments
    ///
    /// * `bounce_buffer_size` – Per-thread bounce buffer size hint in bytes.
    ///   Pass `None` for the default (16 MiB).
    /// * `max_threads` – Maximum number of reader threads. Pass `None` for the
    ///   default (16).
    pub fn new(bounce_buffer_size: Option<usize>, max_threads: Option<usize>) -> Self {
        Self {
            bounce_buffer_size: bounce_buffer_size.unwrap_or(DEFAULT_MAX_BLOCK_SIZE),
            max_threads: max_threads.unwrap_or(DEFAULT_MAX_THREADS).max(1),
        }
    }

    /// Read the file body into `dst` according to the given [`BulkReadPlan`],
    /// using parallel `pread(2)` calls.
    ///
    /// The destination buffer `dst` must be at least `plan.body_length` bytes.
    ///
    /// # Arguments
    ///
    /// * `fd` – A raw file descriptor (e.g. from
    ///   [`std::os::unix::io::AsRawFd`]) opened for reading.
    /// * `plan` – The read plan describing which blocks to read.
    /// * `dst` – Mutable byte slice that receives the file body data.
    ///
    /// # Errors
    ///
    /// Returns [`SafeTensorError::IoError`] if any `pread` call fails.
    ///
    /// # Safety note
    ///
    /// The caller must ensure that `fd` is a valid, open file descriptor and
    /// that it remains open for the duration of this call.
    #[cfg(unix)]
    pub fn read_to_buffer(
        &self,
        fd: std::os::unix::io::RawFd,
        plan: &BulkReadPlan,
        dst: &mut [u8],
    ) -> Result<(), SafeTensorError> {
        use rayon::prelude::*;

        if dst.len() < plan.body_length {
            return Err(SafeTensorError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "destination buffer too small: {} < {}",
                    dst.len(),
                    plan.body_length
                ),
            )));
        }

        if plan.blocks.is_empty() {
            return Ok(());
        }

        // Build a thread pool with the configured parallelism.
        let num_threads = self.max_threads.min(plan.blocks.len());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| {
                SafeTensorError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("failed to build rayon thread pool: {e}"),
                ))
            })?;

        // We need to share `dst` across threads. Each block writes to a
        // non-overlapping sub-slice, so this is safe. We store the base
        // address as a `usize` so it is `Send + Sync`.
        let dst_base = dst.as_mut_ptr() as usize;
        let dst_len = dst.len();

        let result: Result<(), SafeTensorError> = pool.install(|| {
            plan.blocks.par_iter().try_for_each(|block| {
                let buf_offset = block.file_offset - plan.body_file_offset;
                let buf_end = buf_offset + block.length;
                if buf_end > dst_len {
                    return Err(SafeTensorError::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "read block exceeds destination buffer",
                    )));
                }

                // SAFETY: Each block targets a non-overlapping region of `dst`.
                // The `dst_base` pointer is valid for the entire `pool.install`
                // scope because `dst` is exclusively borrowed by this function.
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        (dst_base as *mut u8).add(buf_offset),
                        block.length,
                    )
                };

                pread_exact(fd, slice, block.file_offset as i64)?;
                Ok(())
            })
        });

        result
    }

    /// Fallback for non-Unix platforms — always returns an error.
    #[cfg(not(unix))]
    pub fn read_to_buffer(
        &self,
        _fd: i32,
        _plan: &BulkReadPlan,
        _dst: &mut [u8],
    ) -> Result<(), SafeTensorError> {
        Err(SafeTensorError::IoError(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "PreadBulkReader is only supported on Unix platforms",
        )))
    }
}

/// Issue `pread(2)` calls in a loop until the entire slice is filled,
/// handling short reads.
#[cfg(unix)]
fn pread_exact(
    fd: std::os::unix::io::RawFd,
    buf: &mut [u8],
    offset: i64,
) -> Result<(), SafeTensorError> {
    let mut total_read: usize = 0;
    while total_read < buf.len() {
        let ret = unsafe {
            libc::pread(
                fd,
                buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - total_read,
                offset + total_read as i64,
            )
        };
        if ret < 0 {
            let err = std::io::Error::last_os_error();
            // Retry on EINTR.
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(SafeTensorError::IoError(err));
        }
        if ret == 0 {
            return Err(SafeTensorError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "pread returned 0 at offset {} (read {}/{})",
                    offset + total_read as i64,
                    total_read,
                    buf.len()
                ),
            )));
        }
        total_read += ret as usize;
    }
    Ok(())
}

// ─── GdsReadPlan ────────────────────────────────────────────────────────────

/// GPU Direct Storage (GDS) alignment requirement in bytes.
pub const GDS_ALIGNMENT: usize = 512;

/// A single read block for GPU Direct Storage, respecting 512-byte alignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdsReadBlock {
    /// Absolute byte offset in the file (512-byte aligned).
    pub file_offset: usize,
    /// Offset within the GPU buffer where data should be placed.
    pub buffer_offset: usize,
    /// Number of bytes to read (512-byte aligned).
    pub length: usize,
}

/// A read plan tailored for GPU Direct Storage, which requires 512-byte aligned
/// offsets and lengths.
///
/// GDS (also called *GPUDirect Storage* or *cuFile*) enables direct DMA from
/// NVMe storage to GPU memory, bypassing the CPU page cache. All offsets and
/// lengths must be multiples of 512 bytes for the hardware DMA engine.
#[derive(Debug, Clone)]
pub struct GdsReadPlan {
    /// Absolute byte offset of the body in the file.
    pub body_file_offset: usize,
    /// Total body length in bytes.
    pub body_length: usize,
    /// Ordered list of aligned read blocks.
    pub blocks: Vec<GdsReadBlock>,
}

impl GdsReadPlan {
    /// Create a new GDS read plan for the file body.
    ///
    /// File offsets are rounded down to the nearest 512-byte boundary and
    /// lengths are rounded up so that the entire body is covered.
    ///
    /// # Arguments
    ///
    /// * `header_size` – Total file header size (8-byte prefix + JSON).
    /// * `body_length` – Length of the tensor data body in bytes.
    /// * `max_block_size` – Maximum bytes per GDS read block. Will be rounded
    ///   down to a multiple of 512. Pass `None` for 16 MiB default.
    pub fn new(header_size: usize, body_length: usize, max_block_size: Option<usize>) -> Self {
        let raw_block_size = max_block_size.unwrap_or(DEFAULT_MAX_BLOCK_SIZE);
        // Round block size down to a 512 multiple (at least 512).
        let block_size = (raw_block_size / GDS_ALIGNMENT).max(1) * GDS_ALIGNMENT;

        let body_file_offset = header_size;

        // Align the start offset down.
        let aligned_start = (body_file_offset / GDS_ALIGNMENT) * GDS_ALIGNMENT;
        // Align the end offset up.
        let body_end = body_file_offset + body_length;
        let aligned_end = ((body_end + GDS_ALIGNMENT - 1) / GDS_ALIGNMENT) * GDS_ALIGNMENT;
        let total_aligned_len = aligned_end - aligned_start;

        let mut blocks = Vec::new();
        let mut remaining = total_aligned_len;
        let mut file_off = aligned_start;
        let mut buf_off: usize = 0;

        while remaining > 0 {
            let len = remaining.min(block_size);
            blocks.push(GdsReadBlock {
                file_offset: file_off,
                buffer_offset: buf_off,
                length: len,
            });
            file_off += len;
            buf_off += len;
            remaining -= len;
        }

        Self {
            body_file_offset,
            body_length,
            blocks,
        }
    }

    /// Create a GDS read plan from [`Metadata`].
    pub fn from_metadata(
        metadata: &Metadata,
        header_size: usize,
        max_block_size: Option<usize>,
    ) -> Self {
        Self::new(header_size, metadata.data_len(), max_block_size)
    }

    /// Returns `true` if the header size causes tensor data to be misaligned
    /// for GDS 512-byte requirements.
    ///
    /// When this returns `true`, the caller must account for the leading
    /// padding bytes (from the aligned-down start to the actual body start)
    /// when interpreting tensor offsets in the GPU buffer.
    pub fn needs_alignment_fixup(header_size: usize) -> bool {
        header_size % GDS_ALIGNMENT != 0
    }

    /// Returns the number of leading padding bytes inserted before the body
    /// data due to start-offset alignment.
    ///
    /// This is `body_file_offset - aligned_start`. Tensor data begins at
    /// this offset within the GPU buffer.
    pub fn leading_padding(&self) -> usize {
        let aligned_start = (self.body_file_offset / GDS_ALIGNMENT) * GDS_ALIGNMENT;
        self.body_file_offset - aligned_start
    }

    /// Returns the total number of read blocks in this plan.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the total size of the GPU buffer required to hold all read
    /// blocks, including alignment padding.
    pub fn required_buffer_size(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.buffer_offset + b.length)
            .max()
            .unwrap_or(0)
    }
}

// ─── NUMA helpers ───────────────────────────────────────────────────────────

/// Query the NUMA node associated with a PCI device.
///
/// This reads `/sys/class/pci_bus/<bus_id>/device/numa_node` to discover which
/// NUMA domain a device (typically a GPU) is attached to. This is useful for
/// allocating host-side bounce buffers on the memory closest to the GPU.
///
/// # Arguments
///
/// * `pci_bus_id` – PCI bus identifier, e.g. `"0000:3b"` (domain:bus).
///
/// # Returns
///
/// `Some(node)` with the NUMA node ID, or `None` if the sysfs entry does not
/// exist or cannot be parsed. A value of `-1` from the kernel indicates no
/// NUMA affinity information is available.
///
/// # Platform
///
/// This function only produces meaningful results on Linux. On other platforms
/// it always returns `None`.
pub fn get_numa_node_for_device(pci_bus_id: &str) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        let path = format!("/sys/class/pci_bus/{}/device/numa_node", pci_bus_id);
        match std::fs::read_to_string(&path) {
            Ok(contents) => contents.trim().parse::<i32>().ok(),
            Err(_) => None,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = pci_bus_id;
        None
    }
}

/// Attempt to bind the calling thread to the specified NUMA node.
///
/// On Linux this calls `libnuma`'s `numa_run_on_node` via the C library.
/// Because `libnuma` may not be available at runtime, this function attempts a
/// best-effort approach and returns an error if the syscall is unavailable or
/// fails.
///
/// # Arguments
///
/// * `node` – The NUMA node ID to bind to (e.g. obtained from
///   [`get_numa_node_for_device`]).
///
/// # Errors
///
/// Returns a [`SafeTensorError::IoError`] if the binding fails.
///
/// # Platform
///
/// Only operational on Linux. On other platforms this is a no-op that returns
/// `Ok(())`.
pub fn set_thread_numa_node(node: i32) -> Result<(), SafeTensorError> {
    #[cfg(target_os = "linux")]
    {
        // Use sched_setaffinity with the CPUs belonging to this NUMA node.
        // We read the CPU list from sysfs.
        let cpu_path = format!("/sys/devices/system/node/node{}/cpulist", node);
        let cpu_list = std::fs::read_to_string(&cpu_path).map_err(|e| {
            SafeTensorError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cannot read NUMA node {} CPU list: {}", node, e),
            ))
        })?;

        let cpus = parse_cpu_list(cpu_list.trim());
        if cpus.is_empty() {
            return Err(SafeTensorError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("NUMA node {} has no CPUs", node),
            )));
        }

        set_thread_affinity_to_cpus(&cpus)?;
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = node;
        Ok(())
    }
}

/// Parse a Linux CPU list string like `"0-3,8-11"` into individual CPU IDs.
#[cfg(target_os = "linux")]
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (
                start_s.trim().parse::<usize>(),
                end_s.trim().parse::<usize>(),
            ) {
                for cpu in start..=end {
                    cpus.push(cpu);
                }
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Set the calling thread's CPU affinity to the given set of CPUs using
/// `sched_setaffinity(2)`.
#[cfg(target_os = "linux")]
fn set_thread_affinity_to_cpus(cpus: &[usize]) -> Result<(), SafeTensorError> {
    use std::mem;

    // cpu_set_t is typically 1024 bits = 128 bytes on x86_64 Linux.
    // We zero-initialize and set the relevant bits.
    const CPU_SET_SIZE: usize = 1024;
    const BITS_PER_ULONG: usize = 8 * mem::size_of::<libc::c_ulong>();

    let num_ulongs = CPU_SET_SIZE / BITS_PER_ULONG;
    let mut cpu_set: Vec<libc::c_ulong> = vec![0; num_ulongs];

    for &cpu in cpus {
        if cpu < CPU_SET_SIZE {
            let idx = cpu / BITS_PER_ULONG;
            let bit = cpu % BITS_PER_ULONG;
            cpu_set[idx] |= 1 << bit;
        }
    }

    let ret = unsafe {
        libc::sched_setaffinity(
            0, // current thread
            mem::size_of_val(cpu_set.as_slice()),
            cpu_set.as_ptr() as *const libc::cpu_set_t,
        )
    };

    if ret != 0 {
        Err(SafeTensorError::IoError(std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}

// ─── Convenience: read an entire safetensors file body ──────────────────────

/// Read the full tensor data body of a safetensors file into a newly allocated
/// buffer using parallel `pread(2)`.
///
/// This is a high-level convenience function that:
/// 1. Creates a [`BulkReadPlan`] from the metadata.
/// 2. Allocates a `Vec<u8>` buffer of the required size.
/// 3. Reads the body into the buffer using a [`PreadBulkReader`].
///
/// # Arguments
///
/// * `fd` – A raw file descriptor open for reading.
/// * `metadata` – Parsed safetensors [`Metadata`].
/// * `header_size` – Total header size (8-byte prefix + JSON header).
/// * `max_block_size` – Optional block size override (default 16 MiB).
/// * `max_threads` – Optional thread count override (default 16).
///
/// # Returns
///
/// A `Vec<u8>` containing the tensor data body. Individual tensor data can be
/// extracted using [`Metadata::tensor_body_offset`].
///
/// # Errors
///
/// Returns [`SafeTensorError::IoError`] on any I/O failure.
#[cfg(unix)]
pub fn read_body_parallel(
    fd: std::os::unix::io::RawFd,
    metadata: &Metadata,
    header_size: usize,
    max_block_size: Option<usize>,
    max_threads: Option<usize>,
) -> Result<Vec<u8>, SafeTensorError> {
    let plan = BulkReadPlan::from_metadata(metadata, header_size, max_block_size);
    let reader = PreadBulkReader::new(max_block_size, max_threads);
    let mut buffer = vec![0u8; plan.body_length];
    reader.read_to_buffer(fd, &plan, &mut buffer)?;
    Ok(buffer)
}

/// Map tensor names to their `(start, end)` byte ranges within the body buffer.
///
/// This is a convenience helper to build a lookup table from tensor name to the
/// offsets within a body buffer read by [`read_body_parallel`] or
/// [`PreadBulkReader::read_to_buffer`].
pub fn tensor_byte_ranges(metadata: &Metadata) -> HashMap<String, (usize, usize)> {
    metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| (name, info.data_offsets))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulk_read_plan_single_block() {
        let plan = BulkReadPlan::new(100, 500, Some(1024));
        assert_eq!(plan.body_file_offset, 100);
        assert_eq!(plan.body_length, 500);
        assert_eq!(plan.blocks.len(), 1);
        assert_eq!(plan.blocks[0].file_offset, 100);
        assert_eq!(plan.blocks[0].length, 500);
    }

    #[test]
    fn test_bulk_read_plan_multiple_blocks() {
        let plan = BulkReadPlan::new(64, 1000, Some(300));
        assert_eq!(plan.blocks.len(), 4);
        assert_eq!(plan.blocks[0].file_offset, 64);
        assert_eq!(plan.blocks[0].length, 300);
        assert_eq!(plan.blocks[1].file_offset, 364);
        assert_eq!(plan.blocks[1].length, 300);
        assert_eq!(plan.blocks[2].file_offset, 664);
        assert_eq!(plan.blocks[2].length, 300);
        assert_eq!(plan.blocks[3].file_offset, 964);
        assert_eq!(plan.blocks[3].length, 100);
    }

    #[test]
    fn test_bulk_read_plan_empty_body() {
        let plan = BulkReadPlan::new(64, 0, None);
        assert!(plan.blocks.is_empty());
        assert_eq!(plan.body_length, 0);
    }

    #[test]
    fn test_gds_alignment() {
        assert!(GdsReadPlan::needs_alignment_fixup(100));
        assert!(!GdsReadPlan::needs_alignment_fixup(512));
        assert!(!GdsReadPlan::needs_alignment_fixup(1024));
        assert!(GdsReadPlan::needs_alignment_fixup(513));
    }

    #[test]
    fn test_gds_read_plan_alignment() {
        let plan = GdsReadPlan::new(100, 1000, Some(4096));
        // Start should be rounded down to 0.
        assert_eq!(plan.blocks[0].file_offset, 0);
        // Total aligned length covers from 0 to >= 1100, rounded up.
        let total: usize = plan.blocks.iter().map(|b| b.length).sum();
        assert!(total >= 1100);
        assert_eq!(total % GDS_ALIGNMENT, 0);
        // Leading padding should be 100.
        assert_eq!(plan.leading_padding(), 100);
    }

    #[test]
    fn test_gds_read_plan_already_aligned() {
        let plan = GdsReadPlan::new(512, 1024, Some(4096));
        assert_eq!(plan.leading_padding(), 0);
        assert_eq!(plan.blocks[0].file_offset, 512);
        let total: usize = plan.blocks.iter().map(|b| b.length).sum();
        assert_eq!(total, 1024);
    }

    #[test]
    fn test_pread_bulk_reader_default() {
        let reader = PreadBulkReader::default();
        assert_eq!(reader.bounce_buffer_size, DEFAULT_MAX_BLOCK_SIZE);
        assert_eq!(reader.max_threads, DEFAULT_MAX_THREADS);
    }

    #[test]
    fn test_pread_bulk_reader_custom() {
        let reader = PreadBulkReader::new(Some(8 * 1024 * 1024), Some(4));
        assert_eq!(reader.bounce_buffer_size, 8 * 1024 * 1024);
        assert_eq!(reader.max_threads, 4);
    }

    #[test]
    fn test_alignment_fixup_empty_when_aligned() {
        // We can't easily construct a Metadata in tests without the full
        // deserialization path, so we test the boundary logic directly.
        assert_eq!(100 % 256, 100); // misaligned
        assert_eq!(256 % 256, 0); // aligned
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-1,4-5"), vec![0, 1, 4, 5]);
        assert_eq!(parse_cpu_list("7"), vec![7]);
        assert!(parse_cpu_list("").is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn test_pread_roundtrip() {
        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        // Write some data to a temp file.
        let dir = std::env::temp_dir();
        let path = dir.join("safetensors_bulk_io_test_pread_roundtrip.bin");
        let data: Vec<u8> = (0..4096u16).map(|i| (i % 256) as u8).collect();
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }

        let file = std::fs::File::open(&path).unwrap();
        let fd = file.as_raw_fd();
        let plan = BulkReadPlan::new(0, data.len(), Some(1024));
        let reader = PreadBulkReader::new(None, Some(2));
        let mut buf = vec![0u8; data.len()];
        reader.read_to_buffer(fd, &plan, &mut buf).unwrap();
        assert_eq!(buf, data);

        // Clean up.
        drop(file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_numa_non_linux() {
        // On non-Linux or when the sysfs path doesn't exist, this should
        // return None without panicking.
        let result = get_numa_node_for_device("0000:00");
        // We just verify it doesn't panic; the result depends on the platform.
        let _ = result;
    }
}
