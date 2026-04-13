//! Parallel bulk file reader for fast safetensors loading.
//!
//! Reads the body of safetensors files (skipping the header) using parallel
//! `pread(2)` operations with per-thread bounce buffers, then copies the data
//! into device memory via [`crate::device_buffer::copy_host_to_device_buffer`].
//!
//! The design mirrors the fastsafetensors approach:
//! 1. Parse the header to learn tensor offsets/sizes.
//! 2. Allocate a single large device buffer.
//! 3. Bulk-read the file body into host bounce buffers using parallel I/O threads.
//! 4. Copy each chunk from host to device.
//! 5. Instantiate tensor views from the device buffer (no per-tensor copies).
//!
//! GDS (GPU Direct Storage) support is available behind the `gds` feature flag.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io;

use crate::device_buffer::{self, DeviceBuffer};

// ---- pread portability shim ------------------------------------------------

/// Reads `buf.len()` bytes from `file` at the given absolute `offset` without
/// altering the file's seek position.
///
/// This is a thin wrapper around POSIX `pread(2)` on Unix and falls back to
/// `seek` + `read` on other platforms (holding an internal mutex so the
/// seek+read pair is atomic with respect to other calls through this function).
#[cfg(unix)]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let mut total_read: usize = 0;
    while total_read < buf.len() {
        let ret = unsafe {
            libc::pread(
                fd,
                buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - total_read,
                (offset + total_read as u64) as libc::off_t,
            )
        };
        if ret < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if ret == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "pread: unexpected EOF after {total_read} of {} bytes at offset {offset}",
                    buf.len()
                ),
            ));
        }
        total_read += ret as usize;
    }
    Ok(())
}

#[cfg(not(unix))]
fn pread_exact(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::Mutex;

    // A global mutex ensures the seek+read pair is atomic.  This is the
    // non-Unix fallback and is not expected to be performance-critical.
    static LOCK: Mutex<()> = Mutex::new(());

    let _guard = LOCK
        .lock()
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "pread fallback mutex poisoned"))?;

    // We need a mutable reference for seek/read, but we only have &File.
    // Clone the handle so each call operates on its own cursor.
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(offset))?;
    f.read_exact(buf)?;
    Ok(())
}

// ---- BulkFileReader --------------------------------------------------------

/// Default bounce-buffer size per thread: 4 MiB.
const DEFAULT_BOUNCE_BUFFER_SIZE: usize = 4 * 1024 * 1024;

/// Default number of I/O threads (0 = use rayon default, which is num CPUs).
const DEFAULT_MAX_THREADS: usize = 0;

/// Parallel file reader that bulk-reads safetensors bodies using `pread` and
/// bounce buffers.
///
/// The reader maintains a dedicated [`rayon::ThreadPool`] so that I/O
/// parallelism does not interfere with compute work on the global pool.
pub struct BulkFileReader {
    /// Maximum number of I/O threads (0 means rayon's default).
    #[allow(dead_code)]
    max_threads: usize,
    /// Bounce buffer size per thread in bytes.
    bounce_buffer_size: usize,
    /// Dedicated rayon thread pool for I/O work.
    pool: rayon::ThreadPool,
}

impl BulkFileReader {
    /// Creates a new `BulkFileReader`.
    ///
    /// # Arguments
    ///
    /// * `max_threads`          — Number of I/O threads.  Pass `0` to use one
    ///                            thread per logical CPU (rayon default).
    /// * `bounce_buffer_size_kb` — Per-thread bounce buffer size **in
    ///                             kilobytes**.  Pass `0` to use the default
    ///                             (4 MiB).
    ///
    /// # Panics
    ///
    /// Panics if the rayon thread pool cannot be built (should never happen
    /// under normal circumstances).
    pub fn new(max_threads: usize, bounce_buffer_size_kb: usize) -> Self {
        let threads = if max_threads == 0 {
            DEFAULT_MAX_THREADS
        } else {
            max_threads
        };

        let bounce = if bounce_buffer_size_kb == 0 {
            DEFAULT_BOUNCE_BUFFER_SIZE
        } else {
            bounce_buffer_size_kb * 1024
        };

        let mut builder = rayon::ThreadPoolBuilder::new();
        if threads > 0 {
            builder = builder.num_threads(threads);
        }
        let pool = builder
            .thread_name(|idx| format!("st-io-{idx}"))
            .build()
            .expect("BulkFileReader: failed to build rayon thread pool");

        Self {
            max_threads: if threads == 0 {
                rayon::current_num_threads()
            } else {
                threads
            },
            bounce_buffer_size: bounce,
            pool,
        }
    }

    /// Returns the configured maximum number of I/O threads.
    #[allow(dead_code)]
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// Returns the per-thread bounce buffer size in bytes.
    #[allow(dead_code)]
    pub fn bounce_buffer_size(&self) -> usize {
        self.bounce_buffer_size
    }

    /// Reads the file body (everything after the header) into a host `Vec<u8>`.
    ///
    /// The header is **skipped** — the returned bytes start at the first tensor
    /// data byte.  The caller should compute `header_size` as
    /// `8 + json_header_length` (the 8-byte little-endian length prefix plus
    /// the JSON header itself).
    ///
    /// # Arguments
    ///
    /// * `path`        — Path to the safetensors file.
    /// * `header_size` — Total header size in bytes (8 + JSON header length).
    /// * `body_size`   — Expected body size in bytes (file size − header size).
    ///
    /// # Errors
    ///
    /// Returns an error string if the file cannot be opened or any `pread`
    /// call fails.
    pub fn read_file_body_to_host(
        &self,
        path: &str,
        header_size: usize,
        body_size: usize,
    ) -> Result<Vec<u8>, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open '{path}': {e}"))?;

        // Pre-allocate the output buffer.
        let mut output = vec![0u8; body_size];

        // Partition into chunks for parallel reads.
        let chunk_size = self.bounce_buffer_size;
        let chunks: Vec<(usize, usize)> = ChunkIter::new(body_size, chunk_size).collect();

        // Build a vec of disjoint mutable slices — one per chunk.
        // Each slice covers exactly [chunk_offset .. chunk_offset + chunk_len].
        let mut slices: Vec<&mut [u8]> = Vec::with_capacity(chunks.len());
        {
            let mut remainder: &mut [u8] = &mut output;
            for &(_chunk_offset, chunk_len) in &chunks {
                let (head, tail) = remainder.split_at_mut(chunk_len);
                slices.push(head);
                remainder = tail;
            }
        }

        // Zip slices with their file offsets and read in parallel.
        // Each thread reads into its own bounce buffer and then copies to the
        // unique mutable slice, so there is no data race.
        let indexed: Vec<(u64, &mut [u8])> = chunks
            .iter()
            .zip(slices)
            .map(|(&(chunk_offset, _), slice)| (header_size as u64 + chunk_offset as u64, slice))
            .collect();

        let result: Result<(), String> = self.pool.install(|| {
            indexed.into_par_iter().try_for_each(|(file_offset, dest)| {
                let chunk_len = dest.len();
                let mut bounce = vec![0u8; chunk_len];
                pread_exact(&file, &mut bounce, file_offset).map_err(|e| {
                    format!("pread failed at file_offset={file_offset}, len={chunk_len}: {e}")
                })?;
                dest.copy_from_slice(&bounce);
                Ok(())
            })
        });

        result?;
        Ok(output)
    }

    /// Reads a safetensors file body directly into a device buffer.
    ///
    /// For **CPU** device buffers the data is read directly into the buffer via
    /// `pread`.  For **GPU** (or other non-CPU) devices, each I/O thread reads
    /// into a host-side bounce buffer and then copies to the device buffer using
    /// [`device_buffer::copy_host_to_device_buffer`].
    ///
    /// # Arguments
    ///
    /// * `py`             — Active Python GIL token.
    /// * `path`           — Path to the safetensors file.
    /// * `header_size`    — Total header size in bytes.
    /// * `body_size`      — Body size in bytes.
    /// * `device_buffer`  — Pre-allocated device buffer.
    /// * `buffer_offset`  — Byte offset within the device buffer where the body
    ///                      data should start.
    ///
    /// # Errors
    ///
    /// Returns a `PyErr` if file I/O or device copies fail.
    pub fn read_file_to_device_buffer(
        &self,
        py: Python<'_>,
        path: &str,
        header_size: usize,
        body_size: usize,
        device_buffer: &DeviceBuffer,
        buffer_offset: usize,
    ) -> PyResult<()> {
        if buffer_offset
            .checked_add(body_size)
            .map_or(true, |end| end > device_buffer.size)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "read_file_to_device_buffer: buffer_offset ({buffer_offset}) + body_size \
                 ({body_size}) exceeds device buffer size ({})",
                device_buffer.size
            )));
        }

        let is_cpu = device_buffer.device == "cpu" || device_buffer.device.starts_with("cpu:");

        if is_cpu {
            self.read_file_body_to_cpu_buffer(
                py,
                path,
                header_size,
                body_size,
                device_buffer,
                buffer_offset,
            )
        } else {
            self.read_file_body_to_gpu_buffer(
                py,
                path,
                header_size,
                body_size,
                device_buffer,
                buffer_offset,
            )
        }
    }

    // -- private helpers -----------------------------------------------------

    /// Reads file body directly into a CPU device buffer via `numpy()` data
    /// pointer or by going through the host→device copy path.
    fn read_file_body_to_cpu_buffer(
        &self,
        py: Python<'_>,
        path: &str,
        header_size: usize,
        body_size: usize,
        device_buffer: &DeviceBuffer,
        buffer_offset: usize,
    ) -> PyResult<()> {
        // For CPU tensors we can read through a host Vec and then copy_.
        // Alternatively, we could try to write directly to the tensor's data
        // pointer, but going through the standard copy path is simpler and
        // safe.
        let host_data = self
            .read_file_body_to_host(path, header_size, body_size)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        device_buffer::copy_host_to_device_buffer(py, &host_data, device_buffer, buffer_offset)
    }

    /// Reads file body into a GPU device buffer via bounce buffers + copy_.
    fn read_file_body_to_gpu_buffer(
        &self,
        py: Python<'_>,
        path: &str,
        header_size: usize,
        body_size: usize,
        device_buffer: &DeviceBuffer,
        buffer_offset: usize,
    ) -> PyResult<()> {
        let file = File::open(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to open '{path}': {e}"))
        })?;

        let chunk_size = self.bounce_buffer_size;
        let chunks: Vec<(usize, usize)> = ChunkIter::new(body_size, chunk_size).collect();

        // We cannot easily parallelise the device copy because it requires the
        // GIL and PyTorch CUDA operations are serialised on the default stream
        // anyway.  So we read chunks in parallel into host memory, then copy
        // them to the device sequentially.
        //
        // Collect all chunks into host memory first.
        let host_chunks: Result<Vec<(usize, Vec<u8>)>, String> = self.pool.install(|| {
            chunks
                .par_iter()
                .map(|&(chunk_offset, chunk_len)| {
                    let mut bounce = vec![0u8; chunk_len];
                    let file_offset = header_size as u64 + chunk_offset as u64;
                    pread_exact(&file, &mut bounce, file_offset).map_err(|e| {
                        format!("pread failed at file_offset={file_offset}, len={chunk_len}: {e}")
                    })?;
                    Ok((chunk_offset, bounce))
                })
                .collect()
        });

        let host_chunks = host_chunks.map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        // Copy each chunk to the device buffer (requires GIL).
        for (chunk_offset, data) in &host_chunks {
            device_buffer::copy_host_to_device_buffer(
                py,
                data,
                device_buffer,
                buffer_offset + chunk_offset,
            )?;
        }

        Ok(())
    }

    /// Reads a safetensors file body to GPU via `pread` + direct `cudaMemcpy`
    /// (no Python GIL needed).
    ///
    /// Pre-allocates one pinned bounce buffer per rayon thread, then
    /// processes chunks in batches so each parallel task reuses its
    /// pre-allocated buffer instead of calling `cudaHostAlloc`/`cudaFreeHost`
    /// on every chunk.
    ///
    /// Each rayon worker thread performs:
    /// 1. `cudaSetDevice` — bind the thread to the target GPU
    /// 2. `pread` — read a chunk from disk into the pre-allocated pinned buffer
    /// 3. `cudaMemcpy(H2D)` — copy from pinned buffer to device memory
    /// 4. `cudaDeviceSynchronize` — ensure the copy completes
    ///
    /// This completely bypasses Python and PyTorch, enabling true parallel
    /// disk-to-GPU transfers from Rust worker threads.
    ///
    /// # Arguments
    ///
    /// * `cuda`        — The process-wide CUDA runtime handle.
    /// * `path`        — Path to the safetensors file.
    /// * `header_size` — Total header size in bytes (body starts here).
    /// * `body_size`   — Number of body bytes to read.
    /// * `dev_ptr`     — Base CUDA device pointer for the destination buffer.
    /// * `device_id`   — CUDA device ordinal (e.g. 0 for `cuda:0`).
    ///
    /// # Errors
    ///
    /// Returns an error string if any I/O or CUDA operation fails.
    ///
    /// # Safety contract (on caller)
    ///
    /// `dev_ptr` must be a valid CUDA device pointer with at least `body_size`
    /// bytes allocated on `device_id`.
    #[cfg(feature = "gds")]
    pub fn read_file_to_gpu_direct(
        &self,
        cuda: &crate::cuda_runtime::CudaRuntime,
        path: &str,
        header_size: usize,
        body_size: usize,
        dev_ptr: *mut std::ffi::c_void,
        device_id: i32,
    ) -> Result<(), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open '{path}': {e}"))?;

        let bounce_size = self.bounce_buffer_size;
        let chunks: Vec<(usize, usize)> = ChunkIter::new(body_size, bounce_size).collect();

        if chunks.is_empty() {
            return Ok(());
        }

        // Convert the raw device pointer to a usize so it can safely cross
        // thread boundaries — each thread will reconstruct a pointer into its
        // own disjoint region of the device buffer.
        let dev_base = dev_ptr as usize;

        // Pre-allocate ONE pinned bounce buffer per rayon thread.
        // This avoids the expensive cudaHostAlloc / cudaFreeHost on every chunk.
        // We cap at the number of chunks (no point allocating more buffers than
        // chunks) and at max_threads.
        let n_buffers = self.max_threads.max(1).min(chunks.len());
        let mut pinned_bufs: Vec<usize> = Vec::with_capacity(n_buffers);
        for _ in 0..n_buffers {
            let ptr = unsafe {
                cuda.set_device(device_id)?;
                cuda.host_alloc(bounce_size)?
            };
            // Store as usize so the Vec is Send-safe for rayon threads.
            pinned_bufs.push(ptr as usize);
        }

        // Process chunks in batches of n_buffers so that each parallel task
        // in a batch gets its own pre-allocated pinned buffer (indexed by its
        // position within the batch). Batches are sequential; tasks within a
        // batch run in parallel.
        let result: Result<(), String> = self.pool.install(|| {
            for batch_start in (0..chunks.len()).step_by(n_buffers) {
                let batch_end = (batch_start + n_buffers).min(chunks.len());
                let batch = &chunks[batch_start..batch_end];

                batch.par_iter().enumerate().try_for_each(
                    |(i, &(chunk_offset, chunk_len))| -> Result<(), String> {
                        // Safety: each task in the batch uses a distinct
                        // buffer (indexed by `i`) and writes to a disjoint
                        // region of the device buffer.
                        let bounce_ptr = pinned_bufs[i] as *mut std::ffi::c_void;

                        unsafe {
                            cuda.set_device(device_id)?;

                            // pread from disk into the pre-allocated pinned buffer.
                            let bounce_slice =
                                std::slice::from_raw_parts_mut(bounce_ptr as *mut u8, chunk_len);
                            let file_offset = header_size as u64 + chunk_offset as u64;

                            pread_exact(&file, bounce_slice, file_offset).map_err(|e| {
                                format!(
                                    "pread failed at offset={file_offset}, len={chunk_len}: {e}"
                                )
                            })?;

                            // cudaMemcpy H2D into the correct offset of the device buffer.
                            let dst = (dev_base + chunk_offset) as *mut std::ffi::c_void;
                            cuda.memcpy_h2d(dst, bounce_ptr as *const std::ffi::c_void, chunk_len)
                                .map_err(|e| {
                                    format!(
                                        "cudaMemcpy H2D failed at dev_offset={chunk_offset}: {e}"
                                    )
                                })?;

                            // Wait for the copy to finish before reusing the bounce buffer.
                            cuda.synchronize()
                                .map_err(|e| format!("cudaDeviceSynchronize failed: {e}"))?;
                        }

                        Ok(())
                    },
                )?;
            }
            Ok(())
        });

        // Free all pinned bounce buffers.
        for buf in pinned_bufs {
            unsafe {
                cuda.host_free(buf as *mut std::ffi::c_void);
            }
        }

        result
    }

    /// Reads a compressed safetensors file body and decompresses it into
    /// a device buffer.
    ///
    /// The pipeline:
    /// 1. Read the compressed body from disk (parallel pread into host memory)
    /// 2. For GPU targets with nvcomp:
    ///    a. Allocate a temporary compressed buffer on GPU
    ///    b. Copy compressed data to GPU
    ///    c. Run nvcomp batched zstd decompression on GPU
    ///    d. Free the compressed GPU buffer
    /// 3. For CPU targets (or without nvcomp):
    ///    a. Decompress on CPU using the zstd crate
    ///    b. Copy decompressed data to the device buffer
    ///
    /// # Arguments
    ///
    /// * `py` — Active Python GIL token.
    /// * `path` — Path to the compressed safetensors file.
    /// * `header_size` — Total header size in bytes.
    /// * `compressed_body_size` — Size of the compressed body in bytes.
    /// * `decompressed_body_size` — Expected decompressed size.
    /// * `device_buffer` — Pre-allocated device buffer for decompressed output.
    /// * `chunks` — Chunk descriptors: `(comp_offset, comp_size, decomp_offset, decomp_size)`.
    ///
    /// # Errors
    ///
    /// Returns a `PyErr` if I/O, decompression, or device operations fail.
    #[cfg(feature = "nvcomp")]
    pub fn read_compressed_file_to_device(
        &self,
        py: Python<'_>,
        path: &str,
        header_size: usize,
        compressed_body_size: usize,
        decompressed_body_size: usize,
        device_buffer: &DeviceBuffer,
        chunks: &[(usize, usize, usize, usize)],
        algorithm: &str,
    ) -> PyResult<()> {
        let is_gpu = !(device_buffer.device == "cpu" || device_buffer.device.starts_with("cpu:"));

        // Read compressed body from disk (shared by all paths).
        let compressed_host = self
            .read_file_body_to_host(path, header_size, compressed_body_size)
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to read compressed body: {e}"))
            })?;

        if is_gpu {
            // GPU path: use nvcomp C shim for ANS/ZSTD decompression.
            // ANS can ONLY be decoded on GPU (no CPU fallback).
            // ZSTD can fall back to CPU if nvcomp is unavailable.
            let nvcomp_available = {
                use crate::nvcomp_runtime::NvcompRuntime;
                let avail = NvcompRuntime::is_available();
                if !avail {
                    eprintln!(
                        "[safetensors] nvcomp shim not available (libnvcomp_shim.so not found on LD_LIBRARY_PATH)"
                    );
                }
                avail
            };

            let is_ans = algorithm.eq_ignore_ascii_case("ans");

            if nvcomp_available {
                // Allocate compressed staging buffer on GPU, copy data, decompress.
                let device_str = &device_buffer.device;
                let compressed_gpu_buf =
                    device_buffer::allocate_device_buffer(py, compressed_body_size, device_str)?;

                device_buffer::copy_host_to_device_buffer(
                    py,
                    &compressed_host,
                    &compressed_gpu_buf,
                    0,
                )?;

                drop(compressed_host);

                device_buffer::decompress_on_gpu(
                    py,
                    &compressed_gpu_buf,
                    device_buffer,
                    chunks,
                    algorithm,
                )?;
            } else if is_ans {
                // ANS cannot be decoded on CPU — fail with a clear error.
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "ANS-compressed safetensors require nvcomp for GPU decompression, \
                     but libnvcomp_shim.so was not found. Ensure nvcomp is installed \
                     and libnvcomp_shim.so is on LD_LIBRARY_PATH.",
                ));
            } else {
                // ZSTD CPU fallback: decompress on host, then copy to GPU.
                let decompressed =
                    device_buffer::decompress_cpu_zstd(&compressed_host, decompressed_body_size)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

                drop(compressed_host);

                device_buffer::copy_host_to_device_buffer(py, &decompressed, device_buffer, 0)?;
            }
        } else {
            // CPU device path.
            let is_ans = algorithm.eq_ignore_ascii_case("ans");
            if is_ans {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "ANS-compressed safetensors cannot be loaded on CPU. \
                     Use a CUDA device with nvcomp installed.",
                ));
            }
            // ZSTD CPU path: decompress on CPU, copy to buffer
            let decompressed =
                device_buffer::decompress_cpu_zstd(&compressed_host, decompressed_body_size)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

            drop(compressed_host);

            device_buffer::copy_host_to_device_buffer(py, &decompressed, device_buffer, 0)?;
        }

        Ok(())
    }

    /// Reads a compressed safetensors file body directly to GPU via GDS,
    /// then decompresses on GPU using nvcomp.
    ///
    /// This is the highest-throughput compressed loading path:
    /// 1. cuFileRead compressed data directly from NVMe to GPU (bypasses CPU)
    /// 2. nvcomp batched zstd decompression on GPU
    ///
    /// # Arguments
    ///
    /// * `py` — Active Python GIL token.
    /// * `path` — Path to the compressed safetensors file.
    /// * `header_size` — Total header size in bytes.
    /// * `compressed_body_size` — Compressed body size in bytes.
    /// * `decompressed_buf` — Pre-allocated device buffer for decompressed output.
    /// * `chunks` — Chunk descriptors.
    /// * `device_str` — Device string (e.g. "cuda:0").
    ///
    /// # Returns
    ///
    /// `Ok(true)` if GDS + nvcomp succeeded, `Ok(false)` if GDS is not available,
    /// `Err` on failure.
    #[cfg(all(feature = "nvcomp", feature = "gds"))]
    pub fn read_compressed_file_gds_decompress(
        &self,
        py: Python<'_>,
        path: &str,
        header_size: usize,
        compressed_body_size: usize,
        decompressed_buf: &DeviceBuffer,
        chunks: &[(usize, usize, usize, usize)],
        device_str: &str,
        algorithm: &str,
    ) -> PyResult<bool> {
        // Check if GDS is available
        let gds = match device_buffer::get_gds_context() {
            Ok(g) => g,
            Err(_) => return Ok(false),
        };

        // Allocate compressed staging buffer on GPU via the buffer pool
        let compressed_gpu_buf =
            device_buffer::GdsBufferPool::acquire(py, compressed_body_size, device_str).map_err(
                |e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "GDS buffer pool acquire for compressed data failed: {e}"
                    ))
                },
            )?;

        let dev_ptr = compressed_gpu_buf.data_ptr as *mut std::ffi::c_void;

        // Read compressed data directly from NVMe to GPU via GDS
        unsafe {
            device_buffer::gds_read_file_body_pooled(
                gds,
                path,
                header_size,
                compressed_body_size,
                dev_ptr,
                0,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "GDS read of compressed data failed: {e}"
                ))
            })?;
        }

        // Decompress on GPU
        device_buffer::decompress_on_gpu(
            py,
            &compressed_gpu_buf,
            decompressed_buf,
            chunks,
            algorithm,
        )?;

        Ok(true)
    }
}

// ---- GDS bulk read ---------------------------------------------------------

/// Reads a safetensors file body directly from storage into GPU memory using
/// GPU Direct Storage (cuFile).
///
/// This bypasses host CPU memory entirely — data flows from the NVMe drive
/// through the PCIe fabric directly to GPU VRAM.
///
/// The file must reside on a GDS-compatible filesystem (e.g. ext4 with
/// `O_DIRECT` support, or a parallel filesystem with cuFile plugin).
///
/// If a [`device_buffer::CudaContext`] is available the function first checks
/// whether the target GPU supports GPUDirect RDMA; when it does not, an error
/// is returned immediately so the caller can fall back to the `pread` path.
///
/// The device buffer is registered with cuFile **once** for the entire
/// transfer (via [`device_buffer::gds_read_file_body`]) and I/O is split into
/// `max_block_size` chunks (default 1 GiB).
///
/// # Arguments
///
/// * `gds`            — An initialised [`device_buffer::GdsContext`].
/// * `path`           — Path to the safetensors file.
/// * `header_size`    — Total header size in bytes.
/// * `body_size`      — Body size in bytes.
/// * `dev_ptr`        — Base address of the CUDA device buffer.
/// * `dev_offset`     — Byte offset within the device buffer.
/// * `max_block_size` — Maximum bytes per `cuFileRead` call.  `None` defaults
///                       to 1 GiB.
///
/// # Safety
///
/// The caller must ensure that `dev_ptr` is a valid CUDA device pointer with
/// at least `dev_offset + body_size` bytes allocated.
#[cfg(feature = "gds")]
#[allow(dead_code)]
pub unsafe fn read_file_to_device_gds(
    gds: &device_buffer::GdsContext,
    path: &str,
    header_size: usize,
    body_size: usize,
    dev_ptr: *mut std::ffi::c_void,
    dev_offset: u64,
    max_block_size: Option<u64>,
) -> Result<(), String> {
    /// Default maximum chunk size: 1 GiB.
    const DEFAULT_MAX_BLOCK: u64 = 1 << 30;

    let block_size = max_block_size.unwrap_or(DEFAULT_MAX_BLOCK);

    // Safety: the caller guarantees dev_ptr validity and buffer size.
    // gds_read_file_body handles O_DIRECT open, handle/buffer registration,
    // chunked reads, and cleanup internally.
    unsafe {
        device_buffer::gds_read_file_body(
            gds,
            path,
            header_size,
            body_size,
            dev_ptr,
            dev_offset,
            block_size,
        )
    }
}

// ---- NUMA helpers ----------------------------------------------------------

/// Returns the NUMA node associated with a PCI device.
///
/// Reads `/sys/bus/pci/devices/{pci_bus_id}/numa_node`.  The `pci_bus_id`
/// should be in the format printed by `nvidia-smi` (e.g. `0000:03:00.0`).
///
/// Returns `None` if the sysfs file does not exist or cannot be parsed.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn get_device_numa_node(pci_bus_id: &str) -> Option<i32> {
    let path = format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node");
    let content = std::fs::read_to_string(&path).ok()?;
    let node: i32 = content.trim().parse().ok()?;
    // A value of -1 means "no NUMA affinity information available".
    if node < 0 {
        None
    } else {
        Some(node)
    }
}

/// Binds the calling thread to the CPUs on the given NUMA `node`.
///
/// Uses `sched_setaffinity(2)` to restrict the calling thread to the set of
/// CPUs belonging to `node`, as reported by
/// `/sys/devices/system/node/node{N}/cpulist`.
///
/// # Errors
///
/// Returns an error string if the sysfs file cannot be read, the CPU list
/// cannot be parsed, or `sched_setaffinity` fails.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn bind_thread_to_numa_node(node: i32) -> Result<(), String> {
    let cpulist_path = format!("/sys/devices/system/node/node{node}/cpulist");
    let cpulist = std::fs::read_to_string(&cpulist_path)
        .map_err(|e| format!("Failed to read '{cpulist_path}': {e}"))?;

    let cpus = parse_cpu_list(cpulist.trim())?;
    if cpus.is_empty() {
        return Err(format!("No CPUs found for NUMA node {node}"));
    }

    // Build a cpu_set_t and call sched_setaffinity.
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        for cpu in &cpus {
            libc::CPU_SET(*cpu as usize, &mut cpuset);
        }
        let ret = libc::sched_setaffinity(
            0, // 0 = calling thread
            std::mem::size_of::<libc::cpu_set_t>(),
            &cpuset,
        );
        if ret != 0 {
            return Err(format!(
                "sched_setaffinity failed: {}",
                io::Error::last_os_error()
            ));
        }
    }

    Ok(())
}

/// Sets the NUMA memory allocation policy to prefer the given `node`.
///
/// Uses `set_mempolicy(2)` with `MPOL_PREFERRED` so that future memory
/// allocations by this thread will prefer the specified node.
///
/// # Errors
///
/// Returns an error string if the syscall fails.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn set_numa_preferred_node(node: i32) -> Result<(), String> {
    // MPOL_PREFERRED = 1, per <linux/mempolicy.h>
    const MPOL_PREFERRED: i32 = 1;

    let nodemask: u64 = 1u64 << node as u64;
    let maxnode: u64 = 64; // supports up to 64 NUMA nodes

    let ret = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_PREFERRED,
            &nodemask as *const u64,
            maxnode,
        )
    };

    if ret != 0 {
        return Err(format!(
            "set_mempolicy(MPOL_PREFERRED, node={node}) failed: {}",
            io::Error::last_os_error()
        ));
    }

    Ok(())
}

// ---- Internal utilities ----------------------------------------------------

/// A simple iterator that yields `(offset, length)` pairs to partition
/// `total_size` bytes into chunks of at most `chunk_size`.
struct ChunkIter {
    total: usize,
    chunk_size: usize,
    offset: usize,
}

impl ChunkIter {
    fn new(total: usize, chunk_size: usize) -> Self {
        Self {
            total,
            chunk_size: chunk_size.max(1),
            offset: 0,
        }
    }
}

impl Iterator for ChunkIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.total {
            return None;
        }
        let remaining = self.total - self.offset;
        let len = remaining.min(self.chunk_size);
        let item = (self.offset, len);
        self.offset += len;
        Some(item)
    }
}

/// Parses a Linux CPU-list string (e.g. `"0-3,8-11"`) into a `Vec<u32>`.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn parse_cpu_list(s: &str) -> Result<Vec<u32>, String> {
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: u32 = lo
                .trim()
                .parse()
                .map_err(|e| format!("Invalid CPU number '{lo}': {e}"))?;
            let hi: u32 = hi
                .trim()
                .parse()
                .map_err(|e| format!("Invalid CPU number '{hi}': {e}"))?;
            for cpu in lo..=hi {
                cpus.push(cpu);
            }
        } else {
            let cpu: u32 = part
                .parse()
                .map_err(|e| format!("Invalid CPU number '{part}': {e}"))?;
            cpus.push(cpu);
        }
    }
    Ok(cpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_iter_exact() {
        let chunks: Vec<_> = ChunkIter::new(100, 25).collect();
        assert_eq!(chunks, vec![(0, 25), (25, 25), (50, 25), (75, 25)]);
    }

    #[test]
    fn test_chunk_iter_remainder() {
        let chunks: Vec<_> = ChunkIter::new(100, 30).collect();
        assert_eq!(chunks, vec![(0, 30), (30, 30), (60, 30), (90, 10)]);
    }

    #[test]
    fn test_chunk_iter_zero_size() {
        let chunks: Vec<_> = ChunkIter::new(0, 64).collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_iter_single_chunk() {
        let chunks: Vec<_> = ChunkIter::new(10, 1024).collect();
        assert_eq!(chunks, vec![(0, 10)]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list_range() {
        let cpus = parse_cpu_list("0-3").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list_mixed() {
        let cpus = parse_cpu_list("0-2,5,8-9").unwrap();
        assert_eq!(cpus, vec![0, 1, 2, 5, 8, 9]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list_single() {
        let cpus = parse_cpu_list("7").unwrap();
        assert_eq!(cpus, vec![7]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpu_list_empty() {
        let cpus = parse_cpu_list("").unwrap();
        assert!(cpus.is_empty());
    }

    #[test]
    fn test_bulk_file_reader_creation() {
        let reader = BulkFileReader::new(2, 64); // 2 threads, 64 KiB bounce
        assert_eq!(reader.max_threads(), 2);
        assert_eq!(reader.bounce_buffer_size(), 64 * 1024);
    }

    #[test]
    fn test_bulk_file_reader_defaults() {
        let reader = BulkFileReader::new(0, 0);
        assert_eq!(reader.bounce_buffer_size(), DEFAULT_BOUNCE_BUFFER_SIZE);
        // max_threads should be > 0 (falls back to rayon default)
        assert!(reader.max_threads() > 0);
    }
}
