//! Device buffer management for fast tensor loading.
//!
//! This module provides the Rust-side logic for allocating and managing
//! device memory buffers that hold bulk-loaded safetensors file data.
//! Instead of DLPack, we use PyTorch-native tensor operations:
//!
//! - Allocate via `torch.empty(size, dtype=torch.uint8, device=device)`
//! - Transfer via `tensor.copy_()` from CPU staging buffers
//! - Create views via slice / `.view(dtype=...)` / `.reshape()`
//!
//! GDS (GPU Direct Storage) support is available behind the `gds` feature flag,
//! using runtime dynamic loading of `libcufile.so` so the library is not required
//! at link time.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[cfg(feature = "gds")]
use std::sync::OnceLock;

#[cfg(feature = "gds")]
use std::sync::Mutex;

/// A device memory buffer backed by a PyTorch `torch.uint8` tensor.
///
/// This is an internal helper (not exposed as a `#[pyclass]`) that wraps
/// a PyTorch tensor allocated on an arbitrary device (CPU, CUDA, etc.)
/// and provides convenience methods for offset calculations.
pub struct DeviceBuffer {
    /// The PyTorch tensor (`torch.uint8`) that owns the device memory.
    pub tensor: PyObject,
    /// The raw data pointer obtained from `tensor.data_ptr()`, used for offset math.
    #[allow(dead_code)]
    pub data_ptr: u64,
    /// Total size of the buffer in bytes.
    pub size: usize,
    /// Device string (e.g. `"cuda:0"`, `"cpu"`).
    pub device: String,
}

/// Imports the `torch` module, caching the import for the lifetime of the process.
fn import_torch(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("torch")
}

/// Allocates a device buffer of the given `size` (in bytes) on `device`.
///
/// Internally calls `torch.empty(size, dtype=torch.uint8, device=device)`.
///
/// # Arguments
///
/// * `py`     — Active Python GIL token.
/// * `size`   — Number of bytes to allocate.
/// * `device` — PyTorch device string, e.g. `"cuda:0"` or `"cpu"`.
///
/// # Errors
///
/// Returns a `PyErr` if the `torch` module cannot be imported or allocation fails.
pub fn allocate_device_buffer(py: Python<'_>, size: usize, device: &str) -> PyResult<DeviceBuffer> {
    let torch = import_torch(py)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", torch.getattr("uint8")?)?;
    kwargs.set_item("device", device)?;

    let tensor = torch.call_method("empty", (size,), Some(&kwargs))?.unbind();

    let data_ptr: u64 = tensor.bind(py).call_method0("data_ptr")?.extract()?;

    Ok(DeviceBuffer {
        tensor,
        data_ptr,
        size,
        device: device.to_string(),
    })
}

/// Copies `host_data` into `buffer` starting at byte `offset`.
///
/// The implementation creates a CPU tensor via `torch.frombuffer()` wrapping the
/// host data, then copies it into the target slice of the device buffer using
/// `device_tensor[offset:offset+len].copy_(cpu_tensor)`.
///
/// # Arguments
///
/// * `py`        — Active Python GIL token.
/// * `host_data` — Source bytes in host memory.
/// * `buffer`    — Destination device buffer.
/// * `offset`    — Byte offset within `buffer` to start writing.
///
/// # Errors
///
/// Returns a `PyErr` if the offset + data length exceeds the buffer size, or
/// if any PyTorch operation fails.
pub fn copy_host_to_device_buffer(
    py: Python<'_>,
    host_data: &[u8],
    buffer: &DeviceBuffer,
    offset: usize,
) -> PyResult<()> {
    let len = host_data.len();
    if offset
        .checked_add(len)
        .map_or(true, |end| end > buffer.size)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "copy_host_to_device_buffer: offset ({offset}) + length ({len}) exceeds buffer size ({})",
            buffer.size
        )));
    }
    if len == 0 {
        return Ok(());
    }

    let torch = import_torch(py)?;

    // Create a Python `bytes` object so the data is owned by the Python runtime
    // and remains valid for `torch.frombuffer`.
    let py_bytes = pyo3::types::PyBytes::new(py, host_data);

    let frombuffer_kwargs = PyDict::new(py);
    frombuffer_kwargs.set_item("dtype", torch.getattr("uint8")?)?;

    let cpu_tensor = torch.call_method("frombuffer", (py_bytes,), Some(&frombuffer_kwargs))?;

    // Slice the device buffer: buffer.tensor[offset : offset + len]
    let bound_tensor = buffer.tensor.bind(py);
    let slice = bound_tensor.call_method1(
        "__getitem__",
        (pyo3::types::PySlice::new(
            py,
            offset as isize,
            (offset + len) as isize,
            1,
        ),),
    )?;

    // Copy from host to device
    slice.call_method1("copy_", (cpu_tensor,))?;

    Ok(())
}

/// Creates a typed, shaped PyTorch tensor view into a region of the device buffer.
///
/// Steps:
/// 1. Slice the raw `uint8` buffer at `[offset .. offset + length]`.
/// 2. Reinterpret the slice as `dtype` via `.view(dtype=<torch_dtype>)`.
/// 3. Reshape to the requested `shape`.
///
/// The returned tensor is a **view** — it shares storage with the device buffer.
///
/// # Arguments
///
/// * `py`        — Active Python GIL token.
/// * `buffer`    — The source device buffer.
/// * `offset`    — Byte offset of the tensor data within the buffer.
/// * `length`    — Length of the tensor data in bytes.
/// * `dtype_str` — Safetensors dtype string (e.g. `"F32"`, `"BF16"`, `"I64"`).
/// * `shape`     — Desired tensor shape.
///
/// # Errors
///
/// Returns a `PyErr` if the slice is out of bounds, the dtype string is
/// unrecognised, or any PyTorch operation fails.
pub fn create_tensor_view(
    py: Python<'_>,
    buffer: &DeviceBuffer,
    offset: usize,
    length: usize,
    dtype_str: &str,
    shape: Vec<usize>,
) -> PyResult<PyObject> {
    if offset
        .checked_add(length)
        .map_or(true, |end| end > buffer.size)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "create_tensor_view: offset ({offset}) + length ({length}) exceeds buffer size ({})",
            buffer.size
        )));
    }

    let torch = import_torch(py)?;
    let torch_dtype = safetensors_dtype_to_torch(py, &torch, dtype_str)?;

    // Slice: buffer.tensor[offset : offset + length]
    let bound_tensor = buffer.tensor.bind(py);
    let buf_slice = bound_tensor.call_method1(
        "__getitem__",
        (pyo3::types::PySlice::new(
            py,
            offset as isize,
            (offset + length) as isize,
            1,
        ),),
    )?;

    // View as the target dtype
    let view_kwargs = PyDict::new(py);
    view_kwargs.set_item("dtype", torch_dtype)?;
    let typed_view = buf_slice.call_method("view", (), Some(&view_kwargs))?;

    // Reshape to the requested shape
    let shape_tuple = PyTuple::new(py, shape.iter().map(|&s| s as i64))?;
    let reshaped = typed_view.call_method1("reshape", (shape_tuple,))?;

    Ok(reshaped.unbind())
}

/// Fixes alignment issues within a device buffer by copying data internally.
///
/// When a safetensors file has an odd-sized header, the tensor data region
/// may start at an address that is not aligned to the element size of some
/// tensors (e.g. a `float32` tensor at an odd offset). GPUs often require
/// natural alignment for efficient access. This function resolves such issues
/// by copying the affected byte ranges to properly aligned destinations
/// **within the same buffer**.
///
/// Each fixup is a tuple `(src_offset, dst_offset, length)` describing a
/// byte-range copy inside the buffer.
///
/// # Arguments
///
/// * `py`     — Active Python GIL token.
/// * `buffer` — The device buffer to fix up in-place.
/// * `fixups` — A list of `(src_offset, dst_offset, length)` copy operations.
///
/// # Errors
///
/// Returns a `PyErr` if any offset is out of bounds or a PyTorch operation fails.
pub fn fix_alignment(
    py: Python<'_>,
    buffer: &DeviceBuffer,
    fixups: &[(usize, usize, usize)],
) -> PyResult<()> {
    if fixups.is_empty() {
        return Ok(());
    }

    let bound_tensor = buffer.tensor.bind(py);

    for &(src_offset, dst_offset, length) in fixups {
        if src_offset
            .checked_add(length)
            .map_or(true, |end| end > buffer.size)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fix_alignment: src_offset ({src_offset}) + length ({length}) exceeds buffer size ({})",
                buffer.size
            )));
        }
        if dst_offset
            .checked_add(length)
            .map_or(true, |end| end > buffer.size)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fix_alignment: dst_offset ({dst_offset}) + length ({length}) exceeds buffer size ({})",
                buffer.size
            )));
        }

        // We must `.clone()` the source slice first so that it does not alias
        // the destination during `copy_`. PyTorch's `.clone()` allocates a
        // temporary tensor with the same data.
        let src_slice = bound_tensor.call_method1(
            "__getitem__",
            (pyo3::types::PySlice::new(
                py,
                src_offset as isize,
                (src_offset + length) as isize,
                1,
            ),),
        )?;
        let src_cloned = src_slice.call_method0("clone")?;

        let dst_slice = bound_tensor.call_method1(
            "__getitem__",
            (pyo3::types::PySlice::new(
                py,
                dst_offset as isize,
                (dst_offset + length) as isize,
                1,
            ),),
        )?;

        dst_slice.call_method1("copy_", (src_cloned,))?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a safetensors dtype string (e.g. `"F32"`, `"BF16"`) to the
/// corresponding `torch.dtype` Python object.
pub(crate) fn safetensors_dtype_to_torch<'py>(
    _py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    dtype_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // Accept both uppercase safetensors format ("F32", "BF16") and lowercase
    // torch attribute names ("float32", "bfloat16") so callers can use either.
    let attr_name = match dtype_str {
        // Uppercase safetensors format
        "BOOL" => "bool",
        "U8" => "uint8",
        "I8" => "int8",
        "I16" => "int16",
        "U16" => "uint16",
        "I32" => "int32",
        "U32" => "uint32",
        "I64" => "int64",
        "U64" => "uint64",
        "F16" => "float16",
        "BF16" => "bfloat16",
        "F32" => "float32",
        "F64" => "float64",
        "F8_E5M2" => "float8_e5m2",
        "F8_E4M3" => "float8_e4m3fn",
        "F8_E8M0" => "float8_e8m0fnu",
        "F4" => "float4_e2m1fn_x2",
        "C64" => "complex64",
        // Lowercase torch attribute names (already valid)
        "bool" | "uint8" | "int8" | "int16" | "uint16" | "int32" | "uint32" | "int64"
        | "uint64" | "float16" | "bfloat16" | "float32" | "float64" | "float8_e5m2"
        | "float8_e4m3fn" | "float8_e8m0fnu" | "float4_e2m1fn_x2" | "complex64" => dtype_str,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported safetensors dtype for PyTorch conversion: {other}"
            )));
        }
    };
    torch.getattr(attr_name).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to get torch.{attr_name} for dtype '{dtype_str}': {e}"
        ))
    })
}

/// Computes alignment fixups needed for a buffer of tensor data.
///
/// When the header length (including the 8-byte length prefix) is not a
/// multiple of a tensor's element size, the tensor's data inside the
/// buffer will be misaligned. This function scans all tensors and
/// produces a list of `(src_offset, dst_offset, length)` copy operations
/// that, when applied, move each misaligned tensor's data to the next
/// properly aligned position within a buffer that has been allocated
/// with sufficient padding.
///
/// # Arguments
///
/// * `tensors`         — Iterable of `(name, dtype_str, (start_offset, end_offset))`.
/// * `required_align`  — Minimum alignment in bytes (typically 8 or 16).
///
/// # Returns
///
/// A `Vec` of `(src_offset, dst_offset, length)` fixup operations, which
/// may be empty if all tensors are already aligned.
#[allow(dead_code)]
pub fn compute_alignment_fixups(
    tensors: &[(String, usize, usize, usize)], // (name, element_bytes, start, end)
    required_align: usize,
) -> Vec<(usize, usize, usize)> {
    let mut fixups = Vec::new();
    for (_name, elem_bytes, start, end) in tensors {
        let align = (*elem_bytes).max(required_align);
        if *start % align != 0 {
            let aligned_start = (*start + align - 1) & !(align - 1);
            let length = end - start;
            fixups.push((*start, aligned_start, length));
        }
    }
    fixups
}

// ===========================================================================
// GDS (GPU Direct Storage) support
// ===========================================================================

/// Context for CUDA runtime queries (driver version, device attributes).
///
/// Dynamically loads `libcudart.so` at runtime so the binary does not require
/// it at link time.
#[cfg(feature = "gds")]
#[allow(dead_code)]
pub struct CudaContext {
    _lib: libloading::Library,
    device_get_attribute: unsafe extern "C" fn(value: *mut i32, attr: i32, device: i32) -> i32,
    driver_get_version: unsafe extern "C" fn(version: *mut i32) -> i32,
}

#[cfg(feature = "gds")]
impl CudaContext {
    /// Loads the CUDA runtime library and resolves the symbols we need.
    ///
    /// # Safety
    ///
    /// This function loads shared-library symbols at runtime.  The loaded
    /// function pointers are assumed to match the CUDA runtime ABI.
    pub unsafe fn new() -> Result<Self, String> {
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

        let device_get_attribute = load_sym!(
            lib,
            b"cudaDeviceGetAttribute\0",
            unsafe extern "C" fn(*mut i32, i32, i32) -> i32
        );
        let driver_get_version = load_sym!(
            lib,
            b"cudaDriverGetVersion\0",
            unsafe extern "C" fn(*mut i32) -> i32
        );

        Ok(Self {
            _lib: lib,
            device_get_attribute: std::mem::transmute(device_get_attribute),
            driver_get_version: std::mem::transmute(driver_get_version),
        })
    }

    /// Returns `true` if the GPU supports GPUDirect RDMA (a prerequisite for GDS).
    ///
    /// Queries `cudaDevAttrGPUDirectRDMASupported` (attribute id **96**).
    pub fn is_gds_supported(&self, device_id: i32) -> Result<bool, String> {
        let mut value: i32 = 0;
        // cudaDevAttrGPUDirectRDMASupported = 96
        // Safety: we pass a valid pointer and a well-known attribute enum value.
        let status = unsafe { (self.device_get_attribute)(&mut value, 96, device_id) };
        if status != 0 {
            return Err(format!(
                "cudaDeviceGetAttribute(GPUDirectRDMASupported) returned error code {status}"
            ));
        }
        Ok(value != 0)
    }

    /// Returns the CUDA driver version encoded as `major * 1000 + minor * 10`.
    #[allow(dead_code)]
    pub fn driver_version(&self) -> Result<i32, String> {
        let mut version: i32 = 0;
        // Safety: we pass a valid pointer.
        let status = unsafe { (self.driver_get_version)(&mut version) };
        if status != 0 {
            return Err(format!(
                "cudaDriverGetVersion() returned error code {status}"
            ));
        }
        Ok(version)
    }
}

/// Determines whether `O_DIRECT` is needed for GDS based on the CUDA version.
///
/// CUDA >= 12.2 (GDS 1.7) supports non-`O_DIRECT` file descriptors, so
/// `O_DIRECT` is only required for older drivers.
#[cfg(feature = "gds")]
#[allow(dead_code)]
pub fn needs_o_direct(cuda_ctx: &CudaContext) -> Result<bool, String> {
    let version = cuda_ctx.driver_version()?;
    // version is major*1000 + minor*10
    Ok(version < 12020) // CUDA < 12.2 needs O_DIRECT
}

/// Context for GPU Direct Storage operations.
///
/// This struct dynamically loads `libcufile.so` at runtime so that the binary
/// can be built and run on systems where the library is not installed — GDS
/// operations will simply fail with a descriptive error in that case.
#[cfg(feature = "gds")]
#[allow(dead_code)]
pub struct GdsContext {
    _lib: libloading::Library,

    // Function pointers loaded from libcufile.so
    driver_open: unsafe extern "C" fn() -> CUfileError,
    driver_close: unsafe extern "C" fn() -> CUfileError,
    buf_register: unsafe extern "C" fn(
        dev_ptr: *const std::ffi::c_void,
        size: usize,
        flags: i32,
    ) -> CUfileError,
    buf_deregister: unsafe extern "C" fn(dev_ptr: *const std::ffi::c_void) -> CUfileError,
    handle_register:
        unsafe extern "C" fn(fh: *mut CUfileHandle, descr: *mut CUfileDescr) -> CUfileError,
    handle_deregister: unsafe extern "C" fn(fh: CUfileHandle),
    file_read: unsafe extern "C" fn(
        fh: CUfileHandle,
        buf: *mut std::ffi::c_void,
        size: usize,
        file_offset: i64,
        buf_offset: i64,
    ) -> isize,
    get_version: unsafe extern "C" fn(version: *mut i32) -> i32,
}

/// cuFile error type — a pair of error codes.
#[cfg(feature = "gds")]
#[repr(C)]
pub struct CUfileError {
    /// CUfileOpError enum; 0 = CU_FILE_SUCCESS
    pub err: i32,
    /// CUresult
    pub cu_err: i32,
}

/// Opaque file handle returned by cuFileHandleRegister.
#[cfg(feature = "gds")]
pub type CUfileHandle = *mut std::ffi::c_void;

/// cuFile file descriptor — matches CUfileDescr_t layout on Linux x86_64.
#[cfg(feature = "gds")]
#[repr(C)]
pub struct CUfileDescr {
    /// Handle type — `CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1`.
    pub handle_type: i32,
    /// Padding for union alignment.
    _pad: i32,
    /// The file descriptor (union { int fd; void* handle } — pointer-sized).
    pub handle_fd: i64,
    /// Filesystem operations pointer, typically null.
    pub fs_ops: *const std::ffi::c_void,
}

#[cfg(feature = "gds")]
impl CUfileDescr {
    /// Creates a new descriptor wrapping a raw file descriptor.
    ///
    /// `CU_FILE_HANDLE_TYPE_OPAQUE_FD` is defined as `1` in `cufile.h`.
    pub fn from_raw_fd(fd: std::os::unix::io::RawFd) -> Self {
        Self {
            handle_type: 1, // CU_FILE_HANDLE_TYPE_OPAQUE_FD
            _pad: 0,
            handle_fd: fd as i64,
            fs_ops: std::ptr::null(),
        }
    }
}

#[cfg(feature = "gds")]
impl GdsContext {
    /// Attempts to open the cuFile driver by dynamically loading `libcufile.so`.
    ///
    /// # Errors
    ///
    /// Returns an error string if the library cannot be loaded or if
    /// `cuFileDriverOpen()` returns a non-zero status.
    ///
    /// # Safety
    ///
    /// This function loads shared library symbols at runtime. The loaded
    /// function pointers are assumed to match the cuFile ABI.
    pub unsafe fn new() -> Result<Self, String> {
        let lib = libloading::Library::new("libcufile.so").map_err(|e| {
            format!("Failed to load libcufile.so — is GDS / cuFile installed? Error: {e}")
        })?;

        macro_rules! load_sym {
            ($lib:expr, $name:literal, $ty:ty) => {{
                let sym: libloading::Symbol<$ty> = $lib
                    .get($name)
                    .map_err(|e| format!("Failed to load symbol {}: {e}", stringify!($name)))?;
                *sym
            }};
        }

        let driver_open = load_sym!(
            lib,
            b"cuFileDriverOpen\0",
            unsafe extern "C" fn() -> CUfileError
        );
        let driver_close = load_sym!(
            lib,
            b"cuFileDriverClose\0",
            unsafe extern "C" fn() -> CUfileError
        );
        let buf_register = load_sym!(
            lib,
            b"cuFileBufRegister\0",
            unsafe extern "C" fn(*const std::ffi::c_void, usize, i32) -> CUfileError
        );
        let buf_deregister = load_sym!(
            lib,
            b"cuFileBufDeregister\0",
            unsafe extern "C" fn(*const std::ffi::c_void) -> CUfileError
        );
        let handle_register = load_sym!(
            lib,
            b"cuFileHandleRegister\0",
            unsafe extern "C" fn(*mut CUfileHandle, *mut CUfileDescr) -> CUfileError
        );
        let handle_deregister = load_sym!(
            lib,
            b"cuFileHandleDeregister\0",
            unsafe extern "C" fn(CUfileHandle)
        );
        let file_read = load_sym!(
            lib,
            b"cuFileRead\0",
            unsafe extern "C" fn(CUfileHandle, *mut std::ffi::c_void, usize, i64, i64) -> isize
        );
        let get_version = load_sym!(
            lib,
            b"cuFileGetVersion\0",
            unsafe extern "C" fn(*mut i32) -> i32
        );

        let result = (driver_open)();
        if result.err != 0 {
            return Err(format!(
                "cuFileDriverOpen() returned error code {}",
                result.err
            ));
        }

        Ok(Self {
            _lib: lib,
            driver_open: std::mem::transmute(driver_open),
            driver_close: std::mem::transmute(driver_close),
            buf_register: std::mem::transmute(buf_register),
            buf_deregister: std::mem::transmute(buf_deregister),
            handle_register: std::mem::transmute(handle_register),
            handle_deregister: std::mem::transmute(handle_deregister),
            file_read: std::mem::transmute(file_read),
            get_version: std::mem::transmute(get_version),
        })
    }

    /// Returns the cuFile library version.
    ///
    /// The version is encoded as a single integer whose meaning is
    /// library-specific (consult the cuFile release notes).
    #[allow(dead_code)]
    pub fn version(&self) -> Result<i32, String> {
        let mut version: i32 = 0;
        // Safety: we pass a valid pointer for the output parameter.
        let status = unsafe { (self.get_version)(&mut version) };
        if status != 0 {
            return Err(format!("cuFileGetVersion() returned error code {status}"));
        }
        Ok(version)
    }

    /// Registers a device buffer with cuFile for DMA access.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must be a valid CUDA device pointer and `size` must not
    /// exceed the allocation.
    pub unsafe fn register_buffer(
        &self,
        dev_ptr: *mut std::ffi::c_void,
        size: usize,
    ) -> Result<(), String> {
        let result = (self.buf_register)(dev_ptr as *const std::ffi::c_void, size, 0);
        if result.err != 0 {
            return Err(format!(
                "cuFileBufRegister() returned error code {}",
                result.err
            ));
        }
        Ok(())
    }

    /// Deregisters a previously registered device buffer.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must have been previously registered via [`register_buffer`].
    pub unsafe fn deregister_buffer(&self, dev_ptr: *mut std::ffi::c_void) -> Result<(), String> {
        let result = (self.buf_deregister)(dev_ptr as *const std::ffi::c_void);
        if result.err != 0 {
            return Err(format!(
                "cuFileBufDeregister() returned error code {}",
                result.err
            ));
        }
        Ok(())
    }

    /// Registers a file descriptor with cuFile.
    ///
    /// # Safety
    ///
    /// The file descriptor in `descr` must be a valid, open file descriptor
    /// for a file on a GDS-compatible filesystem.
    pub unsafe fn register_handle(&self, descr: &mut CUfileDescr) -> Result<CUfileHandle, String> {
        let mut fh: CUfileHandle = std::ptr::null_mut();
        let result = (self.handle_register)(&mut fh, descr as *mut CUfileDescr);
        if result.err != 0 {
            return Err(format!(
                "cuFileHandleRegister() returned error code {}",
                result.err
            ));
        }
        Ok(fh)
    }

    /// Deregisters a file handle.
    ///
    /// # Safety
    ///
    /// The handle must have been previously registered.
    pub unsafe fn deregister_handle(&self, fh: CUfileHandle) {
        (self.handle_deregister)(fh);
    }

    /// Reads from a file directly into GPU device memory via GDS.
    ///
    /// # Arguments
    ///
    /// * `descr`       — A registered cuFile file descriptor.
    /// * `dev_ptr`     — Base device pointer (must be cuFile-registered).
    /// * `size`        — Number of bytes to read.
    /// * `file_offset` — Byte offset within the file to start reading.
    /// * `dev_offset`  — Byte offset within the device buffer.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.
    ///
    /// # Safety
    ///
    /// All pointers must be valid, the buffer registered, and offsets in range.
    pub unsafe fn read(
        &self,
        fh: CUfileHandle,
        dev_ptr: *mut std::ffi::c_void,
        size: usize,
        file_offset: i64,
        dev_offset: i64,
    ) -> Result<usize, String> {
        let ret = (self.file_read)(fh, dev_ptr, size, file_offset, dev_offset);
        if ret < 0 {
            return Err(format!("cuFileRead() returned error code {ret}"));
        }
        Ok(ret as usize)
    }

    /// Closes the cuFile driver.
    ///
    /// # Safety
    ///
    /// Must only be called once, and only after all cuFile operations are
    /// complete.
    #[allow(dead_code)]
    pub unsafe fn close_driver(&self) -> Result<(), String> {
        let result = (self.driver_close)();
        if result.err != 0 {
            return Err(format!(
                "cuFileDriverClose() returned error code {}",
                result.err
            ));
        }
        Ok(())
    }
}

/// Reads file data directly into a device buffer using GPU Direct Storage.
///
/// This is the legacy convenience function that registers **and** deregisters
/// both the file handle and the device buffer on every call.  For bulk reads
/// prefer [`gds_read_file_body`] which amortises registration cost.
///
/// # Arguments
///
/// * `gds`         — An initialised GDS context.
/// * `fd`          — A raw file descriptor opened with `O_DIRECT`.
/// * `dev_ptr`     — Base address of the device buffer.
/// * `file_offset` — Byte offset into the file (typically `header_size + 8`).
/// * `size`        — Number of bytes to read.
/// * `dev_offset`  — Byte offset into the device buffer where data should land.
///
/// # Safety
///
/// The caller must ensure that `fd` is a valid open file descriptor,
/// `dev_ptr` is a valid CUDA device pointer with at least `dev_offset + size`
/// bytes allocated, and the GDS context has been properly initialised.
#[cfg(feature = "gds")]
#[deprecated(note = "Use gds_read_file_body() which amortises buffer registration")]
#[allow(dead_code)]
pub unsafe fn gds_read_to_device(
    gds: &GdsContext,
    fd: std::os::unix::io::RawFd,
    dev_ptr: *mut std::ffi::c_void,
    file_offset: u64,
    size: u64,
    dev_offset: u64,
) -> Result<u64, String> {
    // Register the file handle
    let mut descr = CUfileDescr::from_raw_fd(fd);
    let fh = gds.register_handle(&mut descr)?;

    // Register the device buffer for cuFile DMA
    gds.register_buffer(dev_ptr, (dev_offset + size) as usize)?;

    // Perform the read — cuFile requires 512-byte aligned file offsets for
    // optimal performance but will handle unaligned offsets with a fallback.
    let bytes_read = match gds.read(
        fh,
        dev_ptr,
        size as usize,
        file_offset as i64,
        dev_offset as i64,
    ) {
        Ok(n) => n,
        Err(e) => {
            // Best-effort cleanup
            let _ = gds.deregister_buffer(dev_ptr);
            gds.deregister_handle(fh);
            return Err(e);
        }
    };

    // Cleanup
    let _ = gds.deregister_buffer(dev_ptr);
    gds.deregister_handle(fh);

    Ok(bytes_read as u64)
}

/// Reads a file body into a registered device buffer using GDS with per-block reads.
///
/// Unlike [`gds_read_to_device`] this function registers the device buffer
/// **once** for the entire read and splits the I/O into `max_block_size`
/// chunks, which avoids the overhead of repeated registration / deregistration.
///
/// The steps are:
///
/// 1. Opens the file with `O_DIRECT`.
/// 2. Registers the file handle with cuFile.
/// 3. Registers the device buffer with cuFile (**once** for the whole buffer).
/// 4. Reads in `max_block_size` chunks via `cuFileRead`, handling partial reads.
/// 5. Deregisters the buffer and file handle.
///
/// # Arguments
///
/// * `gds`            — An initialised GDS context.
/// * `path`           — Filesystem path to the safetensors file.
/// * `header_size`    — Total header size in bytes (body starts at this offset).
/// * `body_size`      — Number of body bytes to read.
/// * `dev_ptr`        — Base address of the CUDA device buffer.
/// * `dev_offset`     — Byte offset within the device buffer where data lands.
/// * `max_block_size` — Maximum bytes per `cuFileRead` call (default 1 GiB).
///
/// # Safety
///
/// The caller must ensure that `dev_ptr` is a valid CUDA device pointer with
/// at least `dev_offset + body_size` bytes allocated, and the GDS context has
/// been properly initialised.
#[cfg(feature = "gds")]
pub unsafe fn gds_read_file_body(
    gds: &GdsContext,
    path: &str,
    header_size: usize,
    body_size: usize,
    dev_ptr: *mut std::ffi::c_void,
    dev_offset: u64,
    max_block_size: u64,
) -> Result<(), String> {
    use std::io;

    // -- Open the file with O_DIRECT for GDS compatibility -------------------
    let c_path = std::ffi::CString::new(path).map_err(|e| format!("Invalid path '{path}': {e}"))?;
    // Safety: c_path is a valid NUL-terminated string.
    let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT);
    if fd < 0 {
        return Err(format!(
            "Failed to open '{path}' with O_DIRECT: {}",
            io::Error::last_os_error()
        ));
    }

    // RAII guard so we always close the fd.
    struct FdGuard(i32);
    impl Drop for FdGuard {
        fn drop(&mut self) {
            // Safety: self.0 is a valid open fd.
            unsafe {
                libc::close(self.0);
            }
        }
    }
    let _fd_guard = FdGuard(fd);

    // -- Register the file handle with cuFile --------------------------------
    let mut descr = CUfileDescr::from_raw_fd(fd);
    let fh = gds.register_handle(&mut descr)?;

    // -- Register the device buffer ONCE for the whole transfer ---------------
    let total_buf_size = (dev_offset as usize) + body_size;
    if let Err(e) = gds.register_buffer(dev_ptr, total_buf_size) {
        // Best-effort: deregister the handle before propagating.
        gds.deregister_handle(fh);
        return Err(e);
    }

    // -- Read in max_block_size chunks, handling partial reads ----------------
    let file_offset_base = header_size as u64;
    let mut remaining = body_size as u64;
    let mut cursor: u64 = 0; // bytes already transferred

    let result: Result<(), String> = (|| {
        while remaining > 0 {
            let chunk = remaining.min(max_block_size);
            let mut chunk_remaining = chunk;

            while chunk_remaining > 0 {
                let bytes_read = gds.read(
                    fh,
                    dev_ptr,
                    chunk_remaining as usize,
                    (file_offset_base + cursor) as i64,
                    (dev_offset + cursor) as i64,
                )?;

                if bytes_read == 0 {
                    return Err(format!(
                        "cuFileRead returned 0 bytes at file_offset={}, expected {chunk_remaining}",
                        file_offset_base + cursor
                    ));
                }

                let n = bytes_read as u64;
                cursor += n;
                remaining -= n;
                chunk_remaining -= n;
            }
        }
        Ok(())
    })();

    // -- Cleanup (best-effort) -----------------------------------------------
    let _ = gds.deregister_buffer(dev_ptr);
    gds.deregister_handle(fh);

    result
}

// ===========================================================================
// GDS driver singleton
// ===========================================================================

/// Process-wide cached GDS driver context.
///
/// The cuFile driver is expensive to open (~743 ms) but only needs to be
/// initialised once per process.  By caching it in a [`OnceLock`] we pay
/// that cost on the first `fast_safe_open` call and amortise it over every
/// subsequent file load.
///
/// The singleton is intentionally **never** closed — `cuFileDriverClose` is
/// not called — because the driver must outlive all cuFile I/O and there is
/// no reliable shutdown hook in a Python extension module.  The CUDA runtime
/// tears it down automatically when the process exits.
#[cfg(feature = "gds")]
static GDS_DRIVER: OnceLock<Result<GdsContext, String>> = OnceLock::new();

/// Returns a reference to the process-wide [`GdsContext`] singleton.
///
/// On the first call the cuFile driver is opened (via `cuFileDriverOpen`).
/// Subsequent calls return the cached result in O(1).
///
/// # Errors
///
/// Returns the original error string if the driver could not be opened.
#[cfg(feature = "gds")]
pub fn get_gds_context() -> Result<&'static GdsContext, String> {
    GDS_DRIVER
        .get_or_init(|| {
            // Safety: GdsContext::new loads libcufile.so symbols at runtime
            // and calls cuFileDriverOpen().
            unsafe { GdsContext::new() }
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Reads a safetensors file body into GPU memory via GDS using the
/// **process-wide cached** driver context.
///
/// This is the recommended entry point for GDS reads.  It is functionally
/// identical to [`gds_read_file_body`] but obtains the [`GdsContext`] from
/// the singleton cache, avoiding the ~743 ms `cuFileDriverOpen` overhead on
/// every file.
///
/// # Arguments
///
/// * `gds`            — A reference to the cached [`GdsContext`] (obtain via
///                      [`get_gds_context`]).
/// * `path`           — Filesystem path to the safetensors file.
/// * `header_size`    — Total header size in bytes (body starts here).
/// * `body_size`      — Number of body bytes to read.
/// * `dev_ptr`        — Base address of the CUDA device buffer.
/// * `dev_offset`     — Byte offset within the device buffer where data lands.
///
/// # Safety
///
/// The caller must ensure that `dev_ptr` is a valid CUDA device pointer with
/// at least `dev_offset + body_size` bytes allocated, and the GDS context has
/// been properly initialised (i.e. obtained from [`get_gds_context`]).
#[cfg(feature = "gds")]
#[allow(dead_code)]
pub unsafe fn gds_read_file_body_cached(
    gds: &GdsContext,
    path: &str,
    header_size: usize,
    body_size: usize,
    dev_ptr: *mut std::ffi::c_void,
    dev_offset: u64,
) -> Result<(), String> {
    // Default block size: 1 GiB — same as the non-cached path.
    let max_block_size: u64 = 1 << 30;

    // Safety: caller guarantees dev_ptr validity and buffer size.
    gds_read_file_body(
        gds,
        path,
        header_size,
        body_size,
        dev_ptr,
        dev_offset,
        max_block_size,
    )
}

// ===========================================================================
// GDS pre-registered buffer pool
// ===========================================================================

/// A process-wide pool of a single pre-registered GPU buffer for GDS reads.
///
/// Registering (`cuFileBufRegister`) and deregistering (`cuFileBufDeregister`)
/// a device buffer with cuFile costs approximately 10 ms and 8 ms respectively.
/// By caching the registration across file loads, we save ~18 ms per file.
///
/// The pool holds a single PyTorch tensor on a specific device.  When a new
/// load request arrives that fits in the existing buffer (same device, large
/// enough), the buffer is reused without re-registration.  If it does not
/// fit, the old buffer is deregistered and a new, larger one is allocated and
/// registered.
///
/// **Lifetime**: The pool tensor is kept alive by the `PyObject` reference.
/// Tensor views created from it hold their own reference (via PyTorch's
/// reference counting), so the underlying memory is not freed until all
/// views are dropped — even after the pool itself is replaced with a larger
/// buffer.
#[cfg(feature = "gds")]
pub struct GdsBufferPool {
    /// The PyTorch tensor (`torch.uint8`) backing the buffer.
    tensor: PyObject,
    /// Raw device pointer obtained from `tensor.data_ptr()`.
    ptr: u64,
    /// Current allocated size in bytes.
    size: usize,
    /// Device string (e.g. `"cuda:0"`).
    device: String,
}

/// Process-wide singleton for the GDS pre-registered buffer pool.
///
/// Protected by a [`Mutex`] so that concurrent Python threads serialise
/// their access.  The lock is held only for the short duration of the
/// pool check / swap — the actual I/O happens outside the lock.
#[cfg(feature = "gds")]
static GDS_BUFFER_POOL: Mutex<Option<GdsBufferPool>> = Mutex::new(None);

#[cfg(feature = "gds")]
impl GdsBufferPool {
    /// Acquires a pre-registered device buffer of at least `min_size` bytes
    /// on `device`.
    ///
    /// If the existing pool buffer is large enough and on the same device it
    /// is returned directly (zero-cost reuse).  Otherwise a new buffer is
    /// allocated, rounded up to the next 256 MiB boundary, and registered
    /// with cuFile.
    ///
    /// The returned [`DeviceBuffer`] borrows the pool's tensor via a
    /// `clone_ref` — PyTorch's reference counting ensures the underlying
    /// memory stays alive even after the pool is later replaced.
    ///
    /// # Errors
    ///
    /// Returns a `PyErr` if buffer allocation fails or if cuFile
    /// registration fails.
    pub fn acquire(py: Python<'_>, min_size: usize, device: &str) -> PyResult<DeviceBuffer> {
        let mut guard = GDS_BUFFER_POOL.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("GDS buffer pool lock poisoned: {e}"))
        })?;

        // Fast path: existing pool is large enough and on the right device.
        if let Some(pool) = guard.as_ref() {
            if pool.size >= min_size && pool.device == device {
                return Ok(DeviceBuffer {
                    tensor: pool.tensor.clone_ref(py),
                    data_ptr: pool.ptr,
                    size: pool.size,
                    device: pool.device.clone(),
                });
            }

            // Pool exists but wrong size/device — deregister the old buffer.
            if let Ok(gds) = get_gds_context() {
                // Safety: pool.ptr was previously registered via register_buffer.
                unsafe {
                    let _ = gds.deregister_buffer(pool.ptr as *mut std::ffi::c_void);
                }
            }
        }

        // Round up to the next 256 MiB boundary for headroom.
        const ALIGN: usize = 256 * 1024 * 1024; // 256 MiB
        let alloc_size = ((min_size + ALIGN - 1) / ALIGN) * ALIGN;

        let buf = allocate_device_buffer(py, alloc_size, device)?;

        // Register the new buffer with cuFile.
        let gds = get_gds_context().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to get GDS context for buffer registration: {e}"
            ))
        })?;
        // Safety: buf.data_ptr is a valid CUDA device pointer from
        // torch.empty() with alloc_size bytes.
        unsafe {
            gds.register_buffer(buf.data_ptr as *mut std::ffi::c_void, alloc_size)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "cuFileBufRegister failed: {e}"
                    ))
                })?;
        }

        let result = DeviceBuffer {
            tensor: buf.tensor.clone_ref(py),
            data_ptr: buf.data_ptr,
            size: alloc_size,
            device: device.to_string(),
        };

        *guard = Some(GdsBufferPool {
            tensor: buf.tensor,
            ptr: buf.data_ptr,
            size: alloc_size,
            device: device.to_string(),
        });

        Ok(result)
    }
}

/// Reads a safetensors file body into a **pre-registered** device buffer
/// using GDS.
///
/// Unlike [`gds_read_file_body`] and [`gds_read_file_body_cached`], this
/// function does **not** call `cuFileBufRegister` / `cuFileBufDeregister`.
/// The caller is responsible for ensuring the buffer is already registered
/// (e.g. via [`GdsBufferPool::acquire`]).  This saves ~18 ms per file.
///
/// The steps are:
///
/// 1. Opens the file with `O_DIRECT`.
/// 2. Registers the file handle with cuFile.
/// 3. Reads in 1 GiB chunks via `cuFileRead`, handling partial reads.
/// 4. Deregisters the file handle.
///
/// # Arguments
///
/// * `gds`         — An initialised GDS context.
/// * `path`        — Filesystem path to the safetensors file.
/// * `header_size` — Total header size in bytes (body starts at this offset).
/// * `body_size`   — Number of body bytes to read.
/// * `dev_ptr`     — Base address of the CUDA device buffer (must already be
///                   registered with cuFile).
/// * `dev_offset`  — Byte offset within the device buffer where data lands.
///
/// # Safety
///
/// The caller must ensure that:
/// - `dev_ptr` is a valid CUDA device pointer with at least
///   `dev_offset + body_size` bytes allocated.
/// - The buffer at `dev_ptr` has been registered with cuFile via
///   `cuFileBufRegister`.
/// - The GDS context has been properly initialised.
#[cfg(feature = "gds")]
pub unsafe fn gds_read_file_body_pooled(
    gds: &GdsContext,
    path: &str,
    header_size: usize,
    body_size: usize,
    dev_ptr: *mut std::ffi::c_void,
    dev_offset: u64,
) -> Result<(), String> {
    use std::io;

    let max_block_size: u64 = 1 << 30; // 1 GiB

    // -- Open the file with O_DIRECT for GDS compatibility -------------------
    let c_path = std::ffi::CString::new(path).map_err(|e| format!("Invalid path '{path}': {e}"))?;
    // Safety: c_path is a valid NUL-terminated string.
    let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECT);
    if fd < 0 {
        return Err(format!(
            "Failed to open '{path}' with O_DIRECT: {}",
            io::Error::last_os_error()
        ));
    }

    // RAII guard so we always close the fd.
    struct FdGuard(i32);
    impl Drop for FdGuard {
        fn drop(&mut self) {
            // Safety: self.0 is a valid open fd.
            unsafe {
                libc::close(self.0);
            }
        }
    }
    let _fd_guard = FdGuard(fd);

    // -- Register the file handle with cuFile --------------------------------
    let mut descr = CUfileDescr::from_raw_fd(fd);
    let fh = gds.register_handle(&mut descr)?;

    // -- Read in max_block_size chunks, handling partial reads ----------------
    // NOTE: We do NOT register/deregister the buffer here — the caller
    // is responsible for that (the pool keeps it registered).
    let file_offset_base = header_size as u64;
    let mut remaining = body_size as u64;
    let mut cursor: u64 = 0;

    let result: Result<(), String> = (|| {
        while remaining > 0 {
            let chunk = remaining.min(max_block_size);
            let mut chunk_remaining = chunk;

            while chunk_remaining > 0 {
                let bytes_read = gds.read(
                    fh,
                    dev_ptr,
                    chunk_remaining as usize,
                    (file_offset_base + cursor) as i64,
                    (dev_offset + cursor) as i64,
                )?;

                if bytes_read == 0 {
                    return Err(format!(
                        "cuFileRead returned 0 bytes at file_offset={}, expected {chunk_remaining}",
                        file_offset_base + cursor
                    ));
                }

                let n = bytes_read as u64;
                cursor += n;
                remaining -= n;
                chunk_remaining -= n;
            }
        }
        Ok(())
    })();

    // -- Cleanup: deregister file handle only (buffer stays registered) -------
    gds.deregister_handle(fh);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_alignment_fixups_empty() {
        let tensors: Vec<(String, usize, usize, usize)> = vec![];
        let fixups = compute_alignment_fixups(&tensors, 8);
        assert!(fixups.is_empty());
    }

    #[test]
    fn test_compute_alignment_fixups_aligned() {
        // All tensors are already aligned to 8 bytes
        let tensors = vec![
            ("a".to_string(), 4, 0, 1024),
            ("b".to_string(), 4, 1024, 2048),
            ("c".to_string(), 8, 2048, 4096),
        ];
        let fixups = compute_alignment_fixups(&tensors, 8);
        assert!(fixups.is_empty());
    }

    #[test]
    fn test_compute_alignment_fixups_misaligned() {
        // Tensor starts at offset 7 with 4-byte elements, required align = 8
        let tensors = vec![("t".to_string(), 4, 7, 1031)];
        let fixups = compute_alignment_fixups(&tensors, 8);
        assert_eq!(fixups.len(), 1);
        let (src, dst, len) = fixups[0];
        assert_eq!(src, 7);
        assert_eq!(dst, 8); // next 8-byte boundary
        assert_eq!(len, 1024);
    }
}
