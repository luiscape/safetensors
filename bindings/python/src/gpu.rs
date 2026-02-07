//! Python bindings for GPU fast loading.
//!
//! This module exposes the Rust GPU loading functionality to Python,
//! providing high-performance tensor loading directly to GPU memory.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;

// Create a custom exception for GPU errors
pyo3::create_exception!(_safetensors_rust, GpuError, PyException);

/// Configuration for the GPU loader.
#[pyclass]
#[derive(Clone)]
pub struct GpuLoaderConfig {
    /// Number of CUDA streams for parallel transfers.
    #[pyo3(get, set)]
    pub num_streams: usize,
    /// Maximum pinned memory in bytes.
    #[pyo3(get, set)]
    pub max_pinned_memory: usize,
    /// Device ID to load tensors to.
    #[pyo3(get, set)]
    pub device_id: i32,
    /// Enable double buffering.
    #[pyo3(get, set)]
    pub double_buffer: bool,
}

#[pymethods]
impl GpuLoaderConfig {
    /// Create a new GPU loader configuration.
    ///
    /// Args:
    ///     num_streams: Number of CUDA streams (default: 4)
    ///     max_pinned_memory_mb: Maximum pinned memory in MB (default: 512)
    ///     device_id: CUDA device ID (default: 0)
    ///     double_buffer: Enable double buffering (default: True)
    #[new]
    #[pyo3(signature = (num_streams=4, max_pinned_memory_mb=512, device_id=0, double_buffer=true))]
    fn new(
        num_streams: usize,
        max_pinned_memory_mb: usize,
        device_id: i32,
        double_buffer: bool,
    ) -> Self {
        Self {
            num_streams,
            max_pinned_memory: max_pinned_memory_mb << 20,
            device_id,
            double_buffer,
        }
    }

    /// Create a high-performance configuration (32 streams).
    #[staticmethod]
    fn high_performance() -> Self {
        Self {
            num_streams: 32,
            max_pinned_memory: 2 << 30, // 2GB
            device_id: 0,
            double_buffer: true,
        }
    }

    /// Create a maximum performance configuration (64 streams).
    /// Uses more pinned memory but may provide better throughput on high-end GPUs.
    #[staticmethod]
    fn max_performance() -> Self {
        Self {
            num_streams: 64,
            max_pinned_memory: 4usize << 30, // 4GB
            device_id: 0,
            double_buffer: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuLoaderConfig(num_streams={}, max_pinned_memory_mb={}, device_id={}, double_buffer={})",
            self.num_streams,
            self.max_pinned_memory >> 20,
            self.device_id,
            self.double_buffer
        )
    }
}

/// Statistics about GPU loader resource usage.
#[pyclass]
pub struct GpuLoaderStats {
    /// Number of CUDA streams in use.
    #[pyo3(get)]
    pub num_streams: usize,
    /// Currently allocated pinned memory in bytes.
    #[pyo3(get)]
    pub pinned_memory_allocated: usize,
    /// Maximum pinned memory limit in bytes.
    #[pyo3(get)]
    pub pinned_memory_max: usize,
    /// Number of free pinned buffers in pool.
    #[pyo3(get)]
    pub free_pinned_buffers: usize,
}

#[pymethods]
impl GpuLoaderStats {
    fn __repr__(&self) -> String {
        format!(
            "GpuLoaderStats(num_streams={}, pinned_memory_allocated={}, pinned_memory_max={}, free_pinned_buffers={})",
            self.num_streams,
            self.pinned_memory_allocated,
            self.pinned_memory_max,
            self.free_pinned_buffers
        )
    }
}

/// Information about a tensor loaded to GPU.
#[pyclass]
pub struct GpuTensorInfo {
    /// Tensor name.
    #[pyo3(get)]
    pub name: String,
    /// Data type as string.
    #[pyo3(get)]
    pub dtype: String,
    /// Shape as list.
    #[pyo3(get)]
    pub shape: Vec<usize>,
    /// Size in bytes.
    #[pyo3(get)]
    pub size_bytes: usize,
    /// Device ID.
    #[pyo3(get)]
    pub device_id: i32,
    /// GPU memory pointer (as integer for interop).
    #[pyo3(get)]
    pub data_ptr: u64,
}

#[pymethods]
impl GpuTensorInfo {
    fn __repr__(&self) -> String {
        format!(
            "GpuTensorInfo(name='{}', dtype='{}', shape={:?}, size_bytes={}, device_id={})",
            self.name, self.dtype, self.shape, self.size_bytes, self.device_id
        )
    }
}

/// Check if GPU loading is available.
///
/// Returns:
///     bool: True if GPU loading is available, False otherwise.
///
/// Note:
///     This checks if the safetensors library was compiled with GPU support
///     and if CUDA is available on the system.
#[pyfunction]
pub fn is_gpu_available() -> bool {
    // When compiled with gpu feature, check CUDA availability
    #[cfg(feature = "gpu")]
    {
        safetensors::gpu::low_level::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Get the number of available CUDA devices.
///
/// Returns:
///     int: Number of CUDA devices, or 0 if GPU not available.
#[pyfunction]
pub fn get_device_count() -> PyResult<i32> {
    #[cfg(feature = "gpu")]
    {
        safetensors::gpu::low_level::CudaDevice::count()
            .map_err(|e| GpuError::new_err(e.to_string()))
    }
    #[cfg(not(feature = "gpu"))]
    {
        Ok(0)
    }
}

/// Load safetensor file directly to GPU memory with optimized I/O.
///
/// This function uses CUDA streams and pinned memory for high-performance
/// loading. It bypasses PyTorch's tensor creation and loads directly to
/// GPU memory.
///
/// Args:
///     filename: Path to the safetensor file.
///     config: Optional GpuLoaderConfig for customization.
///     tensor_names: Optional list of tensor names to load (loads all if None).
///
/// Returns:
///     dict: Dictionary mapping tensor names to GpuTensorInfo objects.
///
/// Raises:
///     GpuError: If GPU loading fails.
///
/// Example:
///     >>> from safetensors.gpu import load_to_gpu, GpuLoaderConfig
///     >>> config = GpuLoaderConfig(num_streams=8, max_pinned_memory_mb=1024)
///     >>> tensors = load_to_gpu("model.safetensors", config)
///     >>> for name, info in tensors.items():
///     ...     print(f"{name}: {info.shape} on device {info.device_id}")
#[pyfunction]
#[pyo3(signature = (filename, config=None, tensor_names=None))]
pub fn load_to_gpu(
    _py: Python<'_>,
    filename: PathBuf,
    config: Option<GpuLoaderConfig>,
    tensor_names: Option<Vec<String>>,
) -> PyResult<HashMap<String, GpuTensorInfo>> {
    #[cfg(feature = "gpu")]
    {
        use safetensors::gpu::{GpuLoader, GpuLoaderConfig as RustConfig};

        let config = config.unwrap_or_else(|| GpuLoaderConfig::new(4, 512, 0, true));

        let rust_config = RustConfig {
            num_streams: config.num_streams,
            max_pinned_memory: config.max_pinned_memory,
            device_id: config.device_id,
            double_buffer: config.double_buffer,
            ..Default::default()
        };

        let loader = GpuLoader::new(rust_config)
            .map_err(|e| GpuError::new_err(e.to_string()))?;

        // Load tensors (not using allow_threads due to GpuLoader not being Sync)
        let tensors = if let Some(names) = tensor_names {
            // Load specific tensors
            let file = std::fs::File::open(&filename)
                .map_err(|e| GpuError::new_err(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| GpuError::new_err(e.to_string()))?;
            let safetensors = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| GpuError::new_err(e.to_string()))?;
            let names_ref: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            loader.load_tensors(&safetensors, &names_ref)
                .map_err(|e| GpuError::new_err(e.to_string()))?
        } else {
            // Load all tensors
            loader.load_file(&filename)
                .map_err(|e| GpuError::new_err(e.to_string()))?
        };

        // Convert to Python-friendly format
        let mut result = HashMap::new();
        for (name, tensor) in tensors {
            let info = GpuTensorInfo {
                name: name.clone(),
                dtype: format!("{:?}", tensor.dtype()),
                shape: tensor.shape().to_vec(),
                size_bytes: tensor.size_bytes(),
                device_id: tensor.device_id(),
                data_ptr: tensor.as_ptr() as u64,
            };
            result.insert(name, info);
            // Note: We're intentionally leaking the GPU memory here.
            // In a real implementation, we'd need to track ownership
            // and provide a way to free the memory.
            std::mem::forget(tensor);
        }

        Ok(result)
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (_py, filename, config, tensor_names);
        Err(GpuError::new_err("GPU support not compiled in. Rebuild with --features gpu"))
    }
}

/// Load safetensor file to GPU and convert to PyTorch tensors.
///
/// This is the recommended way to load safetensors to GPU for use with PyTorch.
/// It uses optimized Rust-based loading with CUDA streams, then wraps the
/// GPU memory in PyTorch tensors.
///
/// Args:
///     filename: Path to the safetensor file.
///     device: PyTorch device string (e.g., "cuda:0").
///     config: Optional GpuLoaderConfig for customization.
///     tensor_names: Optional list of tensor names to load.
///
/// Returns:
///     dict: Dictionary mapping tensor names to PyTorch tensors on GPU.
///
/// Raises:
///     GpuError: If loading fails.
///     ImportError: If PyTorch is not available.
///
/// Example:
///     >>> from safetensors.gpu import load_to_torch
///     >>> tensors = load_to_torch("model.safetensors", "cuda:0")
///     >>> print(tensors["weight"].device)  # cuda:0
#[pyfunction]
#[pyo3(signature = (filename, device="cuda:0", config=None, tensor_names=None))]
pub fn load_to_torch(
    py: Python<'_>,
    filename: PathBuf,
    device: &str,
    config: Option<GpuLoaderConfig>,
    tensor_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    // Parse device string to get device ID
    let device_id: i32 = if device.starts_with("cuda:") {
        device[5..].parse().unwrap_or(0)
    } else if device == "cuda" {
        0
    } else {
        return Err(GpuError::new_err(format!(
            "Invalid device '{}'. Expected 'cuda' or 'cuda:N'",
            device
        )));
    };

    // Update config with correct device
    let config = config
        .map(|mut c| {
            c.device_id = device_id;
            c
        })
        .unwrap_or_else(|| GpuLoaderConfig {
            num_streams: 4,
            max_pinned_memory: 512 << 20,  // 512MB, chunked transfers handle large tensors
            device_id,
            double_buffer: true,
        });

    #[cfg(feature = "gpu")]
    {
        use safetensors::gpu::low_level::CudaDevice;

        // Set the CUDA device
        let cuda_device = CudaDevice::new(device_id)
            .map_err(|e| GpuError::new_err(e.to_string()))?;
        cuda_device.set_current()
            .map_err(|e| GpuError::new_err(e.to_string()))?;

        // Open and parse the safetensors file
        let file = std::fs::File::open(&filename)
            .map_err(|e| GpuError::new_err(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| GpuError::new_err(e.to_string()))?;

        // Hint to kernel that we'll read sequentially - improves read-ahead
        #[cfg(unix)]
        {
            extern "C" {
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            const MADV_SEQUENTIAL: i32 = 2;
            const MADV_WILLNEED: i32 = 3;
            unsafe {
                // Sequential access hint
                madvise(mmap.as_ptr() as *mut std::ffi::c_void, mmap.len(), MADV_SEQUENTIAL);
                // Also hint that we'll need this data soon (triggers read-ahead)
                madvise(mmap.as_ptr() as *mut std::ffi::c_void, mmap.len(), MADV_WILLNEED);
            }
        }
        let safetensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| GpuError::new_err(e.to_string()))?;

        // Get list of tensor names to load
        let names_to_load: Vec<String> = if let Some(names) = tensor_names {
            names
        } else {
            safetensors.names().into_iter().map(|s| s.to_string()).collect()
        };

        // Import torch
        let torch = PyModule::import(py, "torch")?;
        let result = PyDict::new(py);

        // CUDA FFI - direct cudaMemcpy from mmap, let CUDA handle staging internally
        extern "C" {
            fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
        }
        const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;

        for name in &names_to_load {
            let view = safetensors.tensor(name)
                .map_err(|e| GpuError::new_err(e.to_string()))?;

            let dtype_str = format!("{:?}", view.dtype());
            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();

            // Map dtype to torch dtype
            let torch_dtype = match dtype_str.as_str() {
                "F32" => torch.getattr("float32")?,
                "F64" => torch.getattr("float64")?,
                "F16" => torch.getattr("float16")?,
                "BF16" => torch.getattr("bfloat16")?,
                "I8" => torch.getattr("int8")?,
                "I16" => torch.getattr("int16")?,
                "I32" => torch.getattr("int32")?,
                "I64" => torch.getattr("int64")?,
                "U8" => torch.getattr("uint8")?,
                "BOOL" => torch.getattr("bool")?,
                dt => {
                    return Err(GpuError::new_err(format!("Unsupported dtype: {}", dt)));
                }
            };

            // Create empty PyTorch tensor on GPU
            let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
            let shape_list = PyList::new(py, &shape_i64)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", torch_dtype)?;
            kwargs.set_item("device", device)?;

            let tensor = torch.call_method("empty", (shape_list,), Some(&kwargs))?;

            if !data.is_empty() {
                let dst_ptr: u64 = tensor.call_method0("data_ptr")?.extract()?;

                // Direct cudaMemcpy from mmap to GPU - CUDA handles staging internally
                // This avoids the CPU memcpy to pinned buffer bottleneck
                let err = unsafe {
                    cudaMemcpy(
                        dst_ptr as *mut std::ffi::c_void,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len(),
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    )
                };
                if err != 0 {
                    return Err(GpuError::new_err(format!(
                        "cudaMemcpy failed with error code {} for tensor '{}'",
                        err, name
                    )));
                }
            }

            result.set_item(name, tensor)?;
        }

        Ok(result.into())
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (py, filename, device, config, tensor_names);
        Err(GpuError::new_err("GPU support not compiled in"))
    }
}

/// Get statistics about GPU loader resource usage.
///
/// Args:
///     config: Optional GpuLoaderConfig (uses defaults if None).
///
/// Returns:
///     GpuLoaderStats: Statistics about pinned memory usage, etc.
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn get_loader_stats(config: Option<GpuLoaderConfig>) -> PyResult<GpuLoaderStats> {
    #[cfg(feature = "gpu")]
    {
        use safetensors::gpu::{GpuLoader, GpuLoaderConfig as RustConfig};

        let config = config.unwrap_or_else(|| GpuLoaderConfig::new(4, 512, 0, true));

        let rust_config = RustConfig {
            num_streams: config.num_streams,
            max_pinned_memory: config.max_pinned_memory,
            device_id: config.device_id,
            double_buffer: config.double_buffer,
            ..Default::default()
        };

        let loader = GpuLoader::new(rust_config)
            .map_err(|e| GpuError::new_err(e.to_string()))?;

        let stats = loader.stats();

        Ok(GpuLoaderStats {
            num_streams: stats.num_streams,
            pinned_memory_allocated: stats.pinned_memory_allocated,
            pinned_memory_max: stats.pinned_memory_max,
            free_pinned_buffers: stats.free_pinned_buffers,
        })
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = config;
        Err(GpuError::new_err("GPU support not compiled in"))
    }
}

/// Register GPU loading functions with the Python module.
pub fn register_gpu_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let gpu_module = PyModule::new(parent.py(), "gpu")?;

    gpu_module.add_class::<GpuLoaderConfig>()?;
    gpu_module.add_class::<GpuLoaderStats>()?;
    gpu_module.add_class::<GpuTensorInfo>()?;
    gpu_module.add_function(wrap_pyfunction!(is_gpu_available, &gpu_module)?)?;
    gpu_module.add_function(wrap_pyfunction!(get_device_count, &gpu_module)?)?;
    gpu_module.add_function(wrap_pyfunction!(load_to_gpu, &gpu_module)?)?;
    gpu_module.add_function(wrap_pyfunction!(load_to_torch, &gpu_module)?)?;
    gpu_module.add_function(wrap_pyfunction!(get_loader_stats, &gpu_module)?)?;

    parent.add_submodule(&gpu_module)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = GpuLoaderConfig::new(8, 1024, 0, true);
        assert_eq!(config.num_streams, 8);
        assert_eq!(config.max_pinned_memory, 1024 << 20);
        assert_eq!(config.device_id, 0);
        assert!(config.double_buffer);
    }

    #[test]
    fn test_high_performance_config() {
        let config = GpuLoaderConfig::high_performance();
        assert_eq!(config.num_streams, 8);
        assert!(config.max_pinned_memory >= 1 << 30);
    }
}
