"""
High-performance GPU loading for safetensors.

This module provides optimized GPU loading using CUDA streams and pinned memory
for maximum throughput when loading model weights to GPU.

Example usage:
    >>> from safetensors.gpu import load_file, GpuLoaderConfig
    >>>
    >>> # Simple loading with defaults
    >>> tensors = load_file("model.safetensors", device="cuda:0")
    >>>
    >>> # Custom configuration for high performance
    >>> config = GpuLoaderConfig(
    ...     num_streams=8,
    ...     max_pinned_memory_mb=1024,
    ...     double_buffer=True
    ... )
    >>> tensors = load_file("model.safetensors", device="cuda:0", config=config)

Note:
    This module requires safetensors to be compiled with GPU support.
    If GPU support is not available, functions will raise ImportError.
"""

from typing import Dict, List, Optional, Any, Union
import os

__all__ = [
    "load_file",
    "load_file_to_torch",
    "is_available",
    "get_device_count",
    "GpuLoaderConfig",
    "GpuLoaderStats",
    "GpuTensorInfo",
]


# Try to import the Rust GPU module
_gpu_available = False
_gpu_module = None

try:
    from safetensors._safetensors_rust import gpu as _gpu_module
    _gpu_available = True
except ImportError:
    _gpu_available = False


class GpuLoaderConfig:
    """
    Configuration for the GPU tensor loader.

    This class allows fine-tuning the GPU loading process for optimal
    performance on different hardware configurations.

    Attributes:
        num_streams: Number of CUDA streams for parallel transfers.
            More streams can improve throughput but use more resources.
            Recommended: 4-8 for most GPUs.
        max_pinned_memory_mb: Maximum pinned (page-locked) memory in megabytes.
            Pinned memory enables faster DMA transfers.
            Recommended: 256-1024 MB depending on available RAM.
        device_id: CUDA device ID to load tensors to (0-indexed).
        double_buffer: Enable double buffering for overlapped transfers.
            This can significantly improve throughput.

    Example:
        >>> config = GpuLoaderConfig(
        ...     num_streams=8,
        ...     max_pinned_memory_mb=1024,
        ...     device_id=0,
        ...     double_buffer=True
        ... )
    """

    def __init__(
        self,
        num_streams: int = 4,
        max_pinned_memory_mb: int = 512,
        device_id: int = 0,
        double_buffer: bool = True,
    ):
        """
        Initialize GPU loader configuration.

        Args:
            num_streams: Number of CUDA streams (default: 4).
            max_pinned_memory_mb: Maximum pinned memory in MB (default: 512).
            device_id: CUDA device ID (default: 0).
            double_buffer: Enable double buffering (default: True).
        """
        self.num_streams = num_streams
        self.max_pinned_memory_mb = max_pinned_memory_mb
        self.device_id = device_id
        self.double_buffer = double_buffer

    @classmethod
    def high_performance(cls) -> "GpuLoaderConfig":
        """
        Create a high-performance configuration.

        This configuration uses more resources for maximum throughput:
        - 8 CUDA streams
        - 1GB pinned memory
        - Double buffering enabled

        Returns:
            GpuLoaderConfig: High-performance configuration.
        """
        return cls(
            num_streams=8,
            max_pinned_memory_mb=1024,
            device_id=0,
            double_buffer=True,
        )

    def __repr__(self) -> str:
        return (
            f"GpuLoaderConfig(num_streams={self.num_streams}, "
            f"max_pinned_memory_mb={self.max_pinned_memory_mb}, "
            f"device_id={self.device_id}, "
            f"double_buffer={self.double_buffer})"
        )


class GpuLoaderStats:
    """
    Statistics about GPU loader resource usage.

    Attributes:
        num_streams: Number of active CUDA streams.
        pinned_memory_allocated: Currently allocated pinned memory in bytes.
        pinned_memory_max: Maximum pinned memory limit in bytes.
        free_pinned_buffers: Number of free buffers in the pool.
    """

    def __init__(
        self,
        num_streams: int,
        pinned_memory_allocated: int,
        pinned_memory_max: int,
        free_pinned_buffers: int,
    ):
        self.num_streams = num_streams
        self.pinned_memory_allocated = pinned_memory_allocated
        self.pinned_memory_max = pinned_memory_max
        self.free_pinned_buffers = free_pinned_buffers

    def __repr__(self) -> str:
        return (
            f"GpuLoaderStats(num_streams={self.num_streams}, "
            f"pinned_memory_allocated={self.pinned_memory_allocated}, "
            f"pinned_memory_max={self.pinned_memory_max}, "
            f"free_pinned_buffers={self.free_pinned_buffers})"
        )


class GpuTensorInfo:
    """
    Information about a tensor loaded to GPU memory.

    This class provides metadata about tensors loaded via the GPU loader.
    It can be used for debugging or for advanced use cases where you need
    to access the raw GPU memory pointer.

    Attributes:
        name: Tensor name.
        dtype: Data type as string (e.g., "F32", "F16").
        shape: Tensor shape as tuple.
        size_bytes: Size in bytes.
        device_id: CUDA device ID where tensor is stored.
        data_ptr: Raw GPU memory pointer (for advanced use).
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: tuple,
        size_bytes: int,
        device_id: int,
        data_ptr: int,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.size_bytes = size_bytes
        self.device_id = device_id
        self.data_ptr = data_ptr

    def __repr__(self) -> str:
        return (
            f"GpuTensorInfo(name='{self.name}', dtype='{self.dtype}', "
            f"shape={self.shape}, size_bytes={self.size_bytes}, "
            f"device_id={self.device_id})"
        )


def is_available() -> bool:
    """
    Check if GPU loading is available.

    Returns:
        bool: True if GPU loading is available, False otherwise.

    Note:
        GPU loading requires:
        1. safetensors compiled with GPU support
        2. CUDA runtime available
        3. At least one CUDA device

    Example:
        >>> from safetensors.gpu import is_available
        >>> if is_available():
        ...     print("GPU loading is available!")
    """
    if not _gpu_available:
        return False
    try:
        return _gpu_module.is_gpu_available()
    except Exception:
        return False


def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        int: Number of CUDA devices, or 0 if GPU not available.

    Raises:
        RuntimeError: If GPU support is not compiled in.

    Example:
        >>> from safetensors.gpu import get_device_count
        >>> print(f"Found {get_device_count()} CUDA devices")
    """
    if not _gpu_available:
        return 0
    return _gpu_module.get_device_count()


def load_file(
    filename: Union[str, os.PathLike],
    device: str = "cuda:0",
    config: Optional[GpuLoaderConfig] = None,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load safetensor file to GPU and return as PyTorch tensors.

    This is the main function for loading safetensors directly to GPU memory
    with optimized performance. It uses CUDA streams and pinned memory for
    maximum throughput.

    Args:
        filename: Path to the safetensor file.
        device: Target device string (e.g., "cuda:0", "cuda:1").
        config: Optional GpuLoaderConfig for customization.
        tensor_names: Optional list of tensor names to load.
            If None, all tensors are loaded.

    Returns:
        dict: Dictionary mapping tensor names to PyTorch tensors on GPU.

    Raises:
        ImportError: If GPU support is not available.
        FileNotFoundError: If the file doesn't exist.
        RuntimeError: If loading fails.

    Example:
        >>> from safetensors.gpu import load_file
        >>>
        >>> # Load all tensors
        >>> tensors = load_file("model.safetensors", device="cuda:0")
        >>>
        >>> # Load specific tensors
        >>> tensors = load_file(
        ...     "model.safetensors",
        ...     device="cuda:0",
        ...     tensor_names=["weight", "bias"]
        ... )
        >>>
        >>> # Use custom configuration
        >>> from safetensors.gpu import GpuLoaderConfig
        >>> config = GpuLoaderConfig.high_performance()
        >>> tensors = load_file("model.safetensors", "cuda:0", config=config)
    """
    if not _gpu_available:
        raise ImportError(
            "GPU loading is not available. "
            "Please rebuild safetensors with GPU support: "
            "pip install safetensors[gpu]"
        )

    # Convert path to string
    filename = str(filename)

    # Check file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Parse device string
    if device.startswith("cuda:"):
        device_id = int(device[5:])
    elif device == "cuda":
        device_id = 0
    else:
        raise ValueError(
            f"Invalid device '{device}'. Expected 'cuda' or 'cuda:N'"
        )

    # Create Rust config if provided
    rust_config = None
    if config is not None:
        rust_config = _gpu_module.GpuLoaderConfig(
            num_streams=config.num_streams,
            max_pinned_memory_mb=config.max_pinned_memory_mb,
            device_id=device_id,
            double_buffer=config.double_buffer,
        )
    else:
        rust_config = _gpu_module.GpuLoaderConfig(
            num_streams=4,
            max_pinned_memory_mb=512,
            device_id=device_id,
            double_buffer=True,
        )

    # Load using Rust GPU loader and convert to torch
    return _gpu_module.load_to_torch(
        filename,
        device,
        rust_config,
        tensor_names,
    )


def load_file_to_torch(
    filename: Union[str, os.PathLike],
    device: str = "cuda:0",
    config: Optional[GpuLoaderConfig] = None,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Alias for load_file() for compatibility.

    See load_file() for documentation.
    """
    return load_file(filename, device, config, tensor_names)


def get_loader_stats(config: Optional[GpuLoaderConfig] = None) -> GpuLoaderStats:
    """
    Get statistics about GPU loader resource usage.

    This is useful for debugging and monitoring memory usage.

    Args:
        config: Optional GpuLoaderConfig (uses defaults if None).

    Returns:
        GpuLoaderStats: Statistics about the loader.

    Example:
        >>> from safetensors.gpu import get_loader_stats
        >>> stats = get_loader_stats()
        >>> print(f"Pinned memory: {stats.pinned_memory_allocated / 1024**2:.1f} MB")
    """
    if not _gpu_available:
        raise ImportError("GPU support is not available")

    rust_config = None
    if config is not None:
        rust_config = _gpu_module.GpuLoaderConfig(
            num_streams=config.num_streams,
            max_pinned_memory_mb=config.max_pinned_memory_mb,
            device_id=config.device_id,
            double_buffer=config.double_buffer,
        )

    stats = _gpu_module.get_loader_stats(rust_config)
    return GpuLoaderStats(
        num_streams=stats.num_streams,
        pinned_memory_allocated=stats.pinned_memory_allocated,
        pinned_memory_max=stats.pinned_memory_max,
        free_pinned_buffers=stats.free_pinned_buffers,
    )


# Convenience function for PyTorch users
def save_file(
    tensors: Dict[str, Any],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save tensors to a safetensor file.

    This is a convenience wrapper around safetensors.torch.save_file().

    Args:
        tensors: Dictionary of tensor name to PyTorch tensor.
        filename: Output file path.
        metadata: Optional metadata dictionary.

    Example:
        >>> import torch
        >>> from safetensors.gpu import save_file
        >>> tensors = {"weight": torch.randn(100, 100, device="cuda")}
        >>> save_file(tensors, "model.safetensors")
    """
    from safetensors.torch import save_file as _save_file
    _save_file(tensors, str(filename), metadata)
