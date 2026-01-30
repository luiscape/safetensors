"""Type stubs for safetensors.gpu module."""

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

class GpuLoaderConfig:
    """Configuration for the GPU tensor loader."""

    num_streams: int
    max_pinned_memory_mb: int
    device_id: int
    double_buffer: bool

    def __init__(
        self,
        num_streams: int = 4,
        max_pinned_memory_mb: int = 512,
        device_id: int = 0,
        double_buffer: bool = True,
    ) -> None: ...
    @classmethod
    def high_performance(cls) -> GpuLoaderConfig: ...
    def __repr__(self) -> str: ...

class GpuLoaderStats:
    """Statistics about GPU loader resource usage."""

    num_streams: int
    pinned_memory_allocated: int
    pinned_memory_max: int
    free_pinned_buffers: int

    def __init__(
        self,
        num_streams: int,
        pinned_memory_allocated: int,
        pinned_memory_max: int,
        free_pinned_buffers: int,
    ) -> None: ...
    def __repr__(self) -> str: ...

class GpuTensorInfo:
    """Information about a tensor loaded to GPU memory."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    size_bytes: int
    device_id: int
    data_ptr: int

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        size_bytes: int,
        device_id: int,
        data_ptr: int,
    ) -> None: ...
    def __repr__(self) -> str: ...

def is_available() -> bool:
    """
    Check if GPU loading is available.

    Returns:
        True if GPU loading is available, False otherwise.
    """
    ...

def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices, or 0 if GPU not available.
    """
    ...

def load_file(
    filename: Union[str, os.PathLike[str]],
    device: str = "cuda:0",
    config: Optional[GpuLoaderConfig] = None,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load safetensor file to GPU and return as PyTorch tensors.

    Args:
        filename: Path to the safetensor file.
        device: Target device string (e.g., "cuda:0", "cuda:1").
        config: Optional GpuLoaderConfig for customization.
        tensor_names: Optional list of tensor names to load.

    Returns:
        Dictionary mapping tensor names to PyTorch tensors on GPU.

    Raises:
        ImportError: If GPU support is not available.
        FileNotFoundError: If the file doesn't exist.
        RuntimeError: If loading fails.
    """
    ...

def load_file_to_torch(
    filename: Union[str, os.PathLike[str]],
    device: str = "cuda:0",
    config: Optional[GpuLoaderConfig] = None,
    tensor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Alias for load_file() for compatibility.

    See load_file() for documentation.
    """
    ...

def get_loader_stats(config: Optional[GpuLoaderConfig] = None) -> GpuLoaderStats:
    """
    Get statistics about GPU loader resource usage.

    Args:
        config: Optional GpuLoaderConfig (uses defaults if None).

    Returns:
        Statistics about the loader.
    """
    ...

def save_file(
    tensors: Dict[str, Any],
    filename: Union[str, os.PathLike[str]],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save tensors to a safetensor file.

    Args:
        tensors: Dictionary of tensor name to PyTorch tensor.
        filename: Output file path.
        metadata: Optional metadata dictionary.
    """
    ...
