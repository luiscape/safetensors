# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    _safe_open_handle,
    deserialize,
    safe_open,
    serialize,
    serialize_file,
)

# Fast loading support
try:
    from .fast import fast_load_file, fast_open  # noqa: F401
except ImportError:
    pass
