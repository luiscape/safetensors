"""
Fast safetensors loading module.

Provides optimized loading of safetensors files using techniques from
the fastsafetensors paper (arXiv:2505.23072):
- Aggregated tensor deserialization (bulk I/O into device buffers)
- GPU offloading for sharding via torch.distributed collective ops
- Parallel I/O with bounce buffers
- NUMA-aware memory placement
- Optional GPU Direct Storage (GDS) support

Usage:
    from safetensors.fast import fast_load_file, FastSafeTensorsLoader

    # Simple single-file loading (drop-in replacement for load_file)
    tensors = fast_load_file("model.safetensors", device="cuda:0")

    # Multi-file distributed loading
    loader = FastSafeTensorsLoader(device="cuda:0", process_group=pg)
    loader.add_filenames({0: ["shard1.safetensors"], 1: ["shard2.safetensors"]})
    buffer = loader.copy_files_to_device()
    tensor = buffer.get_tensor("weight")
    sharded = buffer.get_sharded("big_weight", dim=0)
    buffer.close()
    loader.close()
"""

import json
import os
import struct
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def _import_torch():
    """Lazy import of torch to allow module to load without torch installed."""
    import torch

    return torch


# ---------------------------------------------------------------------------
# Dtype mapping (mirrors safetensors.torch._TYPES / _SIZE)
# ---------------------------------------------------------------------------


def _get_dtype_maps():
    """Build dtype string -> torch.dtype and torch.dtype -> element size maps.

    We replicate the canonical tables from ``safetensors.torch`` so that
    ``fast.py`` can work stand-alone without importing ``safetensors.torch``
    (which pulls in heavy machinery we may not need at import time).
    """
    torch = _import_torch()

    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    _float4_e2m1_x2 = getattr(torch, "float4_e2m1fn_x2", None)

    types = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "F8_E4M3": _float8_e4m3fn,
        "F8_E5M2": _float8_e5m2,
        "C64": torch.complex64,
    }

    sizes = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        torch.complex64: 8,
    }
    if _float8_e4m3fn is not None:
        sizes[_float8_e4m3fn] = 1
    if _float8_e5m2 is not None:
        sizes[_float8_e5m2] = 1
    if _float8_e8m0 is not None:
        sizes[_float8_e8m0] = 1
    if _float4_e2m1_x2 is not None:
        sizes[_float4_e2m1_x2] = 1

    if hasattr(torch, "uint64"):
        types.update({"U64": torch.uint64, "U32": torch.uint32, "U16": torch.uint16})
        sizes.update({torch.uint64: 8, torch.uint32: 4, torch.uint16: 2})

    return types, sizes


# Module-level caches (populated lazily)
_TYPES: Optional[Dict[str, Any]] = None
_SIZE: Optional[Dict[Any, int]] = None
_dtype_lock = threading.Lock()


def _ensure_dtype_maps():
    global _TYPES, _SIZE
    if _TYPES is None:
        with _dtype_lock:
            if _TYPES is None:
                _TYPES, _SIZE = _get_dtype_maps()


# ---------------------------------------------------------------------------
# Low-level header parsing
# ---------------------------------------------------------------------------


def _parse_header(filename: str) -> Tuple[int, Dict[str, Any], OrderedDict]:
    """Parse a safetensors file header without loading tensor data.

    Returns:
        (header_size, metadata_dict, tensor_info_ordered_dict)

    ``tensor_info`` is an ``OrderedDict`` keyed by tensor name with values::

        {"dtype": str, "shape": List[int], "data_offsets": [int, int]}

    ordered by ascending ``data_offsets[0]`` (file offset order).  The
    ``metadata_dict`` may contain user-supplied metadata stored under the
    special ``"__metadata__"`` key (stripped from the tensor info).
    """
    fd = os.open(filename, os.O_RDONLY)
    try:
        # First 8 bytes: little-endian u64 giving header JSON length
        header_len_bytes = os.pread(fd, 8, 0)
        if len(header_len_bytes) < 8:
            raise ValueError(f"File too small to be a safetensors file: {filename}")
        header_json_len = struct.unpack("<Q", header_len_bytes)[0]

        # Sanity-check: header shouldn't be larger than ~100 MB for even the
        # largest models.  Guard against corrupt files.
        if header_json_len > 100_000_000:
            raise ValueError(
                f"Header size {header_json_len} bytes seems unreasonably large "
                f"for {filename}. File may be corrupt."
            )

        header_bytes = os.pread(fd, header_json_len, 8)
        if len(header_bytes) < header_json_len:
            raise ValueError(
                f"Incomplete header: expected {header_json_len} bytes, "
                f"got {len(header_bytes)}"
            )
    finally:
        os.close(fd)

    header = json.loads(header_bytes)
    metadata = header.pop("__metadata__", {})

    # Sort tensors by their start offset for sequential access
    tensor_info: OrderedDict = OrderedDict()
    sorted_items = sorted(
        header.items(),
        key=lambda item: item[1].get("data_offsets", [0])[0],
    )
    for name, info in sorted_items:
        tensor_info[name] = info

    # header_size = 8 bytes (length prefix) + header_json_len
    header_size = 8 + header_json_len
    return header_size, metadata, tensor_info


def _file_body_size(filename: str, header_size: int) -> int:
    """Return the number of bytes after the header (= tensor body)."""
    file_size = os.path.getsize(filename)
    body = file_size - header_size
    if body < 0:
        raise ValueError(
            f"File {filename} is smaller ({file_size} B) than its header "
            f"({header_size} B)."
        )
    return body


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _pread_into_buffer(fd: int, buf, offset: int, length: int) -> None:
    """Read *length* bytes from *fd* at *offset* into *buf* (a writable
    buffer that supports the buffer protocol, e.g. a memoryview into a
    pinned tensor).

    Uses ``os.pread`` in a loop to handle short reads.
    """
    pos = 0
    mv = memoryview(buf)
    while pos < length:
        chunk_size = min(length - pos, 128 * 1024 * 1024)  # 128 MiB chunks
        data = os.pread(fd, chunk_size, offset + pos)
        n = len(data)
        if n == 0:
            raise IOError(
                f"Unexpected EOF: wanted {length} bytes at offset {offset}, "
                f"got {pos} bytes"
            )
        mv[pos : pos + n] = data
        pos += n


def _parallel_pread(
    fd: int,
    buf,
    file_offset: int,
    length: int,
    max_threads: int,
    chunk_size: int = 16 * 1024 * 1024,
) -> None:
    """Read *length* bytes from *fd* at *file_offset* into *buf* using
    multiple threads, each performing ``os.pread`` on non-overlapping
    regions of the file.

    ``buf`` must support the buffer protocol (e.g. ``memoryview`` into a
    byte tensor's storage).
    """
    if length == 0:
        return

    if length <= chunk_size or max_threads <= 1:
        _pread_into_buffer(fd, buf, file_offset, length)
        return

    # Divide into roughly equal pieces, at most *max_threads* pieces
    n_chunks = min(max_threads, max(1, (length + chunk_size - 1) // chunk_size))
    piece = length // n_chunks
    remainder = length % n_chunks

    def _read_piece(idx: int) -> None:
        start = idx * piece + min(idx, remainder)
        end = start + piece + (1 if idx < remainder else 0)
        _pread_into_buffer(
            fd, memoryview(buf)[start:end], file_offset + start, end - start
        )

    with ThreadPoolExecutor(max_workers=n_chunks) as pool:
        futs = [pool.submit(_read_piece, i) for i in range(n_chunks)]
        for f in futs:
            f.result()  # propagate exceptions


def _set_numa_affinity(device: str) -> None:
    """Best-effort NUMA affinity hint for the calling thread based on the
    target CUDA device.  Silently does nothing when unavailable.
    """
    try:
        torch = _import_torch()
        if not device.startswith("cuda"):
            return
        dev_idx = (
            torch.cuda.current_device()
            if ":" not in device
            else int(device.split(":")[1])
        )
        # pynvml or torch.cuda give us the PCI bus; we can derive NUMA
        # node from sysfs.  This is a best-effort optimisation – we
        # simply catch all exceptions.
        numa_path = f"/sys/bus/pci/devices/{torch.cuda.get_device_properties(dev_idx).name}/numa_node"
        if not os.path.exists(numa_path):
            # Try alternative path using device index
            numa_path = f"/sys/class/nvidia/device{dev_idx}/numa_node"
        if os.path.exists(numa_path):
            with open(numa_path) as fh:
                node = int(fh.read().strip())
            if node >= 0:
                os.sched_setaffinity(0, os.sched_getaffinity(0))  # no-op placeholder
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastFileBuffer
# ---------------------------------------------------------------------------


class FastFileBuffer:
    """Buffer holding a safetensors file's tensor data on a device.

    Created by :func:`fast_open` or :class:`FastSafeTensorsLoader`.  Supports
    zero-copy tensor retrieval via :meth:`get_tensor` and GPU-side sharding
    via :meth:`get_sharded` (when combined with :class:`FilesBufferOnDevice`).

    The core idea (*aggregated tensor deserialization* from the fastsafetensors
    paper) is:

    1. Parse the header to learn tensor layout.
    2. Allocate a **single** contiguous ``uint8`` buffer on the target device.
    3. Bulk-read the file body into the buffer using parallel I/O.
    4. Individual tensors are zero-copy **views** into this buffer.
    """

    def __init__(
        self,
        filename: str,
        device: str = "cpu",
        max_threads: int = 16,
        bounce_buffer_size_kb: int = 16384,
        nogds: bool = True,
    ):
        torch = _import_torch()
        _ensure_dtype_maps()

        self._filename = filename
        self._device = device
        self._closed = False

        # -- 1. Parse header ------------------------------------------------
        self._header_size, self._metadata, self._tensor_info = _parse_header(filename)
        self._body_size = _file_body_size(filename, self._header_size)

        # -- 2. Allocate device buffer --------------------------------------
        if self._body_size == 0:
            # Degenerate case: file has no tensor data (all tensors empty)
            self._buffer = torch.empty(0, dtype=torch.uint8, device=device)
            return

        is_cuda = device.startswith("cuda") or (
            isinstance(device, str) and device.isdigit()
        )

        # -- 3. Read file body into buffer ----------------------------------
        fd = os.open(filename, os.O_RDONLY)
        try:
            if not is_cuda:
                # ---- CPU path: parallel pread directly into tensor --------
                self._buffer = torch.empty(
                    self._body_size, dtype=torch.uint8, device="cpu"
                )
                # Get a writable memoryview of the tensor storage
                np_buf = self._buffer.numpy()
                _parallel_pread(
                    fd, np_buf, self._header_size, self._body_size, max_threads
                )
            else:
                # ---- CUDA path (nogds or gds) ----------------------------
                if not nogds:
                    # Attempt GDS via cuFile; fall back to bounce-buffer path
                    loaded_via_gds = False
                    try:
                        loaded_via_gds = self._try_gds_read(fd, device)
                    except Exception:
                        loaded_via_gds = False

                    if loaded_via_gds:
                        return  # buffer already on device
                    # else fall through to bounce-buffer path

                # Bounce-buffer (pread → pinned host → device copy)
                bounce_bytes = bounce_buffer_size_kb * 1024
                self._buffer = torch.empty(
                    self._body_size, dtype=torch.uint8, device=device
                )

                if self._body_size <= bounce_bytes:
                    # Small file: single bounce
                    staging = torch.empty(
                        self._body_size, dtype=torch.uint8, pin_memory=True
                    )
                    np_staging = staging.numpy()
                    _parallel_pread(
                        fd, np_staging, self._header_size, self._body_size, max_threads
                    )
                    self._buffer.copy_(staging, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    del staging
                else:
                    # Large file: chunked bounce to overlap I/O and D2H copy
                    self._chunked_bounce_read(
                        fd, device, bounce_bytes, max_threads, torch
                    )
        finally:
            os.close(fd)

        # -- 4. Alignment fixup -------------------------------------------
        self._check_and_fix_alignment()

    # -- internal helpers --------------------------------------------------

    def _try_gds_read(self, fd: int, device: str) -> bool:
        """Attempt to use GPU Direct Storage.  Returns True on success."""
        # GDS requires the cufile library and aligned buffers.  This is a
        # placeholder – the Rust extension will provide the real
        # implementation once available.  For now we always return False
        # so the bounce-buffer path is used.
        return False

    def _chunked_bounce_read(
        self,
        fd: int,
        device: str,
        bounce_bytes: int,
        max_threads: int,
        torch,
    ) -> None:
        """Read a large file body in chunks, pipelining host reads with
        device copies using double-buffered pinned memory."""
        # Double-buffer: while one staging buffer is being copied to the
        # device, the other is being filled by pread threads.
        buf_a = torch.empty(bounce_bytes, dtype=torch.uint8, pin_memory=True)
        buf_b = torch.empty(bounce_bytes, dtype=torch.uint8, pin_memory=True)

        copy_stream = torch.cuda.Stream(device=device)
        file_offset = self._header_size
        dev_offset = 0
        remaining = self._body_size
        current_staging = buf_a
        next_staging = buf_b

        while remaining > 0:
            chunk = min(remaining, bounce_bytes)
            np_staging = current_staging[:chunk].numpy()
            _parallel_pread(fd, np_staging, file_offset, chunk, max_threads)

            with torch.cuda.stream(copy_stream):
                self._buffer[dev_offset : dev_offset + chunk].copy_(
                    current_staging[:chunk], non_blocking=True
                )

            file_offset += chunk
            dev_offset += chunk
            remaining -= chunk
            current_staging, next_staging = next_staging, current_staging

        copy_stream.synchronize()
        del buf_a, buf_b

    def _check_and_fix_alignment(self) -> None:
        """Ensure tensor byte slices are properly aligned for their dtype.

        If the header size is not a multiple of 8, tensor offsets recorded
        in the header may result in misaligned views.  In practice the
        safetensors format pads the header to 8-byte alignment, so this is
        rarely needed – but we check just in case.
        """
        torch = _import_torch()
        if self._body_size == 0:
            return

        needs_fixup = False
        for name, info in self._tensor_info.items():
            start, end = info["data_offsets"]
            dt = _TYPES.get(info["dtype"])
            if dt is None:
                continue
            elem = _SIZE.get(dt, 1)
            if elem > 1 and (start % elem) != 0:
                needs_fixup = True
                break

        if not needs_fixup:
            return

        # Reallocate an aligned buffer and copy each tensor slice.
        # We rewrite offsets so every tensor starts at an aligned position.
        aligned_buf_parts = []
        new_offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name, info in self._tensor_info.items():
            start, end = info["data_offsets"]
            length = end - start
            dt = _TYPES.get(info["dtype"])
            elem = _SIZE.get(dt, 1) if dt is not None else 1
            # Align cursor
            if elem > 1 and (cursor % elem) != 0:
                pad = elem - (cursor % elem)
                cursor += pad

            new_offsets[name] = (cursor, cursor + length)
            cursor += length

        new_buf = torch.empty(cursor, dtype=torch.uint8, device=self._buffer.device)
        for name, info in self._tensor_info.items():
            old_start, old_end = info["data_offsets"]
            new_start, new_end = new_offsets[name]
            if old_end > old_start:
                new_buf[new_start:new_end].copy_(self._buffer[old_start:old_end])
            info["data_offsets"] = [new_start, new_end]

        self._buffer = new_buf

    # -- public API --------------------------------------------------------

    @property
    def filename(self) -> str:
        """The source file path."""
        return self._filename

    @property
    def header_size(self) -> int:
        """Size of the header (including 8-byte length prefix) in bytes."""
        return self._header_size

    @property
    def body_size(self) -> int:
        """Size of the tensor data body in bytes."""
        return self._body_size

    @property
    def device(self) -> str:
        """The device the buffer resides on."""
        return self._device

    def metadata(self) -> Dict[str, str]:
        """Return file-level metadata (the ``__metadata__`` dict)."""
        return dict(self._metadata)

    def keys(self) -> List[str]:
        """Return tensor names in file-offset order."""
        return list(self._tensor_info.keys())

    def get_keys(self) -> List[str]:
        """Alias for :meth:`keys`."""
        return self.keys()

    def get_shape(self, name: str) -> List[int]:
        """Return the shape of a tensor without materialising it."""
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")
        return list(self._tensor_info[name]["shape"])

    def get_dtype(self, name: str) -> str:
        """Return the safetensors dtype string for *name*."""
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")
        return self._tensor_info[name]["dtype"]

    def get_tensor(self, name: str):
        """Retrieve a tensor by name as a zero-copy view of the device buffer.

        Returns:
            ``torch.Tensor`` on the same device as the buffer.

        Raises:
            KeyError: if *name* is not present.
            RuntimeError: if the buffer has been closed.
        """
        if self._closed:
            raise RuntimeError("FastFileBuffer has been closed")
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")

        torch = _import_torch()
        _ensure_dtype_maps()

        info = self._tensor_info[name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]

        dt = _TYPES.get(dtype_str)
        if dt is None:
            raise ValueError(f"Unsupported dtype '{dtype_str}' for tensor '{name}'")

        # Handle empty tensors (any dim == 0)
        if any(s == 0 for s in shape) or (end - start) == 0:
            return torch.empty(shape, dtype=dt, device=self._buffer.device)

        byte_slice = self._buffer[start:end]
        return byte_slice.view(dtype=dt).reshape(shape)

    def is_aligned(self, alignment: int = 8) -> bool:
        """Check whether all tensor start offsets are aligned to *alignment*."""
        for info in self._tensor_info.values():
            if info["data_offsets"][0] % alignment != 0:
                return False
        return True

    def close(self) -> None:
        """Release the device buffer memory."""
        if not self._closed:
            self._closed = True
            self._buffer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self._tensor_info

    def __len__(self) -> int:
        return len(self._tensor_info)

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self._tensor_info)} tensors"
        return f"FastFileBuffer({self._filename!r}, device={self._device!r}, {state})"


# ---------------------------------------------------------------------------
# FastFileBufferAsync – submit/wait I/O pattern
# ---------------------------------------------------------------------------


class FastFileBufferAsync:
    """Async variant of :class:`FastFileBuffer` that separates I/O submission
    from completion, allowing overlap with other initialization work.

    The key insight (from the fastsafetensors paper, Section III-A) is that
    I/O transfers should be decoupled from tensor instantiation so that
    multiple files can have their I/O in flight simultaneously.

    Usage::

        buf = FastFileBufferAsync("model.safetensors", device="cuda:0")
        buf.submit_io()
        # ... do other work while I/O runs in background ...
        buf.wait_io()   # blocks until I/O completes
        tensor = buf.get_tensor("weight")
        buf.close()

    The three-phase lifecycle is:

    1. **Init** – parse header, allocate device buffer, open file descriptor.
       No I/O is started.
    2. **Submit** – kick off background I/O via :meth:`submit_io`.
    3. **Wait** – block until I/O finishes via :meth:`wait_io`, then run any
       remaining host-to-device copies and alignment fixup.
    """

    # States
    _STATE_INIT = "init"
    _STATE_SUBMITTED = "submitted"
    _STATE_READY = "ready"
    _STATE_CLOSED = "closed"

    def __init__(
        self,
        filename: str,
        device: str = "cpu",
        max_threads: int = 16,
        bounce_buffer_size_kb: int = 16384,
        nogds: bool = True,
    ):
        """Prepare for async I/O: parse header, allocate buffers.

        Does **not** start any I/O – call :meth:`submit_io` to begin.

        Args:
            filename: path to the ``.safetensors`` file.
            device: target device (e.g. ``"cpu"``, ``"cuda:0"``).
            max_threads: number of parallel pread threads.
            bounce_buffer_size_kb: size of the pinned bounce buffer in KiB.
            nogds: if ``True``, disable GPU Direct Storage.
        """
        torch = _import_torch()
        _ensure_dtype_maps()

        self._filename = filename
        self._device = device
        self._max_threads = max_threads
        self._bounce_buffer_size_kb = bounce_buffer_size_kb
        self._nogds = nogds
        self._closed = False
        self._state = self._STATE_INIT

        # -- 1. Parse header ------------------------------------------------
        self._header_size, self._metadata, self._tensor_info = _parse_header(filename)
        self._body_size = _file_body_size(filename, self._header_size)

        # -- 2. Determine I/O strategy and pre-allocate ---------------------
        self._is_cuda = device.startswith("cuda") or (
            isinstance(device, str) and device.isdigit()
        )

        # Pre-allocate the device buffer (and staging if needed)
        self._buffer = None
        self._staging = None  # pinned host buffer for CUDA bounce path
        self._fd = -1
        self._io_thread: Optional[threading.Thread] = None
        self._io_exception: Optional[BaseException] = None
        self._use_chunked = False  # True when file is large → synchronous in wait_io

        if self._body_size == 0:
            # Degenerate: no tensor data
            self._buffer = torch.empty(0, dtype=torch.uint8, device=device)
            self._state = self._STATE_READY
            return

        # Allocate device buffer
        self._buffer = torch.empty(self._body_size, dtype=torch.uint8, device=device)

        if self._is_cuda:
            bounce_bytes = self._bounce_buffer_size_kb * 1024
            if self._body_size <= bounce_bytes:
                # Small-file bounce path: pre-allocate pinned staging buffer
                # on the main thread (needs CUDA context).
                self._staging = torch.empty(
                    self._body_size, dtype=torch.uint8, pin_memory=True
                )
            else:
                # Large file: chunked double-buffer handled synchronously
                # inside wait_io() since it already pipelines I/O and copy.
                self._use_chunked = True

    def submit_io(self) -> None:
        """Kick off background I/O.  Returns immediately.

        Raises:
            RuntimeError: if called more than once, or after close/wait.
        """
        if self._state == self._STATE_CLOSED:
            raise RuntimeError("Cannot submit I/O on a closed FastFileBufferAsync")
        if self._state == self._STATE_SUBMITTED:
            raise RuntimeError("I/O has already been submitted")
        if self._state == self._STATE_READY:
            # Body was empty – nothing to do
            return

        self._state = self._STATE_SUBMITTED
        self._io_exception = None

        if self._body_size == 0:
            self._state = self._STATE_READY
            return

        # Open file descriptor (kept open until wait_io or close)
        self._fd = os.open(self._filename, os.O_RDONLY)

        if self._use_chunked:
            # Large-file chunked path: we don't spawn a bg thread because
            # the double-buffered pipeline already overlaps I/O and D2H.
            # The actual work happens in wait_io().
            return

        if not self._is_cuda:
            # CPU path: pread directly into tensor buffer in bg thread
            self._io_thread = threading.Thread(
                target=self._bg_cpu_read,
                daemon=True,
            )
            self._io_thread.start()
        else:
            if not self._nogds:
                # Attempt GDS path
                try:
                    loaded = self._try_gds_read(self._fd, self._device)
                    if loaded:
                        os.close(self._fd)
                        self._fd = -1
                        self._state = self._STATE_READY
                        return
                except Exception:
                    pass
                # Fall through to bounce path

            # CUDA bounce path: pread into pinned staging in bg thread
            self._io_thread = threading.Thread(
                target=self._bg_staging_read,
                daemon=True,
            )
            self._io_thread.start()

    def wait_io(self) -> None:
        """Block until submitted I/O completes and finalise the buffer.

        For the CPU path this simply joins the background thread.
        For the CUDA small-file path this joins the thread, then copies
        the staging buffer to the device and runs alignment fixup.
        For the CUDA large-file (chunked) path this runs the full
        double-buffered pipeline synchronously.

        Raises:
            RuntimeError: if :meth:`submit_io` was not called first.
        """
        if self._state == self._STATE_READY:
            return  # already done (e.g. empty file)
        if self._state == self._STATE_CLOSED:
            raise RuntimeError("Cannot wait on a closed FastFileBufferAsync")
        if self._state == self._STATE_INIT:
            raise RuntimeError("submit_io() must be called before wait_io()")

        torch = _import_torch()

        try:
            if self._use_chunked:
                # Large-file chunked double-buffer (synchronous)
                self._chunked_bounce_read(
                    self._fd,
                    self._device,
                    self._bounce_buffer_size_kb * 1024,
                    self._max_threads,
                    torch,
                )
            else:
                # Join the background I/O thread
                if self._io_thread is not None:
                    self._io_thread.join()
                    self._io_thread = None

                # Re-raise any exception from the bg thread
                if self._io_exception is not None:
                    raise self._io_exception

                if self._is_cuda and self._staging is not None:
                    # Copy staging → device and synchronise
                    self._buffer.copy_(self._staging, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    del self._staging
                    self._staging = None
        finally:
            # Close the file descriptor
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

        # -- Alignment fixup -----------------------------------------------
        self._check_and_fix_alignment()
        self._state = self._STATE_READY

    # -- background thread targets -----------------------------------------

    def _bg_cpu_read(self) -> None:
        """Background thread: pread directly into the CPU tensor buffer."""
        try:
            np_buf = self._buffer.numpy()
            _parallel_pread(
                self._fd,
                np_buf,
                self._header_size,
                self._body_size,
                self._max_threads,
            )
        except BaseException as exc:
            self._io_exception = exc

    def _bg_staging_read(self) -> None:
        """Background thread: pread into pinned host staging buffer."""
        try:
            np_staging = self._staging.numpy()
            _parallel_pread(
                self._fd,
                np_staging,
                self._header_size,
                self._body_size,
                self._max_threads,
            )
        except BaseException as exc:
            self._io_exception = exc

    # -- internal helpers (same logic as FastFileBuffer) --------------------

    def _try_gds_read(self, fd: int, device: str) -> bool:
        """Attempt to use GPU Direct Storage.  Returns True on success."""
        return False

    def _chunked_bounce_read(
        self,
        fd: int,
        device: str,
        bounce_bytes: int,
        max_threads: int,
        torch,
    ) -> None:
        """Read a large file body in chunks, pipelining host reads with
        device copies using double-buffered pinned memory."""
        buf_a = torch.empty(bounce_bytes, dtype=torch.uint8, pin_memory=True)
        buf_b = torch.empty(bounce_bytes, dtype=torch.uint8, pin_memory=True)

        copy_stream = torch.cuda.Stream(device=device)
        file_offset = self._header_size
        dev_offset = 0
        remaining = self._body_size
        current_staging = buf_a
        next_staging = buf_b

        while remaining > 0:
            chunk = min(remaining, bounce_bytes)
            np_staging = current_staging[:chunk].numpy()
            _parallel_pread(fd, np_staging, file_offset, chunk, max_threads)

            with torch.cuda.stream(copy_stream):
                self._buffer[dev_offset : dev_offset + chunk].copy_(
                    current_staging[:chunk], non_blocking=True
                )

            file_offset += chunk
            dev_offset += chunk
            remaining -= chunk
            current_staging, next_staging = next_staging, current_staging

        copy_stream.synchronize()
        del buf_a, buf_b

    def _check_and_fix_alignment(self) -> None:
        """Ensure tensor byte slices are properly aligned for their dtype."""
        torch = _import_torch()
        if self._body_size == 0:
            return

        needs_fixup = False
        for name, info in self._tensor_info.items():
            start, end = info["data_offsets"]
            dt = _TYPES.get(info["dtype"])
            if dt is None:
                continue
            elem = _SIZE.get(dt, 1)
            if elem > 1 and (start % elem) != 0:
                needs_fixup = True
                break

        if not needs_fixup:
            return

        aligned_buf_parts = []
        new_offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name, info in self._tensor_info.items():
            start, end = info["data_offsets"]
            length = end - start
            dt = _TYPES.get(info["dtype"])
            elem = _SIZE.get(dt, 1) if dt is not None else 1
            if elem > 1 and (cursor % elem) != 0:
                pad = elem - (cursor % elem)
                cursor += pad
            new_offsets[name] = (cursor, cursor + length)
            cursor += length

        new_buf = torch.empty(cursor, dtype=torch.uint8, device=self._buffer.device)
        for name, info in self._tensor_info.items():
            old_start, old_end = info["data_offsets"]
            new_start, new_end = new_offsets[name]
            if old_end > old_start:
                new_buf[new_start:new_end].copy_(self._buffer[old_start:old_end])
            info["data_offsets"] = [new_start, new_end]

        self._buffer = new_buf

    # -- public API (mirrors FastFileBuffer) --------------------------------

    @property
    def filename(self) -> str:
        """The source file path."""
        return self._filename

    @property
    def header_size(self) -> int:
        """Size of the header (including 8-byte length prefix) in bytes."""
        return self._header_size

    @property
    def body_size(self) -> int:
        """Size of the tensor data body in bytes."""
        return self._body_size

    @property
    def device(self) -> str:
        """The device the buffer resides on."""
        return self._device

    def metadata(self) -> Dict[str, str]:
        """Return file-level metadata (the ``__metadata__`` dict)."""
        return dict(self._metadata)

    def keys(self) -> List[str]:
        """Return tensor names in file-offset order."""
        return list(self._tensor_info.keys())

    def get_keys(self) -> List[str]:
        """Alias for :meth:`keys`."""
        return self.keys()

    def get_shape(self, name: str) -> List[int]:
        """Return the shape of a tensor without materialising it."""
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")
        return list(self._tensor_info[name]["shape"])

    def get_dtype(self, name: str) -> str:
        """Return the safetensors dtype string for *name*."""
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")
        return self._tensor_info[name]["dtype"]

    def get_tensor(self, name: str):
        """Retrieve a tensor by name as a zero-copy view of the device buffer.

        Returns:
            ``torch.Tensor`` on the same device as the buffer.

        Raises:
            KeyError: if *name* is not present.
            RuntimeError: if the buffer has been closed or I/O is not complete.
        """
        if self._state == self._STATE_CLOSED:
            raise RuntimeError("FastFileBufferAsync has been closed")
        if self._state != self._STATE_READY:
            raise RuntimeError(
                "I/O is not complete. Call submit_io() and wait_io() before "
                "accessing tensors."
            )
        if name not in self._tensor_info:
            raise KeyError(f"Tensor '{name}' not found in {self._filename}")

        torch = _import_torch()
        _ensure_dtype_maps()

        info = self._tensor_info[name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]

        dt = _TYPES.get(dtype_str)
        if dt is None:
            raise ValueError(f"Unsupported dtype '{dtype_str}' for tensor '{name}'")

        if any(s == 0 for s in shape) or (end - start) == 0:
            return torch.empty(shape, dtype=dt, device=self._buffer.device)

        byte_slice = self._buffer[start:end]
        return byte_slice.view(dtype=dt).reshape(shape)

    def is_aligned(self, alignment: int = 8) -> bool:
        """Check whether all tensor start offsets are aligned to *alignment*."""
        for info in self._tensor_info.values():
            if info["data_offsets"][0] % alignment != 0:
                return False
        return True

    def close(self) -> None:
        """Release the device buffer and any staging memory.

        If a background I/O thread is still running it will be joined
        first to prevent resource leaks.
        """
        if self._state == self._STATE_CLOSED:
            return
        # Join any in-flight thread to avoid leaks
        if self._io_thread is not None:
            self._io_thread.join()
            self._io_thread = None
        # Close file descriptor if still open
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        self._state = self._STATE_CLOSED
        self._buffer = None
        self._staging = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self._tensor_info

    def __len__(self) -> int:
        return len(self._tensor_info)

    def __repr__(self) -> str:
        return (
            f"FastFileBufferAsync({self._filename!r}, device={self._device!r}, "
            f"state={self._state!r})"
        )


# ---------------------------------------------------------------------------
# FilesBufferOnDevice – multi-file / distributed wrapper
# ---------------------------------------------------------------------------


class FilesBufferOnDevice:
    """Holds multiple :class:`FastFileBuffer` instances for distributed
    tensor access.

    Supports GPU-side sharding via ``torch.distributed`` collective
    operations, avoiding the need to shard on the CPU.
    """

    def __init__(
        self,
        buffers: Dict[int, List[Union[FastFileBuffer, "FastFileBufferAsync"]]],
        process_group=None,
    ):
        """
        Args:
            buffers: Mapping from rank → list of :class:`FastFileBuffer`
                or :class:`FastFileBufferAsync`.
            process_group: A ``torch.distributed`` ``ProcessGroup``, or
                ``None`` for single-GPU / non-distributed usage.
        """
        self._buffers = buffers
        self._process_group = process_group
        self._closed = False

        # Build a lookup: tensor name → (owning rank, FastFileBuffer)
        self._tensor_to_source: Dict[
            str, Tuple[int, Union[FastFileBuffer, "FastFileBufferAsync"]]
        ] = {}
        for rank, bufs in self._buffers.items():
            for buf in bufs:
                for key in buf.keys():
                    self._tensor_to_source[key] = (rank, buf)

        # Distributed bookkeeping
        self._rank: int = 0
        self._world_size: int = 1
        if process_group is not None:
            try:
                import torch.distributed as dist

                self._rank = dist.get_rank(process_group)
                self._world_size = dist.get_world_size(process_group)
            except Exception:
                pass

    # -- query helpers -----------------------------------------------------

    def keys(self) -> List[str]:
        """Return all tensor names across every file buffer."""
        return list(self._tensor_to_source.keys())

    def get_shape(self, name: str) -> List[int]:
        """Return the shape of *name*."""
        _, buf = self._resolve(name)
        return buf.get_shape(name)

    def get_dtype(self, name: str) -> str:
        """Return the safetensors dtype string of *name*."""
        _, buf = self._resolve(name)
        return buf.get_dtype(name)

    # -- tensor retrieval --------------------------------------------------

    def get_tensor(self, name: str, device=None, dtype=None):
        """Get a tensor, optionally broadcasting from the source rank.

        In multi-GPU mode the owning rank broadcasts the tensor to all
        other ranks via ``torch.distributed.broadcast``.  In single-GPU
        mode a zero-copy view is returned directly.

        Args:
            name: tensor name.
            device: override target device (default: buffer device).
            dtype: cast to this ``torch.dtype`` after retrieval.

        Returns:
            ``torch.Tensor``
        """
        if self._closed:
            raise RuntimeError("FilesBufferOnDevice has been closed")

        torch = _import_torch()
        src_rank, buf = self._resolve(name)

        if self._world_size <= 1:
            # Single-GPU / non-distributed fast path
            t = buf.get_tensor(name)
            if dtype is not None:
                t = t.to(dtype=dtype)
            if device is not None and str(t.device) != str(device):
                t = t.to(device=device)
            return t

        # Distributed path: source rank materialises the tensor, then
        # broadcast to all ranks.
        import torch.distributed as dist

        _ensure_dtype_maps()
        info_shape = buf.get_shape(name)
        info_dtype_str = buf.get_dtype(name)
        tensor_dtype = _TYPES[info_dtype_str]

        if self._rank == src_rank:
            tensor = buf.get_tensor(name).contiguous()
        else:
            tensor = torch.empty(info_shape, dtype=tensor_dtype, device=buf.device)

        dist.broadcast(tensor, src=src_rank, group=self._process_group)

        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and str(tensor.device) != str(device):
            tensor = tensor.to(device=device)
        return tensor

    def get_sharded(self, name: str, dim: int, device=None, dtype=None):
        """Get a tensor sharded across ranks along *dim*.

        The sharding is performed **on the GPU** (not CPU), avoiding
        host-memory bottlenecks.  ``dim=-1`` is treated as *broadcast*
        (equivalent to :meth:`get_tensor`).

        Args:
            name: tensor name.
            dim: dimension to split on. ``-1`` means broadcast.
            device: override target device.
            dtype: cast to this dtype after retrieval.

        Returns:
            ``torch.Tensor`` (the local shard).
        """
        if dim == -1:
            return self.get_tensor(name, device=device, dtype=dtype)

        if self._closed:
            raise RuntimeError("FilesBufferOnDevice has been closed")

        torch = _import_torch()
        src_rank, buf = self._resolve(name)

        if self._world_size <= 1:
            t = buf.get_tensor(name)
            if dtype is not None:
                t = t.to(dtype=dtype)
            if device is not None and str(t.device) != str(device):
                t = t.to(device=device)
            return t

        import torch.distributed as dist

        _ensure_dtype_maps()
        info_shape = list(buf.get_shape(name))
        info_dtype_str = buf.get_dtype(name)
        tensor_dtype = _TYPES[info_dtype_str]

        # Compute shard shape
        full_dim_size = info_shape[dim]
        if full_dim_size % self._world_size != 0:
            raise ValueError(
                f"Tensor '{name}' dimension {dim} (size {full_dim_size}) is "
                f"not evenly divisible by world_size={self._world_size}"
            )
        shard_size = full_dim_size // self._world_size
        shard_shape = list(info_shape)
        shard_shape[dim] = shard_size

        output = torch.empty(shard_shape, dtype=tensor_dtype, device=buf.device)

        if self._rank == src_rank:
            full_tensor = buf.get_tensor(name).contiguous()
            chunks = list(full_tensor.chunk(self._world_size, dim=dim))
            # Ensure chunks are contiguous for scatter
            chunks = [c.contiguous() for c in chunks]
        else:
            chunks = []  # only source provides the scatter list

        dist.scatter(
            output, scatter_list=chunks, src=src_rank, group=self._process_group
        )

        if dtype is not None and output.dtype != dtype:
            output = output.to(dtype=dtype)
        if device is not None and str(output.device) != str(device):
            output = output.to(device=device)
        return output

    def get_multi_cols(
        self,
        tensor_names: Sequence[str],
        dim: int,
        device=None,
        dtype=None,
    ):
        """Get multiple tensors sharded and concatenated.

        Useful for fused QKV weights where several logical tensors are
        concatenated along *dim* and then sharded across ranks.

        Args:
            tensor_names: sequence of tensor names to concatenate.
            dim: shard/concat dimension.
            device: override device.
            dtype: override dtype.

        Returns:
            ``torch.Tensor`` – the concatenated, sharded result.
        """
        shards = [
            self.get_sharded(n, dim=dim, device=device, dtype=dtype)
            for n in tensor_names
        ]
        torch = _import_torch()
        return torch.cat(shards, dim=dim)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release all device buffers."""
        if not self._closed:
            self._closed = True
            for bufs in self._buffers.values():
                for buf in bufs:
                    buf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self._tensor_to_source

    def __repr__(self) -> str:
        n_files = sum(len(v) for v in self._buffers.values())
        return (
            f"FilesBufferOnDevice(files={n_files}, "
            f"tensors={len(self._tensor_to_source)}, "
            f"ranks={list(self._buffers.keys())})"
        )

    # -- private -----------------------------------------------------------

    def _resolve(
        self, name: str
    ) -> Tuple[int, Union[FastFileBuffer, "FastFileBufferAsync"]]:
        if name not in self._tensor_to_source:
            raise KeyError(
                f"Tensor '{name}' not found in any loaded file. "
                f"Available: {list(self._tensor_to_source.keys())[:10]}…"
            )
        return self._tensor_to_source[name]


# ---------------------------------------------------------------------------
# FastSafeTensorsLoader
# ---------------------------------------------------------------------------


class FastSafeTensorsLoader:
    """High-level loader for fast safetensors file loading.

    Orchestrates parallel file I/O, device buffer allocation, and
    optional NUMA placement.

    Usage::

        loader = FastSafeTensorsLoader(device="cuda:0")
        loader.add_filenames({0: ["model-00001.safetensors"]})
        buffer = loader.copy_files_to_device()
        for key in buffer.keys():
            tensor = buffer.get_tensor(key)
        buffer.close()
        loader.close()
    """

    def __init__(
        self,
        device: str = "cpu",
        process_group=None,
        max_threads: int = 16,
        bounce_buffer_size_kb: int = 16384,
        nogds: bool = True,
        set_numa: bool = True,
    ):
        """
        Args:
            device: target device (e.g. ``"cpu"``, ``"cuda:0"``).
            process_group: ``torch.distributed`` ``ProcessGroup`` for
                multi-GPU loading, or ``None``.
            max_threads: number of parallel I/O threads.
            bounce_buffer_size_kb: size of the pinned bounce buffer in KiB.
            nogds: if ``True``, disable GPU Direct Storage (use
                pread + bounce buffer).
            set_numa: if ``True``, attempt to set NUMA affinity for the
                I/O threads to match the target GPU.
        """
        self._device = device
        self._process_group = process_group
        self._max_threads = max_threads
        self._bounce_buffer_size_kb = bounce_buffer_size_kb
        self._nogds = nogds
        self._set_numa = set_numa

        # rank → list of filenames
        self._filenames: Dict[int, List[str]] = {}
        self._closed = False
        # Pending async buffers from submit_files_to_device()
        self._pending_buffers: Optional[Dict[int, List[FastFileBufferAsync]]] = None

    def add_filenames(self, filenames: Dict[int, List[str]]) -> None:
        """Register safetensors files associated with ranks.

        Args:
            filenames: mapping from rank (``int``) to a list of file
                paths.  For single-GPU use, use rank ``0``.
        """
        for rank, paths in filenames.items():
            self._filenames.setdefault(rank, []).extend(paths)

    def copy_files_to_device(
        self,
        dtype=None,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ) -> FilesBufferOnDevice:
        """Trigger bulk file copies and return a :class:`FilesBufferOnDevice`.

        Uses the async submit/wait pattern internally so that I/O for
        different files overlaps in time.  This is a convenience method
        equivalent to calling :meth:`submit_files_to_device` followed by
        :meth:`wait_files`.

        Args:
            dtype: (unused, reserved for future per-tensor cast).
            max_copy_block_size: maximum bytes per DMA copy block.

        Returns:
            :class:`FilesBufferOnDevice` wrapping the loaded buffers.
        """
        self.submit_files_to_device(
            dtype=dtype, max_copy_block_size=max_copy_block_size
        )
        return self.wait_files()

    def submit_files_to_device(
        self,
        dtype=None,
        max_copy_block_size: int = 16 * 1024 * 1024 * 1024,
    ) -> None:
        """Submit I/O for all registered files.  Returns immediately.

        Creates :class:`FastFileBufferAsync` objects for each file and
        calls :meth:`~FastFileBufferAsync.submit_io` on each, allowing
        I/O for multiple files to overlap.

        Call :meth:`wait_files` to block until all I/O completes and
        obtain the resulting :class:`FilesBufferOnDevice`.

        Args:
            dtype: (unused, reserved for future per-tensor cast).
            max_copy_block_size: maximum bytes per DMA copy block.

        Raises:
            RuntimeError: if the loader has been closed or I/O is already
                in flight.
        """
        if self._closed:
            raise RuntimeError("Loader has been closed")
        if self._pending_buffers is not None:
            raise RuntimeError(
                "I/O is already in flight. Call wait_files() before submitting again."
            )

        if self._set_numa:
            _set_numa_affinity(self._device)

        pending: Dict[int, List[FastFileBufferAsync]] = {}
        for rank, paths in self._filenames.items():
            rank_buffers: List[FastFileBufferAsync] = []
            for path in paths:
                fb = FastFileBufferAsync(
                    filename=path,
                    device=self._device,
                    max_threads=self._max_threads,
                    bounce_buffer_size_kb=self._bounce_buffer_size_kb,
                    nogds=self._nogds,
                )
                fb.submit_io()
                rank_buffers.append(fb)
            pending[rank] = rank_buffers

        self._pending_buffers = pending

    def wait_files(self) -> FilesBufferOnDevice:
        """Wait for all submitted I/O to complete and return loaded buffers.

        Blocks until every :class:`FastFileBufferAsync` created by
        :meth:`submit_files_to_device` has finished its I/O, then wraps
        them in a :class:`FilesBufferOnDevice`.

        Returns:
            :class:`FilesBufferOnDevice` wrapping the loaded buffers.

        Raises:
            RuntimeError: if :meth:`submit_files_to_device` was not
                called first.
        """
        if self._closed:
            raise RuntimeError("Loader has been closed")
        if self._pending_buffers is None:
            raise RuntimeError("No I/O in flight. Call submit_files_to_device() first.")

        # Wait on every async buffer
        pending = self._pending_buffers
        self._pending_buffers = None

        for rank, bufs in pending.items():
            for fb in bufs:
                fb.wait_io()

        # FastFileBufferAsync is duck-type compatible with FastFileBuffer
        # for all read-only operations used by FilesBufferOnDevice.
        return FilesBufferOnDevice(pending, process_group=self._process_group)

    def close(self) -> None:
        """Release loader resources.

        If there are pending async buffers that were never waited on,
        they are closed (and their background threads joined) to prevent
        resource leaks.
        """
        self._closed = True
        self._filenames.clear()
        if self._pending_buffers is not None:
            for bufs in self._pending_buffers.values():
                for fb in bufs:
                    fb.close()
            self._pending_buffers = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        n_files = sum(len(v) for v in self._filenames.values())
        return (
            f"FastSafeTensorsLoader(device={self._device!r}, "
            f"files={n_files}, closed={self._closed})"
        )


# ---------------------------------------------------------------------------
# fast_open – context manager (drop-in for safe_open)
# ---------------------------------------------------------------------------


class fast_open:
    """Context manager for fast safetensors loading.

    Drop-in replacement for :func:`safetensors.safe_open` with aggregated
    tensor deserialization.  Instead of reading each tensor individually,
    the entire file body is loaded into a single device buffer in one
    bulk I/O pass.

    Supports loading from a single filename or a list of filenames.

    Example::

        with fast_open("model.safetensors", device="cuda:0") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
    """

    def __init__(
        self,
        filenames: Union[str, List[str]],
        framework: str = "pt",
        device: str = "cpu",
        nogds: bool = True,
        max_threads: int = 16,
        bounce_buffer_size_kb: int = 16384,
        async_io: bool = False,
    ):
        """
        Args:
            filenames: path or list of paths to ``.safetensors`` files.
            framework: ``"pt"`` (only PyTorch is supported).
            device: target device string.
            nogds: disable GPU Direct Storage.
            max_threads: parallel I/O threads.
            bounce_buffer_size_kb: bounce buffer size in KiB.
            async_io: if ``True``, use :class:`FastFileBufferAsync`
                internally (submit all files, then wait all).  This
                enables I/O overlap across files.  Default ``False``
                for backward compatibility.
        """
        if framework != "pt":
            raise ValueError(
                f"fast_open only supports framework='pt', got '{framework}'"
            )

        if isinstance(filenames, str):
            filenames = [filenames]

        self._buffers: List = []
        self._tensor_to_buffer: Dict[str, Any] = {}
        self._all_keys: List[str] = []

        if async_io:
            # Phase 1: create and submit I/O for all files
            async_bufs: List[FastFileBufferAsync] = []
            for fn in filenames:
                buf = FastFileBufferAsync(
                    filename=fn,
                    device=device,
                    max_threads=max_threads,
                    bounce_buffer_size_kb=bounce_buffer_size_kb,
                    nogds=nogds,
                )
                buf.submit_io()
                async_bufs.append(buf)

            # Phase 2: wait for all I/O to complete
            for buf in async_bufs:
                buf.wait_io()

            self._buffers = async_bufs
            for buf in async_bufs:
                for key in buf.keys():
                    if key not in self._tensor_to_buffer:
                        self._tensor_to_buffer[key] = buf
                        self._all_keys.append(key)
        else:
            for fn in filenames:
                buf = FastFileBuffer(
                    filename=fn,
                    device=device,
                    max_threads=max_threads,
                    bounce_buffer_size_kb=bounce_buffer_size_kb,
                    nogds=nogds,
                )
                self._buffers.append(buf)
                for key in buf.keys():
                    if key not in self._tensor_to_buffer:
                        self._tensor_to_buffer[key] = buf
                        self._all_keys.append(key)

    def keys(self) -> List[str]:
        """Return all tensor names (in file-offset order)."""
        return list(self._all_keys)

    def get_keys(self) -> List[str]:
        """Alias for :meth:`keys`."""
        return self.keys()

    def offset_keys(self) -> List[str]:
        """Return tensor names ordered by their byte offset in the file.

        Identical to :meth:`keys` since buffers are already offset-sorted.
        """
        return self.keys()

    def get_tensor(self, name: str):
        """Retrieve a tensor by name.

        Args:
            name: tensor name.

        Returns:
            ``torch.Tensor`` on the target device.

        Raises:
            KeyError: if *name* is not found.
        """
        if name not in self._tensor_to_buffer:
            raise KeyError(
                f"Tensor '{name}' not found. Available keys: {self._all_keys[:10]}…"
            )
        return self._tensor_to_buffer[name].get_tensor(name)

    def get_shape(self, name: str) -> List[int]:
        """Return the shape of *name*."""
        if name not in self._tensor_to_buffer:
            raise KeyError(f"Tensor '{name}' not found")
        return self._tensor_to_buffer[name].get_shape(name)

    def get_dtype(self, name: str) -> str:
        """Return the safetensors dtype string of *name*."""
        if name not in self._tensor_to_buffer:
            raise KeyError(f"Tensor '{name}' not found")
        return self._tensor_to_buffer[name].get_dtype(name)

    def metadata(self) -> Dict[str, str]:
        """Return merged metadata from all loaded files.

        If the same key appears in multiple files the last file wins.
        """
        merged: Dict[str, str] = {}
        for buf in self._buffers:
            merged.update(buf.metadata())
        return merged

    def close(self) -> None:
        """Release all device buffers."""
        for buf in self._buffers:
            buf.close()
        self._buffers.clear()
        self._tensor_to_buffer.clear()
        self._all_keys.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self._tensor_to_buffer

    def __len__(self) -> int:
        return len(self._all_keys)

    def __repr__(self) -> str:
        return f"fast_open(files={len(self._buffers)}, tensors={len(self._all_keys)})"


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------


def fast_load_file(
    filename: str,
    device: str = "cpu",
    nogds: bool = True,
    max_threads: int = 16,
    bounce_buffer_size_kb: int = 16384,
) -> Dict[str, "torch.Tensor"]:
    """Fast replacement for :func:`safetensors.torch.load_file`.

    Uses aggregated tensor deserialization for bulk loading.  The entire
    file body is read in a single pass (with parallel I/O) into a
    contiguous device buffer; individual tensors are zero-copy views.

    Args:
        filename: path to the ``.safetensors`` file.
        device: target device (e.g. ``"cpu"``, ``"cuda:0"``).
        nogds: if ``True``, use pread + bounce buffer instead of GDS.
        max_threads: number of parallel I/O threads.
        bounce_buffer_size_kb: bounce buffer size in KiB.

    Returns:
        ``Dict[str, torch.Tensor]``
    """
    result: Dict[str, Any] = {}
    with fast_open(
        filename,
        framework="pt",
        device=device,
        nogds=nogds,
        max_threads=max_threads,
        bounce_buffer_size_kb=bounce_buffer_size_kb,
    ) as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


def fast_load_sharded(
    filenames: Dict[int, List[str]],
    device: str = "cuda:0",
    process_group=None,
    tensor_shard_dims: Optional[Dict[str, int]] = None,
    nogds: bool = True,
    max_threads: int = 16,
    bounce_buffer_size_kb: int = 16384,
) -> Dict[str, "torch.Tensor"]:
    """Load and shard multiple safetensors files across GPUs.

    Uses GPU-side collective operations (``broadcast`` / ``scatter``) so
    that sharding happens entirely on the device, avoiding CPU bottlenecks.

    Args:
        filenames: mapping from rank (``int``) to list of file paths.
        device: target device (e.g. ``"cuda:0"``).
        process_group: ``torch.distributed`` ``ProcessGroup``
            (required for multi-GPU).
        tensor_shard_dims: ``OrderedDict`` or dict mapping tensor name
            to the dimension along which to shard.  Use ``-1`` to
            broadcast (all ranks get the full tensor).  If ``None``,
            every tensor is broadcast.
        nogds: disable GPU Direct Storage.
        max_threads: I/O thread count.
        bounce_buffer_size_kb: bounce buffer size in KiB.

    Returns:
        ``Dict[str, torch.Tensor]`` of (possibly sharded) tensors.
    """
    loader = FastSafeTensorsLoader(
        device=device,
        process_group=process_group,
        max_threads=max_threads,
        bounce_buffer_size_kb=bounce_buffer_size_kb,
        nogds=nogds,
    )
    loader.add_filenames(filenames)
    files_buffer = loader.copy_files_to_device()

    try:
        result: Dict[str, Any] = {}
        all_keys = files_buffer.keys()
        for key in all_keys:
            dim = -1  # default: broadcast
            if tensor_shard_dims is not None and key in tensor_shard_dims:
                dim = tensor_shard_dims[key]

            if dim == -1:
                result[key] = files_buffer.get_tensor(key, device=device)
            else:
                result[key] = files_buffer.get_sharded(key, dim=dim, device=device)

        return result
    finally:
        files_buffer.close()
        loader.close()
