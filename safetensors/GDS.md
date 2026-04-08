# GPU Direct Storage (GDS) Integration for Safetensors

This document describes the fast GPU loading path added to safetensors, based on
the techniques from the [fastsafetensors paper](https://arxiv.org/abs/2505.23072)
(IEEE CLOUD 2025). The implementation lives in the Rust core crate, the PyO3
Python bindings, and a pure-Python fallback layer.

---

## Overview

The default safetensors loading path uses `mmap` to map the file into host
memory, then instantiates each tensor one-by-one and copies it to the GPU.
This works well when the OS page cache is warm but severely under-utilises
NVMe storage bandwidth on cold reads because:

1. Each tensor triggers a separate page fault and DMA transfer.
2. The host CPU and memory act as a bounce buffer for every byte.
3. Python's GIL serialises tensor instantiation.

The fast path replaces this with **aggregated tensor deserialization**:

```text
┌─────────────┐     cuFileRead      ┌──────────┐
│  NVMe SSD   │ ──────────────────► │ GPU VRAM │
│ (safetensors│   (one bulk DMA,    │ (single  │
│   file)     │    bypasses CPU)    │  buffer) │
└─────────────┘                     └──────────┘
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                         tensor_a    tensor_b   tensor_c
                        (view)       (view)      (view)
```

1. Parse the header to learn all tensor offsets and sizes.
2. Allocate a **single** contiguous `uint8` buffer on the GPU.
3. Read the entire file body into that buffer in one bulk I/O pass.
4. Create zero-copy **views** into the buffer for each tensor.

When the NVIDIA cuFile (GDS) driver is available, step 3 uses `cuFileRead`
to transfer data directly from the NVMe controller to GPU VRAM via
peer-to-peer DMA over PCIe, bypassing host CPU and memory entirely.

When GDS is not available, the fallback path uses parallel `pread(2)` into
pinned host memory with `cudaMemcpy` H2D transfers, or (on CPU targets)
parallel `pread` directly into the tensor buffer.

---

## Architecture

### Rust Core Crate (`safetensors/src/bulk_io.rs`)

Gated behind the `fast_io` Cargo feature:

- **`BulkReadPlan`** — Splits a file body into transfer blocks for parallel I/O.
- **`PreadBulkReader`** — Parallel file reader using `libc::pread` + rayon.
- **`GdsReadPlan`** — 512-byte-aligned read blocks for cuFile.
- **`AlignmentFixup`** — Handles misaligned tensors from odd-sized headers.
- **NUMA helpers** — `get_numa_node_for_device`, `set_thread_numa_node`.

### Python Bindings (`bindings/python/src/`)

Four new Rust modules exposed to Python via PyO3:

| Module | Purpose |
|---|---|
| `device_buffer.rs` | GPU buffer allocation via PyTorch, tensor view creation, GDS FFI (`GdsContext`), buffer pool singleton |
| `bulk_reader.rs` | Parallel pread reader, direct `cudaMemcpy` bounce path, GDS file reader |
| `cuda_runtime.rs` | Runtime-loaded CUDA FFI (`cudaMalloc`, `cudaMemcpy`, `cudaHostAlloc`) |
| `lib.rs` (`fast_safe_open`) | PyO3 class that orchestrates the full pipeline |

Key design decisions:

- **Singleton cuFile driver** — `cuFileDriverOpen()` costs ~740 ms. A
  process-wide `OnceLock<GdsContext>` ensures this happens at most once.
- **Pre-registered buffer pool** — `cuFileBufRegister()` costs ~10 ms. A
  `Mutex<Option<GdsBufferPool>>` caches the registration across loads.
- **No DLPack** — Tensor views are created with PyTorch-native operations
  (`tensor[start:end].view(dtype).reshape(shape)`), avoiding DLPack spec
  limitations noted in the paper's Section VI.
- **Automatic fallback** — If GDS initialisation fails (no cuFile, no GPU
  support), the code silently falls back to the pread + bounce path.

### Python Layer (`py_src/safetensors/fast.py`)

Pure-Python implementation for environments without the GDS Cargo feature:

- **`FastFileBuffer`** — Parallel `os.pread` with `ThreadPoolExecutor`,
  double-buffered pinned-memory bounce for CUDA targets.
- **`FastFileBufferAsync`** — Async submit/wait pattern for multi-file overlap.
- **`FilesBufferOnDevice`** — Multi-GPU sharding via `torch.distributed`
  collective operations (`broadcast`, `scatter`).
- **`fast_open`** — Drop-in context manager replacement for `safe_open`.

### Integration into `load_file`

As of this implementation, `safetensors.torch.load_file` **automatically**
uses the fast path when the target device is CUDA:

```python
from safetensors.torch import load_file

# Automatically uses GDS when available — no code changes needed
tensors = load_file("model.safetensors", device="cuda:0")
```

The selection logic (in `torch.py`):

1. If `device` is CUDA, `torch.cuda.is_available()`, and `safetensors.fast`
   is importable → call `fast_load_file` which tries:
   - Rust `fast_safe_open` with `nogds=False` (GDS singleton + pool), or
   - Python parallel-pread fallback.
2. Otherwise → use the existing mmap path.

Set `SAFETENSORS_FAST_GPU=0` to force the old mmap behaviour.

---

## cuFile FFI Details

The GDS integration dynamically loads `libcufile.so` at runtime so the
binary runs on systems without GDS installed. The FFI matches the cuFile 1.x
ABI:

```c
// Loaded at runtime via libloading (Rust) or ctypes (Python)
CUfileError_t cuFileDriverOpen(void);
CUfileError_t cuFileDriverClose(void);
CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr);
void          cuFileHandleDeregister(CUfileHandle_t fh);
CUfileError_t cuFileBufRegister(const void *ptr, size_t size, int flags);
CUfileError_t cuFileBufDeregister(const void *ptr);
ssize_t       cuFileRead(CUfileHandle_t fh, void *buf, size_t size,
                         off_t file_offset, off_t buf_offset);
```

The `CUfileDescr_t` struct layout on Linux x86-64:

```c
struct CUfileDescr_t {
    int32_t  type;       // CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
    int32_t  _pad;       // alignment for the union
    int64_t  handle_fd;  // pointer-sized union {int fd; void *handle}
    void    *fs_ops;     // NULL for local filesystems
};
```

Files are opened with `O_RDONLY | O_DIRECT` for GDS compatibility.
CUDA driver version ≥ 12.2 relaxes the `O_DIRECT` requirement.

---

## Alignment Handling

The safetensors format places the tensor body immediately after the JSON
header. The header is padded to 8-byte alignment by the serialiser, but
some publicly available models have odd-sized headers that break GPU
pointer alignment requirements.

When a misaligned header is detected:

1. **GDS path** — Data is read with cuFile's internal alignment handling
   (cuFile accepts unaligned file offsets in compat mode). After the read,
   if dtype alignment is violated, an in-GPU-memory copy fixes the offsets.
2. **pread path** — Same post-read alignment fixup via in-buffer copies.

The fixup is computed by `compute_alignment_fixups()` in `bulk_io.rs` and
executed by `fix_alignment()` in `device_buffer.rs`. In practice the
overhead is < 1 ms.

---

## Performance

### System

- **Instance**: AWS g5.12xlarge
- **GPU**: 4× NVIDIA A10G (23 GB each)
- **NVMe**: 3.5 TB Amazon EC2 instance storage (`/dev/nvme1n1`), XFS
- **NVMe sequential read ceiling**: ~3,400 MB/s (measured with `dd O_DIRECT`)
- **CUDA**: 13.2, cuFile 1.15
- **GDS kernel module**: nvidia_fs 2.28.2

### Benchmark: 1,353 MB safetensors file, 196 tensors, cold NVMe

Page cache dropped (`echo 3 > /proc/sys/vm/drop_caches`) before every
iteration. 7 iterations per method, clean process (no module reloads).

| Method | API Call | Time | Throughput | NVMe Ceiling |
|---|---|---|---|---|
| **safetensors mmap** (baseline) | `load_file(f, device="cuda:0")` | 718 ms | 1,885 MB/s | 55% |
| **fastsafetensors GDS** (paper) | `SafeTensorsFileLoader(nogds=False)` | 504 ms | 2,683 MB/s | 79% |
| **This implementation** (auto) | `load_file(f, device="cuda:0")` | 428 ms | 3,164 MB/s | 93% |
| cuFileRead raw (floor) | Direct FFI, no tensor views | 414 ms | 3,272 MB/s | 96% |

### Speedup

```
load_file on CUDA:  1.68× faster  (718 ms → 428 ms)

vs fastsafetensors: 1.18× faster  (504 ms → 428 ms)

NVMe utilisation:   55% → 93%
```

### Where the remaining 10% goes

Step-by-step profiling of a single cold load:

| Step | Time | Notes |
|---|---|---|
| `cuFileRead` | 414 ms | Actual NVMe → GPU DMA (97% of total) |
| `cuFileHandleRegister` | 5 ms | Per-file, unavoidable |
| File open + mmap header parse | 2 ms | `open(O_DIRECT)` + mmap + JSON |
| `torch.empty` (alloc) | 0.1 ms | Buffer pool hit |
| `get_all_tensors` | 1 ms | Batched `torch.split` + view + reshape |
| Rust/Python GIL transitions | ~3 ms | Single merged `with_gil` block |
| Deregister + close | 3 ms | `cuFileHandleDeregister` + `close(fd)` |
| **Total** | **~428 ms** | |

Tensor views are created in a single batched `get_all_tensors()` call that
holds the GIL once, uses `torch.split` to slice the buffer, then views and
reshapes each chunk — replacing 196 individual `get_tensor` round-trips.
The constructor similarly merges its torch-import and device-string-conversion
into one GIL acquisition.

The ~740 ms `cuFileDriverOpen` cost is paid **once per process** (singleton)
and amortised over all subsequent loads.

### Warm page cache comparison

When the file is already in the OS page cache, the mmap path is faster for
CPU targets because it avoids actual I/O:

| Target | mmap | fast path | Winner |
|---|---|---|---|
| CPU (warm cache) | 38 ms | 1,350 ms | mmap (35×) |
| CUDA (warm cache) | 335 ms | 310 ms | fast (1.08×) |
| CUDA (cold NVMe) | 722 ms | 442 ms | **fast (1.63×)** |

This is why the auto-detection only activates the fast path for CUDA
devices — CPU loads stay on mmap.

### Batching impact

Tensor view creation via `get_all_tensors()` (one GIL hold + `torch.split`)
versus 196 individual `get_tensor()` calls:

| Approach | Time (196 tensors) |
|---|---|
| Per-key `get_tensor` loop | 1.8 ms |
| Batched `get_all_tensors` | 1.0 ms |

The savings are modest in absolute terms because tensor view creation is
already fast. The batching mainly eliminates PyO3 method dispatch overhead
(196 Rust→Python round-trips → 1). On cold NVMe the cuFileRead (414 ms)
dominates, so these milliseconds are within the noise floor — but the
improvement matters for warm-cache or smaller-file scenarios where
overhead is a larger fraction of total time.

---

## Building

### Default build (no GDS, fast_io only)

```bash
# Rust core crate with parallel pread support
cargo build --features fast_io

# Python bindings (parallel pread, no GDS)
cd bindings/python
maturin develop --release
```

### GDS build

Requires `libcufile.so` (NVIDIA GDS / cuFile) on the system:

```bash
cd bindings/python
maturin develop --release --features gds
```

The `gds` feature enables:
- Runtime loading of `libcufile.so` and `libcudart.so`
- `GdsContext` singleton with cuFile driver management
- `GdsBufferPool` with pre-registered GPU memory
- `CudaRuntime` with direct `cudaMemcpy` from Rust threads
- `read_file_to_device_gds` for cuFile bulk reads

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SAFETENSORS_FAST_GPU` | `1` | Set to `0` to disable the fast GPU path in `load_file` |

---

## System Requirements for GDS

| Requirement | Details |
|---|---|
| GPU | NVIDIA with GPUDirect RDMA support (`cudaDevAttrGPUDirectRDMASupported`) |
| Driver | NVIDIA driver with `nvidia_fs` kernel module loaded |
| cuFile | `libcufile.so` installed (part of CUDA Toolkit ≥ 11.4) |
| Filesystem | XFS, ext4, or other GDS-compatible filesystem |
| File open | `O_DIRECT` support (relaxed in CUDA ≥ 12.2) |

When any of these are unavailable, the implementation automatically falls
back to parallel `pread` + pinned bounce buffer + `cudaMemcpy`, which still
outperforms the mmap path on cold storage.

---

## Comparison with fastsafetensors

| Feature | fastsafetensors | This implementation |
|---|---|---|
| Language | Python + C++ (pybind11) | Rust + Python (PyO3) |
| Tensor instantiation | DLPack capsules | PyTorch-native views |
| GDS driver lifecycle | Per-loader instance | Process-wide singleton (saves 740 ms) |
| Buffer registration | Per-file register/deregister | Pooled (saves 18 ms/file) |
| Bounce buffer | `cudaHostAlloc` in C++ threads | `cudaHostAlloc` in rayon threads |
| Tensor view creation | 1-by-1 via DLPack | Batched `torch.split` via `get_all_tensors` |
| Multi-GPU sharding | `torch.distributed` broadcast/scatter | Same |
| Async I/O | `submit_read`/`wait_read` | `FastFileBufferAsync` submit/wait |
| Integration | Separate library, explicit API | Drop-in via `load_file` auto-detection |
| Throughput (1.3 GB, cold NVMe) | 2,683 MB/s (79% ceiling) | 3,164 MB/s (93% ceiling) |

---

## References

- Yoshimura, T. et al. "Speeding up Model Loading with fastsafetensors."
  arXiv:2505.23072, IEEE CLOUD 2025.
- [fastsafetensors source](https://github.com/foundation-model-stack/fastsafetensors)
- [NVIDIA GPUDirect Storage Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)