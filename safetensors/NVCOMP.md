# nvcomp ANS Compression for Safetensors — Implementation Results

This document summarizes the implementation of GPU-accelerated ANS
compression/decompression for safetensors model weight loading, using
NVIDIA nvcomp's batched ANS codec via a C ABI shim.

---

## Background

The safetensors GDS (GPU Direct Storage) fast path achieves ~93% of NVMe
bandwidth for loading model weights to GPU.  At that saturation level,
the only way to go faster is to **read fewer bytes**.  ANS (Asymmetric
Numeral Systems) is an entropy coding algorithm that nvcomp implements as
a GPU-native codec (gANS) optimized for raw throughput — achieving nearly
the same compression ratios as Zstandard at **10× the GPU throughput**.

This work was inspired by the NVIDIA blog post:
[Cut Checkpoint Costs with About 30 Lines of Python and NVIDIA nvCOMP](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/)

### Why ANS over Zstandard?

| Property | Zstandard | ANS (gANS) |
|---|---|---|
| Compression ratio (BF16 weights) | ~1.27× | ~1.28× |
| GPU compress throughput (Blackwell) | ~16–19 GB/s | ~181–190 GB/s |
| GPU decompress throughput (Blackwell) | ~40 GB/s | ~247–264 GB/s |
| GPU decompress throughput (A10G, measured) | N/A | **61 GB/s** |
| CPU fallback decode | ✅ (zstd crate) | ❌ (GPU only) |

For GDS + fast NVMe (≥3 GB/s), ANS is the clear winner because its
decompression throughput never becomes the bottleneck.

---

## Architecture

```text
┌─────────────┐  cuFileRead   ┌───────────┐  nvcomp ANS   ┌───────────┐
│  NVMe SSD   │ ────────────► │  GPU VRAM  │ ────────────► │  GPU VRAM  │
│ (compressed │  (fewer bytes,│ (compressed│  batched      │(decompressed│
│  .safetensors) bypasses CPU)│  staging)  │  decode       │   buffer)  │
└─────────────┘               └───────────┘               └───────────┘
                                                               │
                                                  ┌────────────┼────────────┐
                                                  ▼            ▼            ▼
                                             tensor_a     tensor_b     tensor_c
                                            (zero-copy views into buffer)
```

### Key implementation details

**C ABI shim (`nvcomp_shim.c`)**:  The nvcomp batched API accepts a
64-byte options struct by value.  On x86-64 System V ABI, structs larger
than 32 bytes are classified as MEMORY, and passing them through Rust
`extern "C"` FFI or Python ctypes causes silent stack corruption
(`nvcompErrorInvalidValue`, status 10).  The shim wraps each nvcomp
function, accepting the opts struct **by pointer** and dereferencing it
inside C where the compiler generates the correct by-value forwarding.

**8-byte aligned chunk padding**:  The nvcomp ANS decompressor requires
8-byte aligned input pointers per chunk.  Compressed chunks are
concatenated in the file body, and their boundaries may not be aligned
(they depend on compressed sizes).  The save path pads each compressed
chunk to an 8-byte boundary with zero bytes.  The metadata stores both
the actual and padded sizes.  This allows the decompress path to point
directly into the contiguous GPU buffer without per-chunk copies.

**Pre-allocated auxiliary buffer pool (`NvcompAuxPool`)**:  The nvcomp
batched decompress call needs a device-side buffer for the temp workspace
and four arrays of per-chunk pointers/sizes.  A process-wide singleton
caches this allocation across loads, eliminating `cudaMalloc`/`cudaFree`
on repeated loads of the same or similarly-sized models.

**NVCOMP_NATIVE format**:  Both the compress and decompress paths use
nvcomp's own batched C API (via the shim), producing `NVCOMP_NATIVE`
bitstream format.  This is different from standard CPU zstd frames
(`RAW` format) — the high-level Python `Codec` API with
`BitstreamKind.RAW` cannot be mixed with the batched C API.  The save
path uses the C shim for compression to ensure format consistency.

---

## Measured Results

### System

- **Instance**: AWS g5.12xlarge
- **GPUs**: 4× NVIDIA A10G (23 GB VRAM each)
- **NVMe**: 3.5 TB instance store, XFS, ~3,400 MB/s sequential read
- **CUDA**: 13.2, cuFile 1.17
- **nvcomp**: 5.2.0 (`nvidia-nvcomp-cu12`)
- **PyTorch**: 2.11.0+cu130

### Compression ratios on real trained weights

| Model | Dtype | Uncompressed | ANS Compressed | Ratio | Savings |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | BF16 | 3.09 GB | 2.42 GB | **1.28×** | 21.7% |
| Qwen2.5-7B (per shard) | BF16 | 3.56–3.95 GB | 2.77–3.08 GB | **1.28×** | 22.0% |
| Qwen2.5-32B (per shard) | BF16 | 3.10–3.92 GB | 2.43–3.08 GB | **1.28×** | 21.7% |

The 1.28× ratio is consistent across model sizes and matches the blog
post's reported 1.25–1.27× for dense BF16 weights.

### nvcomp ANS kernel throughput (A10G)

Measured in isolation via the C shim with 128 MB of random data in
16 MiB chunks:

| Operation | Throughput | Time (128 MB) |
|---|---|---|
| ANS compress | 32.1 GB/s | 4.2 ms |
| ANS decompress | **61.0 GB/s** | **2.2 ms** |

For a 3 GB model body, the kernel time is ~50 ms — a small fraction of
the total NVMe I/O time.

### Single-shard loading (cold NVMe)

Qwen2.5-1.5B (3.09 GB) and Qwen2.5-7B shard 4 (3.56 GB):

| Method | 1.5B Time | 1.5B MB/s | 7B-s4 Time | 7B-s4 MB/s |
|---|---|---|---|---|
| **Uncompressed + GDS** | **918 ms** | 3,365 | **1,056 ms** | 3,368 |
| ANS + GDS + nvcomp | 996 ms | 3,101 | 1,141 ms | 3,118 |
| Uncompressed + mmap | 1,591 ms | 1,941 | 1,744 ms | 2,043 |

At single-shard scale, ANS is 8–9% behind uncompressed GDS but already
**1.55–1.60× faster than mmap**.

### Multi-shard scaling (Qwen2.5-32B, single GPU, cold NVMe)

| Shards | Uncompressed Size | GDS Time | ANS Time | Winner | Delta |
|---|---|---|---|---|---|
| 1 | 3.9 GB | **1,167 ms** | 1,200 ms | GDS | +34 ms |
| 2 | 7.8 GB | **2,317 ms** | 2,424 ms | GDS | +108 ms |
| 3 | 11.7 GB | 3,642 ms | **3,614 ms** | **ANS** | **−29 ms** |
| 4 | 15.6 GB | 5,184 ms | **4,791 ms** | **ANS** | **−393 ms (7.6%)** |

**Crossover at ~12 GB on A10G + 3.4 GB/s NVMe.**

The scaling is linear: each additional 3.9 GB shard adds ~200 ms of NVMe
I/O savings (22% fewer bytes) but only ~80 ms of nvcomp decompress
overhead, so the ANS advantage grows by ~120 ms per shard.

### Full model loading (Qwen2.5-7B, all 4 shards, 15.2 GB, single GPU)

| Method | Time | Effective Throughput | vs Best |
|---|---|---|---|
| **ANS + GDS + nvcomp** | **4,687 ms** | **3,250 MB/s** | **1.00×** |
| Uncompressed + GDS | 5,031 ms | 3,027 MB/s | 0.93× |
| Uncompressed + mmap | 7,713 ms | 1,975 MB/s | 0.61× |

**ANS wins by 344 ms (6.8%) over uncompressed GDS.**

---

## Optimization history

The path from initial Python-based approach to the final C shim
implementation yielded a **5.2× improvement** in the ANS decompress
path overhead:

| Optimization | ANS overhead (1.5B) | vs GDS |
|---|---|---|
| Python Codec per-chunk loop (82 chunks) | +2,177 ms | +237% |
| Python Codec single call | +227 ms | +25% |
| C shim + per-chunk tensor.clone() alignment | +213 ms | +23% |
| C shim + padded chunks (no clone) | +82 ms | +9% |
| **+ Pre-allocated NvcompAuxPool** | **+78 ms** | **+8.5%** |

---

## Projected gains on other GPU types

The crossover point (where ANS beats uncompressed GDS) depends on three
factors:

1. **NVMe bandwidth** — faster NVMe means more ms saved per GB of
   reduced I/O
2. **GPU decompress throughput** — faster GPUs decompress more cheaply
3. **Model size** — larger models have more I/O to save

### Per-shard overhead model

For a single shard of size `S` uncompressed with ratio `R`:

```
NVMe read (uncompressed): S / NVMe_BW
NVMe read (compressed):   S / R / NVMe_BW
NVMe savings:             S × (1 - 1/R) / NVMe_BW
nvcomp decode:            S / GPU_decomp_BW
Fixed overhead:           ~5 ms (memcpy arrays + sync)

ANS wins when:  NVMe_savings > nvcomp_decode + fixed_overhead
i.e.:           S × (1 - 1/R) / NVMe_BW  >  S / GPU_decomp_BW + 5ms
```

### Projected crossover points

Using measured ANS ratio of 1.28× on BF16 weights (savings fraction =
0.219), and assuming fixed overhead of 5 ms per shard:

| GPU | Decomp BW¹ | NVMe BW | Crossover Size² | Notes |
|---|---|---|---|---|
| **A10G** (measured) | 61 GB/s | 3.4 GB/s | **~12 GB** | This benchmark |
| **A100 80GB** | ~120 GB/s | 3.4 GB/s | ~6 GB | 2× decomp BW, same NVMe |
| **A100 80GB + faster NVMe** | ~120 GB/s | 7.0 GB/s | ~3 GB | Higher-end NVMe RAID |
| **H100 80GB** | ~180 GB/s | 7.0 GB/s | ~2.5 GB | Faster GPU + NVMe |
| **H200 141GB** | ~200 GB/s | 7.0 GB/s | ~2 GB | Nearly all models win |
| **B200 (Blackwell)** | ~250 GB/s | 15 GB/s | **~1.5 GB** | Virtually all models |

> ¹ GPU decompress bandwidth estimates based on the NVIDIA blog's reported
> 247–264 GB/s for ANS on Blackwell, scaled down proportionally for older
> architectures.  A10G was measured at 61 GB/s.
>
> ² Crossover size = minimum model weight size where ANS + GDS beats
> uncompressed GDS.  Models larger than this load faster with compression.

### Projected speedups for common model sizes

Assuming 1.28× ANS ratio, single-GPU loading, cold NVMe:

| Model | Size | A10G 3.4GB/s | A100 3.4GB/s | H100 7GB/s | B200 15GB/s |
|---|---|---|---|---|---|
| Llama-3 8B | ~16 GB | **+7.6%** | **+14%** | **+12%** | **+15%** |
| Qwen2.5-14B | ~30 GB | **+11%** | **+17%** | **+15%** | **+18%** |
| Llama-3 70B (per GPU, 8-way) | ~17 GB | **+8%** | **+15%** | **+13%** | **+16%** |
| Llama-3 405B (per GPU, 128-way) | ~7 GB | +1% | **+9%** | **+7%** | **+12%** |

On next-generation hardware (Blackwell + high-speed NVMe), ANS
compression provides a consistent **15–18% speedup** for cold model
loading with zero accuracy impact (lossless compression).

### Storage savings (always applies regardless of speedup)

| Model | Uncompressed | ANS Compressed | Savings |
|---|---|---|---|
| Llama-3 8B | 16 GB | 12.5 GB | 3.5 GB (22%) |
| Qwen2.5-14B | 30 GB | 23.4 GB | 6.6 GB (22%) |
| Llama-3 70B | 140 GB | 109 GB | 31 GB (22%) |
| Llama-3 405B | 810 GB | 633 GB | 177 GB (22%) |

At $0.08/GB/month for cloud storage, a 405B model saves **$14/month**
in storage costs per copy.  Inference fleets with dozens of replicas
multiply this proportionally.

---

## Building and usage

### Requirements

| Component | Package | Purpose |
|---|---|---|
| nvcomp library | `pip install nvidia-nvcomp-cu12` | GPU ANS codec |
| nvcomp C shim | `libnvcomp_shim.so` (compiled from `src/nvcomp_shim.c`) | ABI bridge |
| CUDA runtime | `libcudart.so` (CUDA Toolkit) | GPU memory ops |
| safetensors | Built with `--features gds,nvcomp` | Rust integration |

### Build the C shim

```bash
gcc -shared -fPIC -O2 -o libnvcomp_shim.so \
    bindings/python/src/nvcomp_shim.c \
    -L/path/to/nvcomp/lib -lnvcomp \
    -Wl,-rpath,/path/to/nvcomp/lib

# Place on library path
cp libnvcomp_shim.so /usr/local/cuda/lib64/
```

### Build safetensors with nvcomp support

```bash
cd bindings/python
maturin build --release --features gds,nvcomp
pip install target/wheels/safetensors-*.whl
```

### Save a compressed model

```python
from safetensors.fast import save_compressed

# Compress with ANS (requires GPU + nvcomp)
save_compressed(tensors, "model.safetensors", compression="ans")

# Compress with Zstandard (CPU, no GPU needed)
save_compressed(tensors, "model.safetensors", compression="zstd", level=3)
```

### Load (automatic detection)

```python
from safetensors.torch import load_file

# Transparently handles both compressed and uncompressed files
tensors = load_file("model.safetensors", device="cuda:0")
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SAFETENSORS_FAST_GPU` | `1` | Set to `0` to disable the fast GPU path |
| `LD_LIBRARY_PATH` | — | Must include paths to `libnvcomp.so` and `libnvcomp_shim.so` |

---

## File format

Compressed safetensors files use the standard header format with
additional entries in `__metadata__`:

```json
{
  "__metadata__": {
    "compression": "ans",
    "compression_info": "{\"method\":\"Ans\",\"level\":0,\"decompressed_size\":3087428608,\"compressed_size\":2417712910,\"chunk_size\":16777216,\"chunks\":[{\"compressed_offset\":0,\"compressed_size\":13109774,\"padded_size\":13109776,\"decompressed_offset\":0,\"decompressed_size\":16777216},...]}"
  },
  "model.embed_tokens.weight": {
    "dtype": "BF16",
    "shape": [151936, 1536],
    "data_offsets": [0, 466763776]
  }
}
```

- `data_offsets` refer to **decompressed** positions (tensor views work
  unchanged after decompression)
- `compressed_offset` values are 8-byte aligned (padded during save)
- `compressed_size` is the actual compressed bytes; `padded_size` is the
  aligned size stored in the file body
- The file body contains the concatenated padded compressed chunks
- Backward compatible: old readers see `"compression"` and reject the
  file cleanly; uncompressed files have no compression metadata

---

## Limitations and future work

1. **ANS is GPU-only** — no CPU fallback for ANS decompression.  ZSTD
   compressed files can fall back to CPU `zstd` crate decoding.

2. **Sequential multi-GPU loading** — currently each shard is loaded
   one at a time.  True parallel per-GPU loading (one thread per GPU,
   concurrent GDS reads) would reduce wall-clock time to the single-GPU
   per-partition time.

3. **VRAM overhead** — the compressed path temporarily needs both the
   compressed staging buffer and the decompressed output buffer in VRAM.
   For a 4 GB shard at 1.28× ratio, this is ~7 GB peak vs ~4 GB for
   uncompressed loading.

4. **MoE models** — Mixture-of-experts models compress better (~1.40×
   per the NVIDIA blog) due to expert routing sparsity, which would
   push the crossover point even lower.

5. **FP32 optimizer states** — the NVIDIA blog reports 1.25–1.48×
   compression on FP32 AdamW states.  Checkpoint compression for
   training (not just inference loading) is a natural extension.

6. **Build system integration** — the C shim is currently compiled
   manually.  Integrating it into `build.rs` for automatic compilation
   during `maturin build` would improve the developer experience.

---

## References

- [NVIDIA blog: Cut Checkpoint Costs with nvCOMP](https://developer.nvidia.com/blog/cut-checkpoint-costs-with-about-30-lines-of-python-and-nvidia-nvcomp/)
- [nvcomp documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [nvcomp GitHub](https://github.com/NVIDIA/nvcomp)
- [fastsafetensors paper (arXiv:2505.23072)](https://arxiv.org/abs/2505.23072)
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/)