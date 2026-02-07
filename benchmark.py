import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Check if required packages are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Some tests will be skipped.")

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not installed. Cannot download models.")
    print("Install with: pip install huggingface_hub")

import safetensors
from safetensors import safe_open

# Try to import GPU module
try:
    from safetensors.gpu import (
        is_available as gpu_is_available,
        get_device_count,
        load_file as gpu_load_file,
        GpuLoaderConfig,
    )
    HAS_GPU_MODULE = True
except ImportError as e:
    HAS_GPU_MODULE = False
    print(f"Warning: GPU module not available: {e}")


# Sample models to test (various sizes)
SAMPLE_MODELS = [
    # Very small model for quick tests (~50KB)
    {
        "repo_id": "hf-internal-testing/tiny-random-bert",
        "filename": "model.safetensors",
        "description": "Tiny Random BERT (for testing)",
        "size": "tiny",
    },
    # Small BERT model (~440MB)
    {
        "repo_id": "bert-base-uncased",
        "filename": "model.safetensors",
        "description": "BERT Base Uncased",
        "size": "small",
    },
    # GPT-2 model (~500MB)
    {
        "repo_id": "gpt2",
        "filename": "model.safetensors",
        "description": "GPT-2 Small",
        "size": "small",
    },
    # Mid-sized model: Phi-2 (~5.5GB)
    {
        "repo_id": "microsoft/phi-2",
        "filename": "model-00001-of-00002.safetensors",
        "description": "Phi-2 (shard 1)",
        "size": "medium",
    },
    # Large model: Qwen1.5-7B (~15GB)
    {
        "repo_id": "Qwen/Qwen1.5-7B",
        "filename": "model-00001-of-00004.safetensors",
        "description": "Qwen1.5-7B (shard 1)",
        "size": "large",
    },
]


def download_model(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """Download a safetensor file from HuggingFace Hub."""
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub is required for downloading models")

    print(f"Downloading {filename} from {repo_id}...")

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        print(f"  Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  Failed to download: {e}")
        raise


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def benchmark_standard_gpu_loading(filepath: str, device: str = "cuda:0") -> Dict[str, Any]:
    """Benchmark standard safetensors loading to GPU."""
    results = {
        "method": "standard (safe_open)",
        "device": device,
        "filepath": filepath,
        "file_size_mb": get_file_size_mb(filepath),
    }

    # Measure loading time
    start = time.perf_counter()

    tensors = {}
    with safe_open(filepath, framework="pt", device=device) as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    elapsed = time.perf_counter() - start

    results["load_time_s"] = elapsed
    results["num_tensors"] = len(tensors)
    results["total_elements"] = sum(t.numel() for t in tensors.values())
    results["throughput_mb_s"] = results["file_size_mb"] / elapsed

    # Clean up
    del tensors
    if HAS_TORCH:
        torch.cuda.empty_cache()

    return results


def benchmark_gpu_loading(filepath: str, device: str = "cuda:0",
                          config: Optional[GpuLoaderConfig] = None,
                          method_name: str = "gpu_load_file") -> Dict[str, Any]:
    """Benchmark GPU-optimized safetensors loading."""
    if not HAS_GPU_MODULE:
        raise RuntimeError("GPU module not available")

    results = {
        "method": method_name,
        "device": device,
        "filepath": filepath,
        "file_size_mb": get_file_size_mb(filepath),
        "config": str(config) if config else "default",
    }

    # Measure loading time
    start = time.perf_counter()

    tensors = gpu_load_file(filepath, device=device, config=config)

    elapsed = time.perf_counter() - start

    results["load_time_s"] = elapsed
    results["num_tensors"] = len(tensors)
    results["throughput_mb_s"] = results["file_size_mb"] / elapsed

    # Clean up
    del tensors
    if HAS_TORCH:
        torch.cuda.empty_cache()

    return results


def print_results(results: Dict[str, Any], indent: int = 2):
    """Pretty print benchmark results."""
    prefix = " " * indent
    print(f"{prefix}Method: {results['method']}")
    print(f"{prefix}Device: {results['device']}")
    print(f"{prefix}File size: {results['file_size_mb']:.2f} MB")
    print(f"{prefix}Load time: {results['load_time_s']:.4f} s")
    print(f"{prefix}Throughput: {results['throughput_mb_s']:.2f} MB/s")
    print(f"{prefix}Tensors loaded: {results['num_tensors']}")
    if "config" in results:
        print(f"{prefix}Config: {results['config']}")


def run_comparison_benchmark(filepath: str, description: str = ""):
    """Run and compare standard GPU vs optimized GPU loading."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {description or filepath}")
    print(f"File: {filepath}")
    print(f"Size: {get_file_size_mb(filepath):.2f} MB")
    print(f"{'='*60}")

    if not HAS_TORCH or not torch.cuda.is_available():
        print("ERROR: PyTorch with CUDA is required for GPU benchmarks")
        return []

    results = []

    # Standard CUDA loading (baseline)
    print("\n[1] Standard GPU loading (safe_open with CUDA)...")
    try:
        r = benchmark_standard_gpu_loading(filepath, device="cuda:0")
        results.append(r)
        print_results(r)
    except Exception as e:
        print(f"  Failed: {e}")

    # GPU-optimized loading (if available)
    if HAS_GPU_MODULE:
        print("\n[2] GPU-optimized loading (default: 4 streams)...")
        try:
            r = benchmark_gpu_loading(filepath, device="cuda:0",
                                      method_name="gpu_load_file (4 streams)")
            results.append(r)
            print_results(r)
        except Exception as e:
            print(f"  Failed: {e}")

        print("\n[3] GPU-optimized loading (32 streams)...")
        try:
            config = GpuLoaderConfig(
                num_streams=32,
                max_pinned_memory_mb=1024,
                device_id=0,
                double_buffer=True
            )
            r = benchmark_gpu_loading(filepath, device="cuda:0", config=config,
                                      method_name="gpu_load_file (32 streams)")
            results.append(r)
            print_results(r)
        except Exception as e:
            print(f"  Failed: {e}")

        print("\n[4] GPU-optimized loading (64 streams)...")
        try:
            config = GpuLoaderConfig(
                num_streams=64,
                max_pinned_memory_mb=2048,
                device_id=0,
                double_buffer=True
            )
            r = benchmark_gpu_loading(filepath, device="cuda:0", config=config,
                                      method_name="gpu_load_file (64 streams)")
            results.append(r)
            print_results(r)
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\nGPU module not available - skipping optimized loading tests")

    # Summary
    if len(results) > 1:
        print(f"\n{'─'*40}")
        print("Summary (comparing to standard GPU loading):")
        baseline = results[0]["load_time_s"]
        for r in results:
            speedup = baseline / r["load_time_s"]
            print(f"  {r['method']}: {r['load_time_s']:.4f}s ({speedup:.2f}x)")

    return results


def test_gpu_module():
    """Test basic GPU module functionality."""
    print("\n" + "="*60)
    print("GPU Module Tests")
    print("="*60)

    print(f"\nSafetensors version: {safetensors.__version__}")

    if not HAS_GPU_MODULE:
        print("❌ GPU module not available")
        return False

    print("✓ GPU module imported successfully")

    # Check GPU availability
    available = gpu_is_available()
    print(f"✓ GPU available: {available}")

    # Check device count
    try:
        count = get_device_count()
        print(f"✓ CUDA device count: {count}")
    except Exception as e:
        print(f"❌ Failed to get device count: {e}")

    # Check PyTorch CUDA
    if HAS_TORCH:
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA device: {torch.cuda.get_device_name(0)}")

    # Test config creation
    try:
        config = GpuLoaderConfig(
            num_streams=4,
            max_pinned_memory_mb=512,
            device_id=0,
            double_buffer=True
        )
        print(f"✓ Config created: {config}")

        # hp_config = GpuLoaderConfig.high_performance()
        hp_config = GpuLoaderConfig(
            num_streams=32,
            max_pinned_memory_mb=1024,
            device_id=0,
            double_buffer=True
        )
        print(f"✓ High-perf config (32 streams): {hp_config}")

        max_config = GpuLoaderConfig(
            num_streams=64,
            max_pinned_memory_mb=2048,
            device_id=0,
            double_buffer=True
        )
        print(f"✓ Max-perf config (64 streams): {max_config}")
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

    return True


def main():
    """Main entry point."""
    print("="*60)
    print("Safetensors GPU Loading Benchmark")
    print("="*60)

    # Test GPU module
    gpu_ok = test_gpu_module()

    if not HAS_HF_HUB:
        print("\n⚠️  huggingface_hub not installed. Skipping download tests.")
        print("Install with: pip install huggingface_hub")
        return

    # Download and test with sample models
    cache_dir = Path.home() / ".cache" / "safetensors_test"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Start with the tiny model for quick testing
    tiny_model = SAMPLE_MODELS[0]  # tiny-random-bert

    print(f"\n\nDownloading test model: {tiny_model['description']}...")
    try:
        filepath = download_model(
            tiny_model["repo_id"],
            tiny_model["filename"],
            cache_dir=str(cache_dir)
        )
    except Exception as e:
        print(f"Failed to download model: {e}")

        # Try to find any existing .safetensors file
        existing = list(cache_dir.glob("**/*.safetensors"))
        if existing:
            filepath = str(existing[0])
            print(f"Using existing file: {filepath}")
        else:
            print("No safetensor files available for testing.")
            return

    # Run benchmarks
    run_comparison_benchmark(filepath, tiny_model["description"])

    # Optionally test with larger models
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\n\nRunning full benchmark suite (small models)...")
        for model in SAMPLE_MODELS[1:3]:  # BERT and GPT-2
            try:
                filepath = download_model(
                    model["repo_id"],
                    model["filename"],
                    cache_dir=str(cache_dir)
                )
                run_comparison_benchmark(filepath, model["description"])
            except Exception as e:
                print(f"Skipping {model['repo_id']}: {e}")

    # Test with mid-sized and large models
    if len(sys.argv) > 1 and sys.argv[1] == "--large":
        print("\n\nRunning large model benchmark suite...")
        for model in SAMPLE_MODELS[3:]:  # Llama-2 and Mistral
            try:
                filepath = download_model(
                    model["repo_id"],
                    model["filename"],
                    cache_dir=str(cache_dir)
                )
                run_comparison_benchmark(filepath, model["description"])
            except Exception as e:
                print(f"Skipping {model['repo_id']}: {e}")

    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
