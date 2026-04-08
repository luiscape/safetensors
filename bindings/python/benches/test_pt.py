import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

try:
    from safetensors.fast import FastSafeTensorsLoader, fast_load_file, fast_open

    _has_fast = True
except ImportError:
    _has_fast = False


def create_gpt2(n_layers: int):
    tensors = {}
    tensors["wte"] = torch.zeros((50257, 768))
    tensors["wpe"] = torch.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((768))
    tensors["ln_f.weight"] = torch.zeros((768))
    tensors["ln_f.bias"] = torch.zeros((768))
    return tensors


def create_lora(n_layers: int):
    tensors = {}
    for i in range(n_layers):
        tensors[f"lora.{i}.up.weight"] = torch.zeros((32, 32))
        tensors[f"lora.{i}.down.weight"] = torch.zeros((32, 32))
    return tensors


def test_pt_pt_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_pt_load_cpu_small(benchmark):
    weights = create_lora(500)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_small(benchmark):
    weights = create_lora(500)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_pt_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_pt_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_sf_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
def test_pt_fast_load_cpu(benchmark):
    """Fast path: bulk I/O with parallel pread on CPU."""
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(fast_load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
def test_pt_fast_load_cpu_small(benchmark):
    """Fast path: bulk I/O on CPU with small LoRA model."""
    weights = create_lora(500)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(fast_load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_fast_load_gpu(benchmark):
    """Fast path: bulk I/O with bounce buffer on GPU."""
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(fast_load_file, f.name, device="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
def test_pt_fast_load_cpu_async(benchmark):
    """Fast path with async submit/wait pattern (multi-file overlap)."""
    weights = create_gpt2(12)

    # Split weights into 3 shards by layer range
    shard0 = {}
    shard1 = {}
    shard2 = {}
    for k, v in weights.items():
        if k.startswith("h."):
            layer_num = int(k.split(".")[1])
            if layer_num < 4:
                shard0[k] = v
            elif layer_num < 8:
                shard1[k] = v
            else:
                shard2[k] = v
        else:
            # Non-layer tensors (wte, wpe, ln_f) go to shard0
            shard0[k] = v

    filenames = []
    for shard in [shard0, shard1, shard2]:
        f = tempfile.NamedTemporaryFile(delete=False)
        save_file(shard, f.name)
        filenames.append(f.name)
        f.close()

    def load_async():
        loader = FastSafeTensorsLoader(device="cpu")
        loader.add_filenames({0: filenames})
        buf = loader.copy_files_to_device()
        result = {}
        for key in buf.keys():
            result[key] = buf.get_tensor(key).clone()
        buf.close()
        loader.close()
        return result

    result = benchmark(load_async)

    for fname in filenames:
        os.unlink(fname)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
def test_pt_fast_load_cpu_large(benchmark):
    """Fast path: larger GPT-2 model (48 layers, ~120 MB)."""
    weights = create_gpt2(48)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(fast_load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_large(benchmark):
    """Baseline: safetensors load_file with larger GPT-2 model (48 layers)."""
    weights = create_gpt2(48)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not _has_fast, reason="safetensors.fast not available")
def test_pt_fast_open_cpu(benchmark):
    """Fast path: using fast_open context manager."""
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_via_fast_open():
            with fast_open(f.name, device="cpu") as fo:
                return {k: fo.get_tensor(k).clone() for k in fo.keys()}

        result = benchmark(load_via_fast_open)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_save_cpu(benchmark):
    weights = create_gpt2(12)

    filename = "tmp.safetensors"

    # XXX: On some platforms (tested on Linux x86_64 ext4), writing to an already existing file is slower than creating a new one.
    # On others, such as MacOS (APFS), it's the opposite. To have more consistent benchmarks,
    # we ensure the file does not exist before each write, which is also closer to real world usage.
    def setup():
        try:
            os.unlink(filename)
        except Exception:
            pass

    benchmark.pedantic(
        save_file, args=(weights, filename), setup=setup, iterations=1, rounds=5
    )

    # Clean up files
    os.unlink(filename)
