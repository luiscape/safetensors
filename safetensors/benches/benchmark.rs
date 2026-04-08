use criterion::{criterion_group, criterion_main, Criterion};
use safetensors::tensor::*;
use std::collections::HashMap;
use std::hint::black_box;

#[cfg(feature = "fast_io")]
use safetensors::bulk_io::{BulkReadPlan, PreadBulkReader};

// Returns a sample data of size 2_MB
fn get_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![1000, 500];
    let dtype = Dtype::F32;
    let nbits = shape.iter().product::<usize>() * dtype.bitsize();
    assert!(nbits % 8 == 0);
    let n: usize = nbits / 8; // 4
    let data = vec![0; n];

    (data, shape, dtype)
}

pub fn bench_serialize(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // 2_MB x 5 = 10_MB
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    c.bench_function("Serialize 10_MB", |b| {
        b.iter(|| {
            let _serialized = serialize(black_box(&metadata), black_box(None));
        })
    });
}

pub fn bench_deserialize(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // 2_MB x 5 = 10_MB
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    let out = serialize(&metadata, None).unwrap();

    c.bench_function("Deserialize 10_MB", |b| {
        b.iter(|| {
            let _deserialized = SafeTensors::deserialize(black_box(&out)).unwrap();
        })
    });
}

// ---------------------------------------------------------------------------
// fast_io benchmarks (gated behind the "fast_io" feature)
// ---------------------------------------------------------------------------

/// Build a GPT-2-like set of tensor views for benchmarking.
///
/// 12 transformer layers, each with 4 weight matrices (q, k, v, o projections)
/// plus 2 MLP weights, plus embeddings. Results in ~30 MB of data and ~50+
/// tensors — a realistic metadata size.
#[cfg(feature = "fast_io")]
fn build_gpt2_model() -> (Vec<(String, Vec<u8>, Vec<usize>, Dtype)>,) {
    let dtype = Dtype::F32;
    let bytes_per_f32 = 4usize;
    let hidden = 768;
    let intermediate = 3072;
    let vocab = 50257;
    let max_pos = 1024;

    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();

    // Token + position embeddings
    let shape = vec![vocab, hidden];
    let n = shape.iter().product::<usize>() * bytes_per_f32;
    tensors.push(("wte.weight".into(), vec![0u8; n], shape, dtype));

    let shape = vec![max_pos, hidden];
    let n = shape.iter().product::<usize>() * bytes_per_f32;
    tensors.push(("wpe.weight".into(), vec![0u8; n], shape, dtype));

    for layer in 0..12 {
        // Attention QKV + output projection
        for name in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            let shape = vec![hidden, hidden];
            let n = shape.iter().product::<usize>() * bytes_per_f32;
            tensors.push((
                format!("h.{layer}.attn.{name}.weight"),
                vec![0u8; n],
                shape,
                dtype,
            ));
        }
        // MLP
        let shape_up = vec![hidden, intermediate];
        let n_up = shape_up.iter().product::<usize>() * bytes_per_f32;
        tensors.push((
            format!("h.{layer}.mlp.fc1.weight"),
            vec![0u8; n_up],
            shape_up,
            dtype,
        ));

        let shape_down = vec![intermediate, hidden];
        let n_down = shape_down.iter().product::<usize>() * bytes_per_f32;
        tensors.push((
            format!("h.{layer}.mlp.fc2.weight"),
            vec![0u8; n_down],
            shape_down,
            dtype,
        ));

        // Layer norms (small, but realistic)
        for ln in &["ln_1", "ln_2"] {
            let shape = vec![hidden];
            let n = shape.iter().product::<usize>() * bytes_per_f32;
            tensors.push((
                format!("h.{layer}.{ln}.weight"),
                vec![0u8; n],
                shape.clone(),
                dtype,
            ));
            tensors.push((format!("h.{layer}.{ln}.bias"), vec![0u8; n], shape, dtype));
        }
    }

    (tensors,)
}

/// Helper: serialize the GPT-2-like model to a temporary file and return the
/// path together with metadata needed for reading it back.
#[cfg(feature = "fast_io")]
fn create_temp_model_file() -> (std::path::PathBuf, usize, usize) {
    let (tensors,) = build_gpt2_model();

    // Build TensorView references
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(name, data, shape, dtype)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data.as_slice()).unwrap(),
            )
        })
        .collect();

    let tmp_dir = std::env::temp_dir();
    let path = tmp_dir.join("safetensors_bench_gpt2.safetensors");

    serialize_to_file(
        views.iter().map(|(n, v)| (n.as_str(), v)),
        None::<HashMap<String, String>>,
        &path,
    )
    .expect("failed to serialize model to temp file");

    // Read back header to determine sizes
    let file_bytes = std::fs::read(&path).expect("failed to read temp file");
    let (n, metadata) = SafeTensors::read_metadata(&file_bytes).unwrap();
    let header_size = 8 + n; // 8-byte length prefix + JSON header
    let body_length = metadata.data_len();

    (path, header_size, body_length)
}

/// Benchmark the overhead of creating a `BulkReadPlan` from realistic metadata.
#[cfg(feature = "fast_io")]
pub fn bench_bulk_read_plan(c: &mut Criterion) {
    // Simulate a large model: 200 tensors, ~2 GB body
    let header_size = 32_768; // 32 KB header (realistic for a large model)
    let body_size = 2_000_000_000; // 2 GB body
    let max_block = Some(16 * 1024 * 1024); // 16 MiB blocks

    let mut group = c.benchmark_group("BulkReadPlan::new");

    group.bench_function("2GB body / 16MiB blocks", |b| {
        b.iter(|| {
            let plan = BulkReadPlan::new(
                black_box(header_size),
                black_box(body_size),
                black_box(max_block),
            );
            black_box(&plan);
        })
    });

    // Smaller block size → more blocks in the plan
    let max_block_small = Some(1024 * 1024); // 1 MiB blocks
    group.bench_function("2GB body / 1MiB blocks", |b| {
        b.iter(|| {
            let plan = BulkReadPlan::new(
                black_box(header_size),
                black_box(body_size),
                black_box(max_block_small),
            );
            black_box(&plan);
        })
    });

    // Tiny body (edge case)
    group.bench_function("4KB body / default blocks", |b| {
        b.iter(|| {
            let plan = BulkReadPlan::new(black_box(header_size), black_box(4096), black_box(None));
            black_box(&plan);
        })
    });

    group.finish();
}

/// Benchmark parallel file I/O with `PreadBulkReader` at different thread counts.
#[cfg(feature = "fast_io")]
pub fn bench_pread_parallel(c: &mut Criterion) {
    use std::os::unix::io::AsRawFd;

    let (path, header_size, body_length) = create_temp_model_file();

    let file = std::fs::File::open(&path).expect("failed to open temp file for bench");
    let fd = file.as_raw_fd();

    let mut group = c.benchmark_group("PreadBulkReader::read_to_buffer");
    // The model is ~30 MB; set a reasonable sample size so the benchmark
    // doesn't take forever.
    group.sample_size(20);

    for &num_threads in &[1, 4, 8, 16] {
        let plan = BulkReadPlan::new(header_size, body_length, None);
        let reader = PreadBulkReader::new(None, Some(num_threads));
        let mut buf = vec![0u8; body_length];

        group.bench_function(format!("{num_threads} threads"), |b| {
            b.iter(|| {
                reader
                    .read_to_buffer(black_box(fd), black_box(&plan), black_box(&mut buf))
                    .expect("pread failed");
            })
        });
    }

    group.finish();

    // Clean up
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// Sequential baseline: read the same file body with a single `std::fs::File::read`.
/// This gives a baseline to compare the parallel `pread` benchmarks against.
#[cfg(feature = "fast_io")]
pub fn bench_read_sequential_baseline(c: &mut Criterion) {
    use std::io::Read;
    use std::io::Seek;

    let (path, header_size, body_length) = create_temp_model_file();

    let mut group = c.benchmark_group("Sequential baseline read");
    group.sample_size(20);

    let mut buf = vec![0u8; body_length];

    group.bench_function("std::fs::File::read", |b| {
        b.iter(|| {
            let mut file = std::fs::File::open(&path).expect("failed to open temp file for bench");
            file.seek(std::io::SeekFrom::Start(header_size as u64))
                .expect("seek failed");
            file.read_exact(black_box(&mut buf))
                .expect("sequential read failed");
        })
    });

    group.finish();

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Criterion groups & main
// ---------------------------------------------------------------------------

criterion_group!(bench_ser, bench_serialize);
criterion_group!(bench_de, bench_deserialize);

#[cfg(feature = "fast_io")]
criterion_group!(
    bench_fast_io,
    bench_bulk_read_plan,
    bench_pread_parallel,
    bench_read_sequential_baseline
);

#[cfg(feature = "fast_io")]
criterion_main!(bench_ser, bench_de, bench_fast_io);

#[cfg(not(feature = "fast_io"))]
criterion_main!(bench_ser, bench_de);
