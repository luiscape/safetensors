use criterion::{criterion_group, criterion_main, Criterion};
use safetensors::tensor::*;
use std::collections::HashMap;
use std::hint::black_box;

#[cfg(feature = "fast_io")]
use safetensors::bulk_io::{BulkReadPlan, PreadBulkReader};

#[cfg(feature = "compression")]
use safetensors::bulk_io::CompressedReadPlan;

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
// Compression benchmarks (gated behind "compression" + "fast_io" features)
// ---------------------------------------------------------------------------

/// Creates a temp model file and its compressed variant for benchmarking.
#[cfg(all(feature = "compression", feature = "fast_io"))]
fn create_compressed_temp_file() -> (
    std::path::PathBuf, // uncompressed path
    std::path::PathBuf, // compressed path
    usize,              // header_size (of compressed file)
    usize,              // uncompressed body_length
    usize,              // compressed body_length
) {
    use safetensors::tensor::{CompressedChunk, CompressionInfo, CompressionMethod};
    use std::io::Write;

    let (uncompressed_path, header_size, body_length) = create_temp_model_file();

    // Read the uncompressed file
    let file_bytes = std::fs::read(&uncompressed_path).expect("failed to read temp file");
    let header_bytes = &file_bytes[8..header_size];
    let body_bytes = &file_bytes[header_size..];
    assert_eq!(body_bytes.len(), body_length);

    // Compress the body in 16 MiB chunks
    let chunk_size: usize = 16 * 1024 * 1024;
    let mut chunks = Vec::new();
    let mut compressed_body = Vec::new();
    let mut compressed_offset = 0usize;

    let mut pos = 0usize;
    while pos < body_length {
        let end = (pos + chunk_size).min(body_length);
        let chunk_data = &body_bytes[pos..end];
        let compressed_chunk = zstd::encode_all(chunk_data, 3).expect("zstd compress failed");
        chunks.push(CompressedChunk {
            compressed_offset,
            compressed_size: compressed_chunk.len(),
            decompressed_offset: pos,
            decompressed_size: chunk_data.len(),
        });
        compressed_offset += compressed_chunk.len();
        compressed_body.extend_from_slice(&compressed_chunk);
        pos = end;
    }

    let compression_info = CompressionInfo {
        method: CompressionMethod::Zstd,
        level: 3,
        decompressed_size: body_length,
        compressed_size: compressed_body.len(),
        chunk_size,
        chunks,
    };

    // Build new header with compression metadata
    let mut header: serde_json::Value = serde_json::from_slice(header_bytes).unwrap();
    let meta = header
        .as_object_mut()
        .unwrap()
        .entry("__metadata__")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .unwrap();
    meta.insert("compression".to_string(), serde_json::json!("zstd"));
    meta.insert(
        "compression_info".to_string(),
        serde_json::Value::String(serde_json::to_string(&compression_info).unwrap()),
    );

    let new_header_json = serde_json::to_string(&header).unwrap();
    let mut padded_header = new_header_json.into_bytes();
    let aligned_len = padded_header.len().next_multiple_of(8);
    padded_header.resize(aligned_len, b' ');

    let compressed_path = uncompressed_path.with_extension("safetensors.zst");
    let mut file =
        std::fs::File::create(&compressed_path).expect("failed to create compressed file");
    file.write_all(&(padded_header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&padded_header).unwrap();
    file.write_all(&compressed_body).unwrap();

    let compressed_body_len = compressed_body.len();
    let new_header_size = 8 + padded_header.len();

    (
        uncompressed_path,
        compressed_path,
        new_header_size,
        body_length,
        compressed_body_len,
    )
}

/// Benchmark zstd compression at different levels (CPU).
#[cfg(all(feature = "compression", feature = "fast_io"))]
pub fn bench_zstd_compress(c: &mut Criterion) {
    let (tensors,) = build_gpt2_model();
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(name, data, shape, dtype)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data.as_slice()).unwrap(),
            )
        })
        .collect();
    let metadata_map: HashMap<String, TensorView> = views.into_iter().collect();
    let serialized = serialize(&metadata_map, None).unwrap();
    let header_len = {
        let n = u64::from_le_bytes(serialized[..8].try_into().unwrap()) as usize;
        8 + n
    };
    let body = serialized[header_len..].to_vec();

    let mut group = c.benchmark_group("zstd compress (CPU)");
    group.sample_size(20);

    for level in [1, 3, 5] {
        group.bench_function(
            format!("level {level} / {:.1} MB", body.len() as f64 / 1e6),
            |b| {
                b.iter(|| {
                    let _compressed = zstd::encode_all(black_box(body.as_slice()), level).unwrap();
                })
            },
        );
    }

    group.finish();
}

/// Benchmark zstd decompression (CPU).
#[cfg(all(feature = "compression", feature = "fast_io"))]
pub fn bench_zstd_decompress(c: &mut Criterion) {
    let (uncompressed_path, compressed_path, header_size, body_length, compressed_body_length) =
        create_compressed_temp_file();

    // Read the compressed body
    let file_bytes = std::fs::read(&compressed_path).expect("failed to read compressed file");
    let compressed_body = file_bytes[header_size..].to_vec();
    assert_eq!(compressed_body.len(), compressed_body_length);

    let mut group = c.benchmark_group("zstd decompress (CPU)");
    group.sample_size(20);

    let ratio = body_length as f64 / compressed_body_length as f64;
    group.bench_function(
        format!(
            "{:.1} MB -> {:.1} MB ({:.2}x ratio)",
            compressed_body_length as f64 / 1e6,
            body_length as f64 / 1e6,
            ratio
        ),
        |b| {
            b.iter(|| {
                let _decompressed =
                    zstd::decode_all(black_box(compressed_body.as_slice())).unwrap();
            })
        },
    );

    group.finish();

    // Print compression stats
    eprintln!(
        "\n[Compression stats] Uncompressed: {:.2} MB, Compressed: {:.2} MB, Ratio: {:.2}x\n",
        body_length as f64 / 1e6,
        compressed_body_length as f64 / 1e6,
        ratio,
    );

    // Clean up
    let _ = std::fs::remove_file(&compressed_path);
    let _ = std::fs::remove_file(&uncompressed_path);
}

/// Benchmark CompressedReadPlan creation.
#[cfg(all(feature = "compression", feature = "fast_io"))]
pub fn bench_compressed_read_plan(c: &mut Criterion) {
    use safetensors::tensor::{CompressedChunk, CompressionInfo, CompressionMethod};

    let mut group = c.benchmark_group("CompressedReadPlan::new");

    // Simulate a large compressed model: 2 GB decompressed, ~1.3 GB compressed, 16 MiB chunks
    let chunk_size = 16 * 1024 * 1024usize;
    let decompressed_size = 2_000_000_000usize;
    let num_chunks = (decompressed_size + chunk_size - 1) / chunk_size;
    let avg_compressed_chunk = 10_500_000usize;

    let chunks: Vec<CompressedChunk> = (0..num_chunks)
        .map(|i| {
            let decomp_off = i * chunk_size;
            let decomp_sz = (decompressed_size - decomp_off).min(chunk_size);
            let comp_off = i * avg_compressed_chunk;
            let comp_sz = (decomp_sz as f64 * 0.65) as usize;
            CompressedChunk {
                compressed_offset: comp_off,
                compressed_size: comp_sz,
                decompressed_offset: decomp_off,
                decompressed_size: decomp_sz,
            }
        })
        .collect();

    let info = CompressionInfo {
        method: CompressionMethod::Zstd,
        level: 3,
        decompressed_size,
        compressed_size: chunks.iter().map(|c| c.compressed_size).sum(),
        chunk_size,
        chunks,
    };

    group.bench_function(format!("2GB decompressed / {} chunks", num_chunks), |b| {
        b.iter(|| {
            let plan = CompressedReadPlan::new(black_box(32768), black_box(&info));
            black_box(&plan);
        })
    });

    group.finish();
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

#[cfg(all(feature = "compression", feature = "fast_io"))]
criterion_group!(
    bench_compression,
    bench_zstd_compress,
    bench_zstd_decompress,
    bench_compressed_read_plan
);

#[cfg(all(feature = "fast_io", feature = "compression"))]
criterion_main!(bench_ser, bench_de, bench_fast_io, bench_compression);

#[cfg(all(feature = "fast_io", not(feature = "compression")))]
criterion_main!(bench_ser, bench_de, bench_fast_io);

#[cfg(all(not(feature = "fast_io"), feature = "compression"))]
criterion_main!(bench_ser, bench_de);

#[cfg(all(not(feature = "fast_io"), not(feature = "compression")))]
criterion_main!(bench_ser, bench_de);
