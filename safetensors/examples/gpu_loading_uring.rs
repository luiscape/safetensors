//! Example demonstrating io_uring-enhanced GPU loading.
//!
//! This example shows how to use the io_uring-based loader for maximum
//! I/O throughput when loading safetensor files to GPU memory on Linux.
//!
//! # Features
//!
//! - io_uring async I/O with batched submissions
//! - Optional SQPOLL for kernel-side polling (reduced syscalls)
//! - Optional O_DIRECT for bypassing page cache
//! - Pipelined loading: overlaps I/O with GPU transfers
//!
//! # Running this example
//!
//! ```bash
//! # Basic io_uring support
//! cargo run --example gpu_loading_uring --features io-uring
//!
//! # Without io_uring feature (falls back to standard I/O)
//! cargo run --example gpu_loading_uring --features gpu
//! ```
//!
//! Note: io_uring requires Linux kernel 5.1+. On other platforms or older
//! kernels, the loader automatically falls back to standard I/O.

#[cfg(feature = "io-uring")]
use safetensors::gpu::{
    AlignedBuffer, IoUringConfig, UringLoader, UringLoaderConfig,
};
use safetensors::gpu::GpuLoader;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Safetensors io_uring GPU Loading Example ===\n");

    // Check io_uring availability
    #[cfg(feature = "io-uring")]
    {
        use safetensors::gpu::io_uring::is_available;
        println!("io_uring available: {}", is_available());
        if !is_available() {
            println!("Note: io_uring not available, falling back to standard I/O\n");
        }
    }
    #[cfg(not(feature = "io-uring"))]
    {
        println!("io_uring feature not enabled, using standard I/O\n");
    }

    // Create sample model
    let sample_data = create_sample_model()?;
    println!(
        "Created sample model with {} bytes of data\n",
        sample_data.len()
    );

    // Example 1: Compare standard vs io_uring loading
    println!("--- Example 1: Standard vs io_uring Loading ---");
    compare_loading_methods(&sample_data)?;

    // Example 2: io_uring configuration options
    #[cfg(feature = "io-uring")]
    {
        println!("\n--- Example 2: io_uring Configuration ---");
        demonstrate_io_uring_config()?;
    }

    // Example 3: Aligned buffer usage
    #[cfg(feature = "io-uring")]
    {
        println!("\n--- Example 3: Aligned Buffers ---");
        demonstrate_aligned_buffers()?;
    }

    // Example 4: Pipelined loading
    #[cfg(feature = "io-uring")]
    {
        println!("\n--- Example 4: Pipelined Loading ---");
        demonstrate_pipelined_loading(&sample_data)?;
    }

    // Example 5: High-performance configuration
    #[cfg(feature = "io-uring")]
    {
        println!("\n--- Example 5: High-Performance Configuration ---");
        high_performance_loading(&sample_data)?;
    }

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Create a sample model with various tensor shapes.
fn create_sample_model() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut tensors: HashMap<String, TensorView<'static>> = HashMap::new();

    // Create several tensors of varying sizes
    static SMALL_DATA: [u8; 4096] = [0u8; 1024 * 4]; // 1K floats
    static MEDIUM_DATA: [u8; 262144] = [0u8; 256 * 256 * 4]; // 64K floats
    static LARGE_DATA: [u8; 1048576] = [0u8; 256 * 1024 * 4]; // 256K floats

    let small = TensorView::new(Dtype::F32, vec![1024], &SMALL_DATA)?;
    tensors.insert("layer.0.bias".to_string(), small);

    let medium = TensorView::new(Dtype::F32, vec![256, 256], &MEDIUM_DATA)?;
    tensors.insert("layer.0.weight".to_string(), medium);

    let large = TensorView::new(Dtype::F32, vec![256, 1024], &LARGE_DATA)?;
    tensors.insert("layer.1.weight".to_string(), large);

    static LARGE_DATA2: [u8; 1048576] = [0u8; 1024 * 256 * 4];
    let large2 = TensorView::new(Dtype::F32, vec![1024, 256], &LARGE_DATA2)?;
    tensors.insert("layer.2.weight".to_string(), large2);

    static SMALL_DATA2: [u8; 16384] = [0u8; 4096 * 4];
    let small2 = TensorView::new(Dtype::F32, vec![4096], &SMALL_DATA2)?;
    tensors.insert("layer.2.bias".to_string(), small2);

    let data = serialize(&tensors, None)?;
    Ok(data)
}

/// Compare standard loading with io_uring-enhanced loading.
fn compare_loading_methods(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    // Write to temp file for file-based loading tests
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("uring_test_model.safetensors");
    std::fs::write(&temp_file, data)?;

    // Standard loading with GpuLoader
    let standard_loader = GpuLoader::with_defaults()?;
    let start = Instant::now();
    let tensors = standard_loader.load_file(&temp_file)?;
    let standard_time = start.elapsed();
    println!(
        "Standard loading: {} tensors in {:?}",
        tensors.len(),
        standard_time
    );
    drop(tensors);

    // io_uring-enhanced loading (if available)
    #[cfg(feature = "io-uring")]
    {
        let uring_loader = UringLoader::with_defaults()?;
        let start = Instant::now();
        let tensors = uring_loader.load_file(&temp_file)?;
        let uring_time = start.elapsed();
        println!(
            "io_uring loading: {} tensors in {:?}",
            tensors.len(),
            uring_time
        );

        if uring_loader.uses_io_uring() {
            let speedup = standard_time.as_secs_f64() / uring_time.as_secs_f64();
            println!("Speedup: {:.2}x", speedup);
        } else {
            println!("(io_uring not available, used fallback)");
        }
    }

    // Cleanup
    std::fs::remove_file(&temp_file)?;

    Ok(())
}

/// Demonstrate io_uring configuration options.
#[cfg(feature = "io-uring")]
fn demonstrate_io_uring_config() -> Result<(), Box<dyn std::error::Error>> {
    // Default configuration
    let default_config = IoUringConfig::default();
    println!("Default configuration:");
    println!("  Queue depth: {}", default_config.queue_depth);
    println!("  SQPOLL: {}", default_config.sq_poll);
    println!("  Direct I/O: {}", default_config.direct_io);
    println!("  Alignment: {} bytes", default_config.alignment);

    // High-performance configuration
    let hp_config = IoUringConfig::high_performance();
    println!("\nHigh-performance configuration:");
    println!("  Queue depth: {}", hp_config.queue_depth);
    println!("  SQPOLL: {}", hp_config.sq_poll);
    println!("  SQPOLL idle: {}ms", hp_config.sq_poll_idle_ms);
    println!("  Direct I/O: {}", hp_config.direct_io);
    println!("  Registered buffers: {}", hp_config.registered_buffers);
    println!("  Buffer size: {} bytes", hp_config.buffer_size);

    // Custom configuration using builder
    let custom_config = IoUringConfig::builder()
        .queue_depth(256)
        .sq_poll(false) // Disable SQPOLL (requires root or CAP_SYS_ADMIN)
        .direct_io(false) // Disable O_DIRECT for compatibility
        .alignment(4096)
        .registered_buffers(8, 1 << 20) // 8 x 1MB buffers
        .build();

    println!("\nCustom configuration:");
    println!("  Queue depth: {}", custom_config.queue_depth);
    println!("  SQPOLL: {}", custom_config.sq_poll);
    println!("  Direct I/O: {}", custom_config.direct_io);
    println!("  Registered buffers: {}", custom_config.registered_buffers);

    Ok(())
}

/// Demonstrate aligned buffer usage for direct I/O.
#[cfg(feature = "io-uring")]
fn demonstrate_aligned_buffers() -> Result<(), Box<dyn std::error::Error>> {
    // Create buffers with different alignments
    let buffer_4k = AlignedBuffer::new(8192, 4096)?;
    let buffer_512 = AlignedBuffer::new(1024, 512)?;

    println!("4KB-aligned buffer:");
    println!("  Capacity: {} bytes", buffer_4k.capacity());
    println!("  Is aligned: {}", buffer_4k.is_aligned());
    println!(
        "  Pointer alignment: {} (mod 4096 = {})",
        buffer_4k.as_ptr() as usize,
        (buffer_4k.as_ptr() as usize) % 4096
    );

    println!("\n512-byte aligned buffer:");
    println!("  Capacity: {} bytes", buffer_512.capacity());
    println!("  Is aligned: {}", buffer_512.is_aligned());
    println!(
        "  Pointer alignment: {} (mod 512 = {})",
        buffer_512.as_ptr() as usize,
        (buffer_512.as_ptr() as usize) % 512
    );

    // Demonstrate capacity rounding
    let buffer_uneven = AlignedBuffer::new(1000, 512)?;
    println!("\nBuffer for 1000 bytes with 512-byte alignment:");
    println!(
        "  Requested: 1000 bytes, Actual capacity: {} bytes",
        buffer_uneven.capacity()
    );
    println!(
        "  (Rounded up to next multiple of 512: {})",
        ((1000 + 511) / 512) * 512
    );

    Ok(())
}

/// Demonstrate pipelined loading.
#[cfg(feature = "io-uring")]
fn demonstrate_pipelined_loading(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("uring_pipeline_test.safetensors");
    std::fs::write(&temp_file, data)?;

    // Non-pipelined loading
    let config_no_pipeline = UringLoaderConfig::builder()
        .num_streams(4)
        .max_pinned_memory_mb(128)
        .pipeline(false)
        .build();

    let loader = UringLoader::new(config_no_pipeline)?;
    let start = Instant::now();
    let tensors = loader.load_file(&temp_file)?;
    let no_pipeline_time = start.elapsed();
    println!(
        "Without pipeline: {} tensors in {:?}",
        tensors.len(),
        no_pipeline_time
    );
    drop(tensors);

    // Pipelined loading
    let config_pipeline = UringLoaderConfig::builder()
        .num_streams(4)
        .max_pinned_memory_mb(128)
        .pipeline(true)
        .pipeline_stages(3)
        .build();

    let loader = UringLoader::new(config_pipeline)?;
    let start = Instant::now();
    let tensors = loader.load_file(&temp_file)?;
    let pipeline_time = start.elapsed();
    println!(
        "With pipeline (3 stages): {} tensors in {:?}",
        tensors.len(),
        pipeline_time
    );

    // Different pipeline depths
    for stages in [2, 4, 6] {
        let config = UringLoaderConfig::builder()
            .num_streams(4)
            .max_pinned_memory_mb(128)
            .pipeline(true)
            .pipeline_stages(stages)
            .build();

        let loader = UringLoader::new(config)?;
        let start = Instant::now();
        let tensors = loader.load_file(&temp_file)?;
        let time = start.elapsed();
        println!("Pipeline ({} stages): {:?}", stages, time);
        drop(tensors);
    }

    std::fs::remove_file(&temp_file)?;
    Ok(())
}

/// Demonstrate high-performance configuration.
#[cfg(feature = "io-uring")]
fn high_performance_loading(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("uring_hp_test.safetensors");
    std::fs::write(&temp_file, data)?;

    // Use high-performance preset
    let loader = UringLoader::high_performance()?;

    println!("High-performance loader configuration:");
    println!("  Streams: {}", loader.config().gpu.num_streams);
    println!(
        "  Pinned memory: {} MB",
        loader.config().gpu.max_pinned_memory >> 20
    );
    println!("  I/O concurrency: {}", loader.config().io_concurrency);
    println!("  Pipeline: {}", loader.config().pipeline);
    println!("  Pipeline stages: {}", loader.config().pipeline_stages);
    println!("  Uses io_uring: {}", loader.uses_io_uring());

    // Benchmark
    let iterations = 5;
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let tensors = loader.load_file(&temp_file)?;
        times.push(start.elapsed());
        drop(tensors);
    }

    let avg_time: std::time::Duration = times.iter().sum::<std::time::Duration>() / iterations as u32;
    let min_time = times.iter().min().unwrap();
    let max_time = times.iter().max().unwrap();

    println!("\nBenchmark ({} iterations):", iterations);
    println!("  Average: {:?}", avg_time);
    println!("  Min: {:?}", min_time);
    println!("  Max: {:?}", max_time);

    // Calculate throughput
    let file_size = data.len() as f64;
    let throughput_mb_s = (file_size / (1024.0 * 1024.0)) / avg_time.as_secs_f64();
    println!("  Throughput: {:.2} MB/s", throughput_mb_s);

    // Show final stats
    let stats = loader.stats();
    println!("\nLoader statistics:");
    println!("  Active streams: {}", stats.num_streams);
    println!(
        "  Pinned memory allocated: {} KB",
        stats.pinned_memory_allocated >> 10
    );
    println!("  Free pinned buffers: {}", stats.free_pinned_buffers);

    std::fs::remove_file(&temp_file)?;
    Ok(())
}
