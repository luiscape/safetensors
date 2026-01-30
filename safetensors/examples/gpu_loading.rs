//! Example demonstrating GPU loading of safetensors.
//!
//! This example shows how to use the GPU loading module to efficiently
//! load model weights directly to GPU memory using CUDA streams and
//! pinned memory pools.
//!
//! # Running this example
//!
//! ```bash
//! cargo run --example gpu_loading --features gpu
//! ```
//!
//! Note: In mock mode (without CUDA), this example simulates GPU operations
//! using regular CPU memory, which is useful for testing and development.

use safetensors::gpu::{GpuLoader, GpuLoaderConfig, GpuTensor};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{serialize, SafeTensors};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Safetensors GPU Loading Example ===\n");

    // Create some sample tensors to demonstrate loading
    let sample_data = create_sample_model()?;
    println!(
        "Created sample model with {} bytes of data\n",
        sample_data.len()
    );

    // Example 1: Basic loading with default configuration
    println!("--- Example 1: Basic Loading ---");
    basic_loading(&sample_data)?;

    // Example 2: Custom configuration for better performance
    println!("\n--- Example 2: Custom Configuration ---");
    custom_config_loading(&sample_data)?;

    // Example 3: Loading specific tensors
    println!("\n--- Example 3: Selective Loading ---");
    selective_loading(&sample_data)?;

    // Example 4: Loading with progress tracking
    println!("\n--- Example 4: Progress Tracking ---");
    progress_tracking(&sample_data)?;

    // Example 5: Resource management
    println!("\n--- Example 5: Resource Management ---");
    resource_management(&sample_data)?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

/// Create a sample model with various tensor shapes and dtypes.
fn create_sample_model() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Simulate a simple neural network with various layers
    let mut tensors: HashMap<String, TensorView<'static>> = HashMap::new();

    // Embedding layer (vocab_size=1000, embed_dim=256)
    static EMBEDDING_DATA: [u8; 1024000] = [0u8; 1000 * 256 * 4]; // F32
    let embedding = TensorView::new(Dtype::F32, vec![1000, 256], &EMBEDDING_DATA)?;
    tensors.insert("embedding.weight".to_string(), embedding);

    // Attention weights
    static ATTN_Q_DATA: [u8; 262144] = [0u8; 256 * 256 * 4];
    let attn_q = TensorView::new(Dtype::F32, vec![256, 256], &ATTN_Q_DATA)?;
    tensors.insert("attention.query.weight".to_string(), attn_q);

    static ATTN_K_DATA: [u8; 262144] = [0u8; 256 * 256 * 4];
    let attn_k = TensorView::new(Dtype::F32, vec![256, 256], &ATTN_K_DATA)?;
    tensors.insert("attention.key.weight".to_string(), attn_k);

    static ATTN_V_DATA: [u8; 262144] = [0u8; 256 * 256 * 4];
    let attn_v = TensorView::new(Dtype::F32, vec![256, 256], &ATTN_V_DATA)?;
    tensors.insert("attention.value.weight".to_string(), attn_v);

    // Feed-forward layer
    static FF1_DATA: [u8; 1048576] = [0u8; 256 * 1024 * 4];
    let ff1 = TensorView::new(Dtype::F32, vec![256, 1024], &FF1_DATA)?;
    tensors.insert("ffn.linear1.weight".to_string(), ff1);

    static FF2_DATA: [u8; 1048576] = [0u8; 1024 * 256 * 4];
    let ff2 = TensorView::new(Dtype::F32, vec![1024, 256], &FF2_DATA)?;
    tensors.insert("ffn.linear2.weight".to_string(), ff2);

    // Bias tensors (smaller)
    static BIAS1_DATA: [u8; 4096] = [0u8; 1024 * 4];
    let bias1 = TensorView::new(Dtype::F32, vec![1024], &BIAS1_DATA)?;
    tensors.insert("ffn.linear1.bias".to_string(), bias1);

    static BIAS2_DATA: [u8; 1024] = [0u8; 256 * 4];
    let bias2 = TensorView::new(Dtype::F32, vec![256], &BIAS2_DATA)?;
    tensors.insert("ffn.linear2.bias".to_string(), bias2);

    // Serialize to safetensors format
    let data = serialize(&tensors, None)?;
    Ok(data)
}

/// Example 1: Basic loading with default configuration.
fn basic_loading(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    // Create loader with default settings
    let loader = GpuLoader::with_defaults()?;

    let start = Instant::now();
    let tensors = loader.load_from_buffer(data)?;
    let elapsed = start.elapsed();

    println!("Loaded {} tensors in {:?}", tensors.len(), elapsed);
    print_tensor_summary(&tensors);

    Ok(())
}

/// Example 2: Custom configuration for production use.
fn custom_config_loading(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    // Configure for high-performance loading
    let config = GpuLoaderConfig::builder()
        .num_streams(8) // More streams for parallel transfers
        .max_pinned_memory_mb(1024) // 1GB pinned memory pool
        .double_buffer(true) // Enable double buffering
        .prefetch_size_mb(128) // Prefetch 128MB ahead
        .device_id(0) // Target GPU 0
        .build();

    println!("Configuration:");
    println!("  Streams: {}", config.num_streams);
    println!("  Pinned memory: {} MB", config.max_pinned_memory >> 20);
    println!("  Double buffering: {}", config.double_buffer);

    let loader = GpuLoader::new(config)?;

    let start = Instant::now();
    let tensors = loader.load_from_buffer(data)?;
    let elapsed = start.elapsed();

    println!("Loaded {} tensors in {:?}", tensors.len(), elapsed);

    // Print loader statistics
    let stats = loader.stats();
    println!("\nLoader statistics:");
    println!("  Active streams: {}", stats.num_streams);
    println!(
        "  Pinned memory allocated: {} KB",
        stats.pinned_memory_allocated >> 10
    );
    println!("  Free buffers in pool: {}", stats.free_pinned_buffers);

    Ok(())
}

/// Example 3: Loading only specific tensors.
fn selective_loading(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let loader = GpuLoader::with_defaults()?;

    // Parse the safetensors to see what's available
    let safetensors = SafeTensors::deserialize(data)?;
    println!("Available tensors:");
    for name in safetensors.names() {
        let info = safetensors.tensor(name)?;
        println!("  {} - {:?} {:?}", name, info.dtype(), info.shape());
    }

    // Load only the attention weights
    let names_to_load = vec![
        "attention.query.weight",
        "attention.key.weight",
        "attention.value.weight",
    ];

    println!("\nLoading only attention weights...");
    let start = Instant::now();
    let tensors = loader.load_tensors(&safetensors, &names_to_load)?;
    let elapsed = start.elapsed();

    println!(
        "Loaded {} tensors (out of {}) in {:?}",
        tensors.len(),
        safetensors.len(),
        elapsed
    );

    Ok(())
}

/// Example 4: Loading with progress tracking.
fn progress_tracking(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    // For this example, we'll write to a temp file since load_file_with_progress
    // requires a file path
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("example_model.safetensors");
    std::fs::write(&temp_file, data)?;

    let loader = GpuLoader::with_defaults()?;

    println!("Loading from file with progress tracking...");
    let start = Instant::now();
    let (tensors, handle) = loader.load_file_with_progress(&temp_file)?;
    let elapsed = start.elapsed();

    // In a real application, you would check progress in a loop:
    // while !handle.is_complete() {
    //     let progress = handle.progress();
    //     println!("Progress: {:.1}%", progress.percentage());
    //     std::thread::sleep(std::time::Duration::from_millis(100));
    // }

    let final_progress = handle.progress();
    println!("Final progress: {:.1}%", final_progress.percentage());
    println!(
        "Transferred {} bytes across {} tensors",
        final_progress.transferred_bytes, final_progress.loaded_tensors
    );
    println!("Loaded {} tensors in {:?}", tensors.len(), elapsed);

    // Clean up
    std::fs::remove_file(&temp_file)?;

    Ok(())
}

/// Example 5: Resource management and cleanup.
fn resource_management(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let loader = GpuLoader::with_defaults()?;

    // Load tensors
    let tensors = loader.load_from_buffer(data)?;
    println!("Loaded {} tensors", tensors.len());

    let stats_before = loader.stats();
    println!(
        "Memory before cleanup: {} KB pinned, {} buffers",
        stats_before.pinned_memory_allocated >> 10,
        stats_before.free_pinned_buffers
    );

    // Release cached buffers to free memory
    loader.release_caches();

    let stats_after = loader.stats();
    println!(
        "Memory after cleanup: {} KB pinned, {} buffers",
        stats_after.pinned_memory_allocated >> 10,
        stats_after.free_pinned_buffers
    );

    // Ensure all GPU operations are complete
    loader.synchronize()?;
    println!("All GPU operations synchronized");

    // Tensors are automatically freed when they go out of scope
    drop(tensors);
    println!("GPU memory freed");

    Ok(())
}

/// Print a summary of loaded tensors.
fn print_tensor_summary(tensors: &HashMap<String, GpuTensor>) {
    let total_bytes: usize = tensors.values().map(|t: &GpuTensor| t.size_bytes()).sum();
    println!("Total GPU memory used: {} KB", total_bytes >> 10);

    println!("\nTensor details:");
    for (name, tensor) in tensors {
        println!(
            "  {} - {:?} {:?} ({} bytes)",
            name,
            tensor.dtype(),
            tensor.shape(),
            tensor.size_bytes()
        );
    }
}
