//! Build script to auto-detect CUDA library paths for linking.
//!
//! This script searches for the CUDA runtime library (libcudart) in common
//! installation locations and adds the appropriate linker search paths.

fn main() {
    // Only run CUDA detection when the cuda feature is enabled
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Always add common CUDA paths first (these are most likely to work)
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.9/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda-12/lib64");

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");

    // Common CUDA installation paths to search
    let cuda_paths: Vec<Option<String>> = vec![
        // Environment variables (highest priority)
        std::env::var("CUDA_PATH").ok(),
        std::env::var("CUDA_HOME").ok(),
        std::env::var("CUDA_ROOT").ok(),
        // Standard Linux paths
        Some("/usr/local/cuda".to_string()),
        Some("/usr/local/cuda-12".to_string()),
        Some("/usr/local/cuda-12.9".to_string()),
        Some("/usr/local/cuda-12.8".to_string()),
        Some("/usr/local/cuda-12.6".to_string()),
        Some("/usr/local/cuda-12.4".to_string()),
        Some("/usr/local/cuda-12.2".to_string()),
        Some("/usr/local/cuda-11".to_string()),
        Some("/usr/local/cuda-11.8".to_string()),
        Some("/opt/cuda".to_string()),
        // NVIDIA HPC SDK
        Some("/opt/nvidia/hpc_sdk/Linux_x86_64/cuda".to_string()),
        // Conda environments
        std::env::var("CONDA_PREFIX").ok(),
    ];

    // Subdirectories where CUDA libraries might be located
    let lib_subdirs = [
        "lib64",
        "lib",
        "lib/x86_64-linux-gnu",
        // NVIDIA Docker images use targets directory
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
        // Stubs directory (for linking only, runtime uses real libs)
        "lib64/stubs",
        "lib/stubs",
    ];

    let mut found = false;

    for cuda_path in cuda_paths.into_iter().flatten() {
        if found {
            break;
        }

        for subdir in &lib_subdirs {
            let lib_path = format!("{}/{}", cuda_path, subdir);

            // Check for libcudart.so (shared library)
            let cudart_so = format!("{}/libcudart.so", lib_path);
            // Check for libcudart_static.a (static library)
            let cudart_static = format!("{}/libcudart_static.a", lib_path);

            if std::path::Path::new(&cudart_so).exists()
                || std::path::Path::new(&cudart_static).exists()
            {
                println!("cargo:rustc-link-search=native={}", lib_path);
                println!("cargo:warning=Found CUDA runtime at: {}", lib_path);
                found = true;
                break;
            }
        }

        // Also check if libcudart.so is directly in the cuda_path
        if !found {
            let direct_path = format!("{}/libcudart.so", cuda_path);
            if std::path::Path::new(&direct_path).exists() {
                println!("cargo:rustc-link-search=native={}", cuda_path);
                println!("cargo:warning=Found CUDA runtime at: {}", cuda_path);
                found = true;
            }
        }
    }

    // Also add system library paths that might contain CUDA
    let system_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    ];

    for path in &system_paths {
        let cudart_path = format!("{}/libcudart.so", path);
        if std::path::Path::new(&cudart_path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
            if !found {
                println!("cargo:warning=Found CUDA runtime at: {}", path);
                found = true;
            }
        }
    }

    // Check LIBRARY_PATH environment variable
    if let Ok(library_path) = std::env::var("LIBRARY_PATH") {
        for path in library_path.split(':') {
            if !path.is_empty() {
                println!("cargo:rustc-link-search=native={}", path);
                if !found {
                    let cudart_path = format!("{}/libcudart.so", path);
                    if std::path::Path::new(&cudart_path).exists() {
                        println!("cargo:warning=Found CUDA runtime via LIBRARY_PATH at: {}", path);
                        found = true;
                    }
                }
            }
        }
    }

    if !found {
        println!("cargo:warning=Could not find CUDA runtime library (libcudart.so)");
        println!("cargo:warning=Please set CUDA_PATH, CUDA_HOME, or LIBRARY_PATH environment variable");
        println!("cargo:warning=Or ensure CUDA is installed in a standard location");
        println!("cargo:warning=Searched paths include /usr/local/cuda and its subdirectories");
    }

    // Always emit the link directive for cudart
    println!("cargo:rustc-link-lib=cudart");
}
