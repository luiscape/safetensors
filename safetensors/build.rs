//! Build script to auto-detect CUDA library paths for linking.

fn main() {
    // Only run CUDA detection when the cuda feature is enabled
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=LIBRARY_PATH");

    // Common CUDA installation paths
    let search_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-11/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ];

    for path in &search_paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    // Add paths from environment variables
    for env_var in &["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(cuda_path) = std::env::var(env_var) {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/targets/x86_64-linux/lib", cuda_path);
        }
    }

    // Add paths from LIBRARY_PATH
    if let Ok(library_path) = std::env::var("LIBRARY_PATH") {
        for path in library_path.split(':') {
            if !path.is_empty() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    println!("cargo:rustc-link-lib=cudart");
}
