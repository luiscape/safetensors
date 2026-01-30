//! Error types for GPU operations.

#![allow(missing_docs)]

use std::fmt;

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Errors that can occur during GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// CUDA driver or runtime error with error code.
    CudaError {
        /// The CUDA error code.
        code: i32,
        /// Human-readable error message.
        message: String,
    },

    /// Failed to allocate pinned (page-locked) memory.
    PinnedAllocationFailed {
        /// Requested size in bytes.
        requested_bytes: usize,
        /// Optional underlying error message.
        reason: Option<String>,
    },

    /// Pinned memory pool exhausted.
    OutOfPinnedMemory {
        /// Requested size in bytes.
        requested_bytes: usize,
        /// Currently allocated bytes.
        allocated_bytes: usize,
        /// Maximum allowed bytes.
        max_bytes: usize,
    },

    /// GPU memory allocation failed.
    GpuAllocationFailed {
        /// Requested size in bytes.
        requested_bytes: usize,
        /// Device ID where allocation failed.
        device_id: i32,
    },

    /// Invalid device ID.
    InvalidDevice {
        /// The invalid device ID.
        device_id: i32,
        /// Number of available devices.
        num_devices: i32,
    },

    /// Stream synchronization failed.
    SynchronizationFailed {
        /// Optional error message.
        message: String,
    },

    /// Memory transfer failed.
    TransferFailed {
        /// Source description (e.g., "host", "device:0").
        from: String,
        /// Destination description.
        to: String,
        /// Size in bytes.
        size: usize,
        /// Optional underlying error.
        reason: Option<String>,
    },

    /// CUDA not available or not initialized.
    CudaNotAvailable {
        /// Reason CUDA is not available.
        reason: String,
    },

    /// GPUDirect Storage error.
    #[cfg(feature = "gds")]
    GdsError {
        /// Error code from cuFile.
        code: i32,
        /// Error message.
        message: String,
    },

    /// I/O error during file operations.
    IoError(std::io::Error),

    /// Safetensor parsing error.
    SafeTensorError(crate::SafeTensorError),

    /// Invalid configuration.
    InvalidConfig {
        /// Description of the configuration error.
        message: String,
    },

    /// Operation was cancelled.
    Cancelled,

    /// Buffer size mismatch.
    BufferSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::CudaError { code, message } => {
                write!(f, "CUDA error (code {}): {}", code, message)
            }
            GpuError::PinnedAllocationFailed {
                requested_bytes,
                reason,
            } => {
                write!(
                    f,
                    "failed to allocate {} bytes of pinned memory",
                    requested_bytes
                )?;
                if let Some(reason) = reason {
                    write!(f, ": {}", reason)?;
                }
                Ok(())
            }
            GpuError::OutOfPinnedMemory {
                requested_bytes,
                allocated_bytes,
                max_bytes,
            } => {
                write!(
                    f,
                    "out of pinned memory: requested {} bytes, {} of {} bytes already allocated",
                    requested_bytes, allocated_bytes, max_bytes
                )
            }
            GpuError::GpuAllocationFailed {
                requested_bytes,
                device_id,
            } => {
                write!(
                    f,
                    "failed to allocate {} bytes on GPU device {}",
                    requested_bytes, device_id
                )
            }
            GpuError::InvalidDevice {
                device_id,
                num_devices,
            } => {
                write!(
                    f,
                    "invalid device ID {}: only {} devices available",
                    device_id, num_devices
                )
            }
            GpuError::SynchronizationFailed { message } => {
                write!(f, "stream synchronization failed: {}", message)
            }
            GpuError::TransferFailed {
                from,
                to,
                size,
                reason,
            } => {
                write!(f, "transfer of {} bytes from {} to {} failed", size, from, to)?;
                if let Some(reason) = reason {
                    write!(f, ": {}", reason)?;
                }
                Ok(())
            }
            GpuError::CudaNotAvailable { reason } => {
                write!(f, "CUDA not available: {}", reason)
            }
            #[cfg(feature = "gds")]
            GpuError::GdsError { code, message } => {
                write!(f, "GPUDirect Storage error (code {}): {}", code, message)
            }
            GpuError::IoError(err) => {
                write!(f, "I/O error: {}", err)
            }
            GpuError::SafeTensorError(err) => {
                write!(f, "safetensor error: {}", err)
            }
            GpuError::InvalidConfig { message } => {
                write!(f, "invalid configuration: {}", message)
            }
            GpuError::Cancelled => {
                write!(f, "operation was cancelled")
            }
            GpuError::BufferSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "buffer size mismatch: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::IoError(err) => Some(err),
            GpuError::SafeTensorError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GpuError {
    fn from(err: std::io::Error) -> Self {
        GpuError::IoError(err)
    }
}

impl From<crate::SafeTensorError> for GpuError {
    fn from(err: crate::SafeTensorError) -> Self {
        GpuError::SafeTensorError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GpuError::CudaError {
            code: 2,
            message: "out of memory".to_string(),
        };
        assert!(err.to_string().contains("CUDA error"));
        assert!(err.to_string().contains("out of memory"));

        let err = GpuError::OutOfPinnedMemory {
            requested_bytes: 1024,
            allocated_bytes: 900,
            max_bytes: 1000,
        };
        assert!(err.to_string().contains("1024"));
        assert!(err.to_string().contains("pinned memory"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let gpu_err: GpuError = io_err.into();
        assert!(matches!(gpu_err, GpuError::IoError(_)));
    }
}
