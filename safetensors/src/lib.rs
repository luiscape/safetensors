#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
pub mod slice;
pub mod tensor;
/// Bulk I/O module for optimized parallel file reading (fastsafetensors integration).
/// Only available when the `fast_io` feature is enabled.
#[cfg(feature = "fast_io")]
pub mod bulk_io;
/// serialize_to_file only valid in std
#[cfg(feature = "std")]
pub use tensor::serialize_to_file;
pub use tensor::{serialize, Dtype, SafeTensorError, SafeTensors, View};

#[cfg(feature = "fast_io")]
pub use bulk_io::{AlignmentFixup, BulkReadPlan, PreadBulkReader};

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;

/// A facade around all the types we need from the `std`, `core`, and `alloc`
/// crates. This avoids elaborate import wrangling having to happen in every
/// module.
mod lib {
    #[cfg(not(feature = "std"))]
    mod no_stds {
        pub use alloc::borrow::Cow;
        pub use alloc::string::{String, ToString};
        pub use alloc::vec::Vec;
        pub use hashbrown::HashMap;
    }
    #[cfg(feature = "std")]
    mod stds {
        pub use std::borrow::Cow;
        pub use std::collections::HashMap;
        pub use std::string::{String, ToString};
        pub use std::vec::Vec;
    }
    /// choose std or no_std to export by feature flag
    #[cfg(not(feature = "std"))]
    pub use no_stds::*;
    #[cfg(feature = "std")]
    pub use stds::*;
}
