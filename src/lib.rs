//! # Microlog
//!
//! A minimal Datalog engine implementation in Rust.
//!
//! ## Features
//!
//! - Rule evaluation with transitive closure
//!
//! ## Example
//!
//! ```rust
//! use microlog::{DatalogEngine, Atom, Rule, Term};
//!
//! let mut engine = DatalogEngine::new();
//! // Add facts and rules to the engine
//! ```

/// Datalog engine.
pub mod engine;
pub use engine::{Atom, DatalogEngine, Rule, Term};
