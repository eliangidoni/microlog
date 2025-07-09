//! # Microlog
//!
//! A minimal Datalog engine implementation.
//!
//! ## Example
//!
//! ```rust
//! use microlog::{DatalogEngine, Atom, Rule, Term, Query};
//!
//! let mut engine = DatalogEngine::new();
//!
//! // Add a fact: edge("a", "b")
//! engine.add_fact(Atom {
//!     predicate: "edge".to_string(),
//!     terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
//! });
//!
//! // Add a rule: path(x, y) <- edge(x, y)
//! engine.add_rule(Rule {
//!     head: Atom {
//!         predicate: "path".to_string(),
//!         terms: vec![Term::Variable("x".to_string()), Term::Variable("y".to_string())],
//!     },
//!     body: vec![Atom {
//!         predicate: "edge".to_string(),
//!         terms: vec![Term::Variable("x".to_string()), Term::Variable("y".to_string())],
//!     }],
//! });
//!
//! // Run evaluation
//! engine.run();
//!
//! // Query: What can we reach from "a"?
//! let query = Query {
//!     atom: Atom {
//!         predicate: "path".to_string(),
//!         terms: vec![Term::Symbol("a".to_string()), Term::Variable("y".to_string())],
//!     },
//! };
//! let results = engine.query(&query);
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].bindings.get("y"), Some(&"b".to_string()));
//! ```

/// Datalog engine.
pub mod engine;
pub use engine::{Atom, DatalogEngine, Query, QueryResult, Rule, Term};
