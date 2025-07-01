# GitHub Copilot Instructions for Rust Project

## Code Style and Conventions

- Follow standard Rust naming conventions (snake_case for functions/variables, PascalCase for types)
- Use `rustfmt` formatting standards
- Prefer explicit types when they improve readability
- Use meaningful variable and function names that express intent
- Add comprehensive documentation comments using `///` for public APIs

## Error Handling

- Use `Result<T, E>` for recoverable errors instead of panicking
- Prefer `?` operator for error propagation
- Use `anyhow` or `thiserror` for error handling when appropriate
- Avoid `unwrap()` and `expect()` in production code - use proper error handling
- Use `Option<T>` for values that may be absent

## Memory Safety and Performance

- Leverage Rust's ownership system instead of fighting it
- Use references (`&`) and borrowing when possible to avoid unnecessary clones
- Prefer `Vec<T>` over arrays for dynamic collections
- Use `String` for owned strings and `&str` for string slices
- Consider using `Cow<str>` when you need flexibility between owned and borrowed strings
- Use `Arc<T>` and `Mutex<T>` for shared mutable state across threads

## Idiomatic Rust Patterns

- Use pattern matching with `match` expressions extensively
- Prefer iterator methods (`.map()`, `.filter()`, `.collect()`) over manual loops
- Use `if let` and `while let` for simple pattern matching
- Implement `From` and `Into` traits for type conversions
- Use `derive` macros for common traits (`Debug`, `Clone`, `PartialEq`, etc.)
- Prefer composition over inheritance

## Testing

- Write unit tests in the same file using `#[cfg(test)]` modules
- Use descriptive test function names that explain what is being tested
- Use `assert_eq!`, `assert_ne!`, and `assert!` macros appropriately
- Write integration tests in the `tests/` directory
- Use `#[should_panic]` for tests that should panic
- Consider using `proptest` or `quickcheck` for property-based testing

## Dependencies and Cargo

- Keep dependencies minimal and well-justified
- Use specific version constraints in `Cargo.toml`
- Prefer widely-used, well-maintained crates
- Group dependencies logically and add comments explaining their purpose
- Use feature flags to make optional dependencies truly optional

## Documentation

- Write module-level documentation using `//!`
- Document all public functions, structs, and enums with `///`
- Include examples in documentation comments using code blocks
- Use `cargo doc` to generate and verify documentation
- Keep README.md updated with current usage examples

## Safety and Security

- Avoid `unsafe` code unless absolutely necessary and thoroughly documented
- When using `unsafe`, clearly document why it's safe
- Be cautious with external input and validate/sanitize appropriately
- Use type system to enforce invariants rather than runtime checks when possible

## Async Programming (if applicable)

- Use `async`/`await` syntax for asynchronous operations
- Prefer `tokio` runtime for async applications
- Use `futures` crate utilities when needed
- Be mindful of blocking operations in async contexts
- Use `spawn` for concurrent tasks

## Logging and Debugging

- Use `log` crate with appropriate log levels
- Include contextual information in log messages
- Use `env_logger` or similar for development
- Consider structured logging with `serde_json` for production

## Code Organization

- Keep modules focused and cohesive
- Use `pub(crate)` for internal APIs
- Organize code into logical modules
- Keep functions small and focused on a single responsibility
- Use `mod.rs` files to organize module hierarchies

## General Best Practices

- Run `cargo clippy` regularly and address warnings
- Use `cargo fmt` to maintain consistent formatting
- Write self-documenting code with clear variable names
- Favor explicit over implicit when it improves clarity
- Consider backwards compatibility for public APIs
- Use semantic versioning for releases