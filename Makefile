# Makefile for microlog Rust project

.PHONY: setup check test clean format lint bench help

# Default target
all: check

# Install git hooks and setup development environment
setup:
	@echo "🔧 Setting up development environment..."
	@chmod +x scripts/check.sh
	@cp scripts/check.sh .git/hooks/pre-push
	@echo "✅ Pre-push hook installed and configured"
	@echo "✅ Development environment setup complete"

# Run quality checks (format, lint, test)
check:
	@./scripts/check.sh

# Run tests only
test:
	@echo "🧪 Running tests..."
	@cargo test

# Format code
format:
	@echo "📝 Formatting code..."
	@cargo fmt

# Run linter
lint:
	@echo "🔍 Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

# Run benchmarks
bench:
	@echo "⚡ Running benchmarks..."
	@cargo bench

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean

# Show help
help:
	@echo "Available targets:"
	@echo "  setup   - Install git hooks and setup development environment"
	@echo "  check   - Run all quality checks (format, lint, test)"
	@echo "  test    - Run tests only"
	@echo "  format  - Format code with cargo fmt"
	@echo "  lint    - Run clippy linter"
	@echo "  bench   - Run benchmarks"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help message"
