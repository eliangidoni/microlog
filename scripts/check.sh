#!/bin/bash

# Quality checks script for the microlog project
# This script runs the same checks as the pre-push hook
# Usage: ./scripts/check.sh

set -e  # Exit on any error

echo "🔧 Running checks..."

# 1. Format code
echo "📝 Running cargo fmt..."
if ! cargo fmt --check; then
    echo "❌ Code formatting issues found. Please run 'cargo fmt' and commit the changes."
    exit 1
fi

# 2. Run clippy (lint)
echo "🔍 Running cargo clippy..."
if ! cargo clippy --all-targets --all-features -- -D warnings; then
    echo "❌ Clippy found issues."
    exit 1
fi

# 3. Run tests
echo "🧪 Running cargo test..."
if ! cargo test; then
    echo "❌ Tests failed."
    exit 1
fi

echo "✅ All checks passed!"
