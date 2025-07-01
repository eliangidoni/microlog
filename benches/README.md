# Microlog Benchmarks

This directory contains performance benchmarks for the Microlog Datalog engine using [Criterion.rs](https://bheisler.github.io/criterion.rs/).

## Running Benchmarks

To run all benchmarks:
```bash
cargo bench
```

To run a specific benchmark:
```bash
cargo bench --bench engine_bench -- bench_transitive_closure
```

To run benchmarks with quick profiling (faster but less accurate):
```bash
cargo bench --bench engine_bench -- --quick
```

## Benchmark Descriptions

### `bench_add_facts`
Measures the performance of adding 1,000 facts to the Datalog engine. This tests the basic fact insertion performance.

### `bench_simple_rule_evaluation`
Tests the performance of evaluating a simple rule (`path(x, y) <- edge(x, y)`) over 100 edge facts. This measures basic rule evaluation without recursion.

### `bench_transitive_closure`
Benchmarks the computation of transitive closure over a linear chain of 50 edges. This creates a worst-case scenario for transitive closure computation where each iteration of the fixed-point algorithm adds new facts.

Expected facts generated: 1,275 path facts (50 + 49 + 48 + ... + 1)

### `bench_complex_graph_analysis`
Tests performance on a more complex scenario with:
- 30 nodes with multiple outgoing edges (90 total edges)
- Node type classification (even/odd)
- Three rules:
  1. Base case: `path(x, y) <- edge(x, y)`
  2. Transitive case: `path(x, z) <- path(x, y), edge(y, z)`
  3. Same-type reachability: `reachable_same_type(x, y) <- path(x, y), node_type(x, t), node_type(y, t)`

### `bench_fact_retrieval`
Measures the performance of retrieving all facts for a predicate from a large relation containing 10,000 facts.

## Performance Expectations

Typical performance on a modern machine:

- **add_facts**: ~500 µs for 1,000 facts
- **simple_rule_evaluation**: ~340 µs for 100 facts + 1 rule
- **transitive_closure**: ~900 ms for 50-node linear chain
- **complex_graph_analysis**: ~350 ms for 30-node complex graph
- **fact_retrieval**: ~2.4 ms for 10,000 facts

## Benchmark Reports

Criterion generates detailed HTML reports in `target/criterion/` with:
- Performance trends over time
- Statistical analysis
- Regression detection
- Detailed timing distributions

Open `target/criterion/report/index.html` in your browser to view the full reports.

## Optimization Notes

The benchmarks help identify performance bottlenecks:

1. **Transitive closure** is the most expensive operation due to the iterative fixed-point computation
2. **Join operations** in complex rules can be costly with large datasets
3. **Fact storage** using `HashSet<Vec<String>>` provides good performance for most use cases
4. **Memory allocation** during rule evaluation can be optimized further

Consider profiling with tools like `perf` or `flamegraph` for deeper performance analysis.
