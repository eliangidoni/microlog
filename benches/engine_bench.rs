#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use microlog::{Atom, DatalogEngine, Rule, Term};

/// Benchmark for adding facts to the engine
fn bench_add_facts(c: &mut Criterion) {
    c.bench_function("add_facts", |b| {
        b.iter(|| {
            let mut engine = DatalogEngine::new();

            // Add multiple facts to measure insertion performance
            for i in 0..1000 {
                engine.add_fact(black_box(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Symbol(format!("node_{i}")),
                        Term::Symbol(format!("node_{}", i + 1)),
                    ],
                }));
            }

            black_box(engine)
        });
    });
}

/// Benchmark for simple rule evaluation (single rule)
fn bench_simple_rule_evaluation(c: &mut Criterion) {
    c.bench_function("simple_rule_evaluation", |b| {
        b.iter(|| {
            let mut engine = DatalogEngine::new();

            // Add base facts
            for i in 0..100 {
                engine.add_fact(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Symbol(format!("n{i}")),
                        Term::Symbol(format!("n{}", i + 1)),
                    ],
                });
            }

            // Add simple rule: path(x, y) <- edge(x, y)
            engine.add_rule(Rule {
                head: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                },
                body: vec![Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                }],
            });

            engine.run();
            black_box(engine.get_facts("path"))
        });
    });
}

/// Benchmark for transitive closure computation
fn bench_transitive_closure(c: &mut Criterion) {
    c.bench_function("transitive_closure", |b| {
        b.iter(|| {
            let mut engine = DatalogEngine::new();

            // Create a linear chain of edges for transitive closure
            // This will create a worst-case scenario for transitive closure
            for i in 0..50 {
                engine.add_fact(Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Symbol(format!("node_{i}")),
                        Term::Symbol(format!("node_{}", i + 1)),
                    ],
                });
            }

            // Add base case rule: path(x, y) <- edge(x, y)
            engine.add_rule(Rule {
                head: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                },
                body: vec![Atom {
                    predicate: "edge".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                }],
            });

            // Add recursive rule: path(x, z) <- path(x, y), edge(y, z)
            engine.add_rule(Rule {
                head: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("z".to_string()),
                    ],
                },
                body: vec![
                    Atom {
                        predicate: "path".to_string(),
                        terms: vec![
                            Term::Variable("x".to_string()),
                            Term::Variable("y".to_string()),
                        ],
                    },
                    Atom {
                        predicate: "edge".to_string(),
                        terms: vec![
                            Term::Variable("y".to_string()),
                            Term::Variable("z".to_string()),
                        ],
                    },
                ],
            });

            engine.run();
            black_box(engine.get_facts("path"))
        });
    });
}

/// Benchmark for complex graph with multiple rules
/// Setup complex graph with edges and node types
fn setup_complex_graph() -> DatalogEngine {
    let mut engine = DatalogEngine::new();

    // Create a more complex graph structure
    // Add nodes with multiple connections
    for i in 0..30 {
        for j in 0..3 {
            engine.add_fact(Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Symbol(format!("n{i}")),
                    Term::Symbol(format!("n{}", (i + j + 1) % 30)),
                ],
            });
        }
    }

    // Add some node properties
    for i in 0..30 {
        engine.add_fact(Atom {
            predicate: "node_type".to_string(),
            terms: vec![
                Term::Symbol(format!("n{i}")),
                Term::Symbol(if i % 2 == 0 {
                    "even".to_string()
                } else {
                    "odd".to_string()
                }),
            ],
        });
    }

    engine
}

/// Add complex graph analysis rules
fn add_complex_analysis_rules(engine: &mut DatalogEngine) {
    // Rule 1: path(x, y) <- edge(x, y)
    engine.add_rule(Rule {
        head: Atom {
            predicate: "path".to_string(),
            terms: vec![
                Term::Variable("x".to_string()),
                Term::Variable("y".to_string()),
            ],
        },
        body: vec![Atom {
            predicate: "edge".to_string(),
            terms: vec![
                Term::Variable("x".to_string()),
                Term::Variable("y".to_string()),
            ],
        }],
    });

    // Rule 2: path(x, z) <- path(x, y), edge(y, z)
    engine.add_rule(Rule {
        head: Atom {
            predicate: "path".to_string(),
            terms: vec![
                Term::Variable("x".to_string()),
                Term::Variable("z".to_string()),
            ],
        },
        body: vec![
            Atom {
                predicate: "path".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ],
            },
            Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Variable("y".to_string()),
                    Term::Variable("z".to_string()),
                ],
            },
        ],
    });

    // Rule 3: reachable_same_type(x, y) <- path(x, y), node_type(x, t), node_type(y, t)
    engine.add_rule(Rule {
        head: Atom {
            predicate: "reachable_same_type".to_string(),
            terms: vec![
                Term::Variable("x".to_string()),
                Term::Variable("y".to_string()),
            ],
        },
        body: vec![
            Atom {
                predicate: "path".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ],
            },
            Atom {
                predicate: "node_type".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("t".to_string()),
                ],
            },
            Atom {
                predicate: "node_type".to_string(),
                terms: vec![
                    Term::Variable("y".to_string()),
                    Term::Variable("t".to_string()),
                ],
            },
        ],
    });
}

fn bench_complex_graph_analysis(c: &mut Criterion) {
    c.bench_function("complex_graph_analysis", |b| {
        b.iter(|| {
            let mut engine = setup_complex_graph();
            add_complex_analysis_rules(&mut engine);

            engine.run();
            let path_facts = engine.get_facts("path");
            let same_type_facts = engine.get_facts("reachable_same_type");
            black_box((path_facts, same_type_facts))
        });
    });
}

/// Benchmark for fact retrieval performance
fn bench_fact_retrieval(c: &mut Criterion) {
    // Setup: Create an engine with many facts
    let mut engine = DatalogEngine::new();

    for i in 0..10000 {
        engine.add_fact(Atom {
            predicate: "large_relation".to_string(),
            terms: vec![
                Term::Symbol(format!("item_{i}")),
                Term::Symbol(format!("value_{}", i % 100)),
            ],
        });
    }

    c.bench_function("fact_retrieval", |b| {
        b.iter(|| black_box(engine.get_facts("large_relation")));
    });
}

criterion_group!(
    benches,
    bench_add_facts,
    bench_simple_rule_evaluation,
    bench_transitive_closure,
    bench_complex_graph_analysis,
    bench_fact_retrieval
);
criterion_main!(benches);
