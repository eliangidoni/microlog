#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use microlog::{Atom, DatalogEngine, Query, Rule, Term};

fn setup_large_graph() -> DatalogEngine {
    let mut engine = DatalogEngine::new();

    // Create a large graph with 1000 nodes
    for i in 0..1000 {
        for j in 0..5 {
            let next = (i + j + 1) % 1000;
            engine.add_fact(Atom {
                predicate: "edge".to_string(),
                terms: vec![
                    Term::Symbol(format!("node_{i}")),
                    Term::Symbol(format!("node_{next}")),
                ],
            });
        }
    }

    // Add transitive closure rules
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
    engine
}

fn query_specific_paths(c: &mut Criterion) {
    let engine = setup_large_graph();

    c.bench_function("query_specific_paths", |b| {
        b.iter(|| {
            let query = Query {
                atom: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Symbol("node_0".to_string()),
                        Term::Variable("x".to_string()),
                    ],
                },
            };
            black_box(engine.query(&query))
        });
    });
}

fn query_existence_check(c: &mut Criterion) {
    let engine = setup_large_graph();

    c.bench_function("query_existence_check", |b| {
        b.iter(|| {
            let query = Query {
                atom: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Symbol("node_0".to_string()),
                        Term::Symbol("node_100".to_string()),
                    ],
                },
            };
            black_box(engine.ask(&query))
        });
    });
}

fn query_all_paths(c: &mut Criterion) {
    let engine = setup_large_graph();

    c.bench_function("query_all_paths", |b| {
        b.iter(|| {
            let query = Query {
                atom: Atom {
                    predicate: "path".to_string(),
                    terms: vec![
                        Term::Variable("x".to_string()),
                        Term::Variable("y".to_string()),
                    ],
                },
            };
            black_box(engine.query(&query))
        });
    });
}

criterion_group!(
    benches,
    query_specific_paths,
    query_existence_check,
    query_all_paths
);
criterion_main!(benches);
