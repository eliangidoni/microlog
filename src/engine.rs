use indexmap::{IndexMap, IndexSet};
use std::hash::Hash;

/// Represents a Datalog variable
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Term {
    /// A variable that can be unified with symbols (e.g., `x`, `y`)
    Variable(String),
    /// A concrete symbol/constant (e.g., `"alice"`, `"bob"`)
    Symbol(String),
}

/// A Datalog predicate (e.g., `edge(x, y)`)
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Atom {
    /// The name of the predicate (e.g., `"edge"`, `"path"`)
    pub predicate: String,
    /// The arguments/terms of the predicate
    pub terms: Vec<Term>,
}

/// A Datalog rule (e.g., `path(x, y) <- edge(x, y)`)
#[derive(Debug, Clone)]
pub struct Rule {
    /// The conclusion/consequent of the rule
    pub head: Atom,
    /// The conditions/antecedents that must be satisfied
    pub body: Vec<Atom>,
}

/// A Datalog query (e.g., `path(x, "alice")` or `edge("bob", y)`)
#[derive(Debug, Clone)]
pub struct Query {
    /// The atom pattern to match against facts
    pub atom: Atom,
}

/// Result of a query - a set of variable bindings
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult {
    /// Variable bindings that satisfy the query
    pub bindings: IndexMap<String, String>,
}

/// Index for fast fact lookup by argument position and value
#[derive(Debug, Clone)]
struct PredicateIndex {
    /// For each argument position, maps values to sets of complete facts
    /// For predicate `edge(a,b), edge(a,c), edge(b,c)`:
    /// - Position 0 index: `{"a" -> {["a","b"], ["a","c"]}, "b" -> {["b","c"]}}`
    /// - Position 1 index: `{"b" -> {["a","b"]}, "c" -> {["a","c"], ["b","c"]}}`
    by_position: Vec<IndexMap<String, IndexSet<Vec<String>>>>,
}

impl PredicateIndex {
    fn new(arity: usize) -> Self {
        Self {
            by_position: (0..arity).map(|_| IndexMap::new()).collect(),
        }
    }

    fn add_fact(&mut self, fact: &[String]) {
        for (pos, value) in fact.iter().enumerate() {
            if pos < self.by_position.len() {
                self.by_position[pos]
                    .entry(value.clone())
                    .or_default()
                    .insert(fact.to_vec());
            }
        }
    }

    /// Get facts that match a specific value at a given position
    fn get_facts_by_position(
        &self,
        position: usize,
        value: &str,
    ) -> Option<&IndexSet<Vec<String>>> {
        self.by_position
            .get(position)
            .and_then(|pos_map| pos_map.get(value))
    }
}

/// The Datalog evaluation engine
#[derive(Debug)]
pub struct DatalogEngine {
    facts_by_pred: IndexMap<String, IndexSet<Vec<String>>>,
    rules: Vec<Rule>,
    /// Track new facts added in the current iteration for semi-naive evaluation
    new_facts_by_pred: IndexMap<String, IndexSet<Vec<String>>>,
    /// Indexes for fast fact lookup by argument position and value
    indexes_by_pred: IndexMap<String, PredicateIndex>,
}

impl Default for DatalogEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DatalogEngine {
    /// Create a new Datalog engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            facts_by_pred: IndexMap::new(),
            rules: Vec::new(),
            new_facts_by_pred: IndexMap::new(),
            indexes_by_pred: IndexMap::new(),
        }
    }

    /// Adds a single fact by predicate name and values
    /// Returns true if any new facts were added, false otherwise
    fn add_fact_vec(&mut self, predicate: String, fact_values: Vec<String>) -> bool {
        let facts = self.facts_by_pred.entry(predicate.clone()).or_default();

        // Track if this is a new fact
        if facts.insert(fact_values.clone()) {
            // Update the index for this predicate
            let index = self
                .indexes_by_pred
                .entry(predicate.clone())
                .or_insert_with(|| PredicateIndex::new(fact_values.len()));

            // Check for arity consistency
            if index.by_position.len() == fact_values.len() {
                index.add_fact(&fact_values);
            } else {
                // Recreate index with new arity if needed
                let mut new_index = PredicateIndex::new(fact_values.len());
                // Re-index all existing facts for this predicate
                for existing_fact in facts.iter() {
                    new_index.add_fact(existing_fact);
                }
                *index = new_index;
            }

            self.new_facts_by_pred
                .entry(predicate)
                .or_default()
                .insert(fact_values);

            true
        } else {
            false
        }
    }

    /// Add a fact to the knowledge base
    ///
    /// # Panics
    ///
    /// Panics if the atom contains any variables (only symbols are allowed in facts)
    pub fn add_fact(&mut self, atom: Atom) {
        let values: Vec<String> = atom
            .terms
            .into_iter()
            .map(|term| match term {
                Term::Symbol(s) => s,
                Term::Variable(_) => panic!("Facts must contain only symbols"),
            })
            .collect();

        self.add_fact_vec(atom.predicate, values);
    }

    /// Add a rule to the knowledge base
    ///
    /// # Panics
    ///
    /// Panics if the rule contains variables in the head that don't appear in the body
    pub fn add_rule(&mut self, rule: Rule) {
        // Validate that all variables in the head appear in the body
        let head_vars: IndexSet<&String> = rule
            .head
            .terms
            .iter()
            .filter_map(|term| match term {
                Term::Variable(var) => Some(var),
                Term::Symbol(_) => None,
            })
            .collect();

        let body_vars: IndexSet<&String> = rule
            .body
            .iter()
            .flat_map(|atom| {
                atom.terms.iter().filter_map(|term| match term {
                    Term::Variable(var) => Some(var),
                    Term::Symbol(_) => None,
                })
            })
            .collect();

        for head_var in &head_vars {
            assert!(
                body_vars.contains(head_var),
                "Rule validation failed: Variable '{head_var}' appears in head but not in body"
            );
        }

        self.rules.push(rule);
    }

    /// Run the Datalog evaluation to fixed point using semi-naive evaluation
    ///
    /// - First iteration: Process all existing facts (naive evaluation)
    /// - Subsequent iterations: For each rule, only generate new facts by using
    ///   at least one "new" fact from the previous iteration in the rule body
    /// - Termination: Stop when no new facts are derived in an iteration
    ///
    /// ## Example
    ///
    /// For a transitive closure rule `path(x,z) :- path(x,y), edge(y,z)`:
    /// - Iteration 1: Derive `path(a,b)` from `edge(a,b)`
    /// - Iteration 2: Only consider joins where either the `path` or `edge` atom
    ///   uses the newly derived `path(a,b)` fact, avoiding redundant work
    pub fn run(&mut self) {
        // Before starting a fresh evaluation clear existing new facts
        self.new_facts_by_pred.clear();

        let mut changed = false;
        for rule_idx in 0..self.rules.len() {
            let new_facts = self.evaluate_rule_naive(&self.rules[rule_idx]);
            let predicate = self.rules[rule_idx].head.predicate.clone();
            for fact in new_facts {
                changed |= self.add_fact_vec(predicate.clone(), fact);
            }
        }
        while changed {
            changed = false;
            // Clear new facts from previous iteration
            let mut current_new_facts = IndexMap::new();
            std::mem::swap(&mut current_new_facts, &mut self.new_facts_by_pred);

            for rule_idx in 0..self.rules.len() {
                let new_facts =
                    self.evaluate_rule_semi_naive(&self.rules[rule_idx], &current_new_facts);
                let predicate = self.rules[rule_idx].head.predicate.clone();
                for fact in new_facts {
                    changed |= self.add_fact_vec(predicate.clone(), fact);
                }
            }
        }
    }

    /// Evaluate a single rule using all facts
    fn evaluate_rule_naive(&self, rule: &Rule) -> IndexSet<Vec<String>> {
        self.evaluate_rule_with_fact_provider(rule, |predicate, _atom_idx| {
            self.facts_by_pred
                .get(predicate)
                .cloned()
                .unwrap_or_default()
        })
    }

    /// Evaluate a single rule using semi-naive evaluation
    /// Only generates facts that involve at least one new fact from the previous iteration
    fn evaluate_rule_semi_naive(
        &self,
        rule: &Rule,
        new_facts_by_pred: &IndexMap<String, IndexSet<Vec<String>>>,
    ) -> IndexSet<Vec<String>> {
        // For each atom in the rule body, evaluate using new facts for that atom
        (0..rule.body.len())
            .flat_map(|atom_idx| {
                self.evaluate_rule_with_fact_provider(rule, |predicate, current_atom_idx| {
                    if current_atom_idx == atom_idx {
                        // Use only new facts for the designated atom
                        new_facts_by_pred
                            .get(predicate)
                            .cloned()
                            .unwrap_or_default()
                    } else {
                        // Use all facts for other atoms
                        self.facts_by_pred
                            .get(predicate)
                            .cloned()
                            .unwrap_or_default()
                    }
                })
            })
            .collect()
    }

    /// Common rule evaluation logic that accepts a fact provider function
    fn evaluate_rule_with_fact_provider<F>(
        &self,
        rule: &Rule,
        fact_provider: F,
    ) -> IndexSet<Vec<String>>
    where
        F: Fn(&str, usize) -> IndexSet<Vec<String>>,
    {
        // Start with a single empty binding and fold through each atom in the rule body
        let final_bindings = rule
            .body
            .iter()
            .enumerate()
            .try_fold(vec![IndexMap::new()], |bindings, (atom_idx, atom)| {
                let facts = fact_provider(&atom.predicate, atom_idx);

                // Early return if no facts available
                if facts.is_empty() {
                    return None;
                }

                // For each existing binding, try to match against all candidate facts
                let new_bindings = bindings
                    .into_iter()
                    .flat_map(|binding| {
                        let candidate_facts = self.filter_facts_by_bindings(atom, &facts, &binding);
                        candidate_facts
                            .into_iter()
                            .filter_map(move |tuple| Self::match_atom(atom, &tuple, &binding))
                    })
                    .collect();

                Some(new_bindings)
            })
            .unwrap_or_default();

        // Generate facts from successful bindings
        final_bindings
            .into_iter()
            .filter_map(|binding| {
                let fact_terms: Option<Vec<String>> = rule
                    .head
                    .terms
                    .iter()
                    .map(|term| match term {
                        Term::Symbol(sym) => Some(sym.clone()),
                        Term::Variable(var) => binding.get(var).cloned(),
                    })
                    .collect();
                fact_terms
            })
            .collect()
    }

    /// Filters facts using indexes:
    /// - Finds all possible indexes for bound variables and constants
    /// - Chooses the most selective index (smallest result set)
    /// - Falls back to returning all facts if no good index is found
    fn filter_facts_by_bindings(
        &self,
        atom: &Atom,
        facts: &IndexSet<Vec<String>>,
        binding: &IndexMap<String, String>,
    ) -> IndexSet<Vec<String>> {
        let Some(index) = self.indexes_by_pred.get(&atom.predicate) else {
            return facts.clone();
        };

        // Collect all possible index lookups with their selectivity
        let mut index_options: Vec<(usize, &IndexSet<Vec<String>>)> = Vec::new();

        for (pos, term) in atom.terms.iter().enumerate() {
            let value = match term {
                Term::Symbol(sym) => Some(sym.as_str()),
                Term::Variable(var) => binding.get(var).map(String::as_str),
            };

            // Only consider indexes where we have an actual value (not unbound variable)
            if let Some(val) = value {
                if let Some(indexed_facts) = index.get_facts_by_position(pos, val) {
                    index_options.push((indexed_facts.len(), indexed_facts));
                }
            }
        }

        match index_options.into_iter().min_by_key(|(size, _)| *size) {
            Some((_size, most_selective_facts)) => {
                most_selective_facts.intersection(facts).cloned().collect()
            }
            None => {
                // fallback to all facts
                facts.clone()
            }
        }
    }

    /// Match an atom against a tuple with current bindings
    fn match_atom(
        atom: &Atom,
        tuple: &[String],
        binding: &IndexMap<String, String>,
    ) -> Option<IndexMap<String, String>> {
        if atom.terms.len() != tuple.len() {
            return None;
        }

        let mut new_binding = binding.clone();

        atom.terms
            .iter()
            .zip(tuple)
            .try_for_each(|(term, value)| match term {
                Term::Symbol(sym) => (sym == value).then_some(()).ok_or(()),
                Term::Variable(var) => {
                    if let Some(bound) = new_binding.get(var) {
                        (bound == value).then_some(()).ok_or(())
                    } else {
                        new_binding.insert(var.clone(), value.clone());
                        Ok(())
                    }
                }
            })
            .ok()?;

        Some(new_binding)
    }

    /// Get all facts for a predicate
    #[must_use]
    pub fn get_facts(&self, predicate: &str) -> Vec<Vec<String>> {
        self.facts_by_pred
            .get(predicate)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Returns all variable bindings that make the query pattern true.
    #[must_use]
    pub fn query(&self, query: &Query) -> Vec<QueryResult> {
        let facts = self
            .facts_by_pred
            .get(&query.atom.predicate)
            .cloned()
            .unwrap_or_default();

        let empty_binding = IndexMap::new();

        // Filter facts and match patterns
        self.filter_facts_by_bindings(&query.atom, &facts, &empty_binding)
            .iter()
            .filter_map(|fact| {
                Self::match_atom(&query.atom, fact, &empty_binding)
                    .map(|bindings| QueryResult { bindings })
            })
            .collect()
    }

    /// Returns whether a binding exists
    #[must_use]
    pub fn ask(&self, query: &Query) -> bool {
        !self.query(query).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transitive_closure_with_path_rules() {
        // Create a new engine instance
        let mut engine = DatalogEngine::new();

        // Add base facts: edge(a, b) and edge(b, c)
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });

        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("b".to_string()), Term::Symbol("c".to_string())],
        });

        // Add rule: path(x, y) <- edge(x, y)
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

        // Add rule: path(x, z) <- path(x, y), edge(y, z)
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

        // Run the engine to compute transitive closure
        engine.run();

        // Get and sort results for consistent testing
        let mut path_facts = engine.get_facts("path");
        path_facts.sort();

        // Verify expected path facts are present
        // Should include: path(a, b), path(b, c), path(a, c)
        let expected_facts = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "c".to_string()],
            vec!["b".to_string(), "c".to_string()],
        ];

        assert_eq!(path_facts.len(), 3, "Should have exactly 3 path facts");

        for expected_fact in expected_facts {
            assert!(
                path_facts.contains(&expected_fact),
                "Missing expected fact: path({}, {})",
                expected_fact[0],
                expected_fact[1]
            );
        }
    }

    #[test]
    fn test_empty_engine_returns_no_facts() {
        let engine = DatalogEngine::new();
        let facts = engine.get_facts("nonexistent");
        assert!(facts.is_empty(), "Empty engine should return no facts");
    }

    #[test]
    fn test_single_fact_without_rules() {
        let mut engine = DatalogEngine::new();

        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![Term::Symbol("value".to_string())],
        });

        let facts = engine.get_facts("test");
        assert_eq!(facts.len(), 1, "Should have exactly one fact");
        assert_eq!(facts[0], vec!["value".to_string()]);
    }

    #[test]
    fn test_engine_creation() {
        let engine = DatalogEngine::new();
        assert!(engine.get_facts("any_predicate").is_empty());
    }

    #[test]
    fn test_longer_chain() {
        let mut engine = DatalogEngine::new();

        // Create a longer chain: a -> b -> c -> d -> e
        let edges = vec![("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")];

        for (from, to) in edges {
            engine.add_fact(Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Symbol(from.to_string()), Term::Symbol(to.to_string())],
            });
        }

        // Add the same transitive closure rules
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

        // Verify we get all expected paths
        let mut path_facts = engine.get_facts("path");
        path_facts.sort();

        // Expected paths: all pairs (x,y) where there's a path from x to y
        let expected_paths = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["a".to_string(), "c".to_string()],
            vec!["a".to_string(), "d".to_string()],
            vec!["a".to_string(), "e".to_string()],
            vec!["b".to_string(), "c".to_string()],
            vec!["b".to_string(), "d".to_string()],
            vec!["b".to_string(), "e".to_string()],
            vec!["c".to_string(), "d".to_string()],
            vec!["c".to_string(), "e".to_string()],
            vec!["d".to_string(), "e".to_string()],
        ];

        assert_eq!(
            path_facts.len(),
            expected_paths.len(),
            "Should have {} path facts, but got {}",
            expected_paths.len(),
            path_facts.len()
        );

        for expected_path in expected_paths {
            assert!(
                path_facts.contains(&expected_path),
                "Missing expected path: path({}, {})",
                expected_path[0],
                expected_path[1]
            );
        }
    }

    #[test]
    fn test_with_complex_rules() {
        // Test evaluation with a rule that has multiple body atoms
        let mut engine = DatalogEngine::new();

        // Add facts for a small graph
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("1".to_string()), Term::Symbol("2".to_string())],
        });
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("2".to_string()), Term::Symbol("3".to_string())],
        });
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("3".to_string()), Term::Symbol("4".to_string())],
        });

        // Add a rule that creates a triangle relationship
        engine.add_rule(Rule {
            head: Atom {
                predicate: "triangle".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                    Term::Variable("z".to_string()),
                ],
            },
            body: vec![
                Atom {
                    predicate: "edge".to_string(),
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

        let triangle_facts = engine.get_facts("triangle");

        // Should find triangles: (1,2,3) and (2,3,4)
        assert_eq!(triangle_facts.len(), 2);

        let expected_triangles = vec![
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
            vec!["2".to_string(), "3".to_string(), "4".to_string()],
        ];

        for expected in expected_triangles {
            assert!(
                triangle_facts.contains(&expected),
                "Missing expected triangle: {expected:?}"
            );
        }
    }

    #[test]
    fn test_correctness_large_graph() {
        // This tests evaluation on a larger, more complex graph structure
        let mut engine = DatalogEngine::new();

        // Create a more complex graph with cycles and multiple paths
        let edges = vec![
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("a", "f"),
            ("f", "g"),
            ("g", "d"),
            ("e", "h"),
            ("h", "i"),
            ("i", "j"),
            ("b", "k"),
            ("k", "l"),
            ("l", "m"),
        ];

        for (from, to) in edges {
            engine.add_fact(Atom {
                predicate: "edge".to_string(),
                terms: vec![Term::Symbol(from.to_string()), Term::Symbol(to.to_string())],
            });
        }

        // Add transitive closure rules
        engine.add_rule(Rule {
            head: Atom {
                predicate: "reachable".to_string(),
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
                predicate: "reachable".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("z".to_string()),
                ],
            },
            body: vec![
                Atom {
                    predicate: "reachable".to_string(),
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

        let reachable_facts = engine.get_facts("reachable");

        // Verify we have a reasonable number of reachability facts
        // The exact number depends on the graph structure
        assert!(
            reachable_facts.len() > 12, // At least the direct edges
            "Should have generated more reachability facts than just direct edges"
        );

        // Verify some specific reachability relationships
        let should_be_reachable = vec![
            ("a", "e"), // a -> b -> c -> d -> e
            ("a", "d"), // a -> f -> g -> d
            ("b", "m"), // b -> k -> l -> m
        ];

        for (from, to) in should_be_reachable {
            let fact = vec![from.to_string(), to.to_string()];
            assert!(
                reachable_facts.contains(&fact),
                "Should be able to reach {to} from {from}"
            );
        }
    }

    #[test]
    fn test_indexing_optimization() {
        let mut engine = DatalogEngine::new();

        // Add some facts to build indexes
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pizza".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("bob".to_string()),
                Term::Symbol("burger".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pasta".to_string()),
            ],
        });

        // Verify that indexes were created
        assert!(engine.indexes_by_pred.contains_key("likes"));
        let index = &engine.indexes_by_pred["likes"];

        // Test that we can retrieve facts by position
        let alice_likes = index.get_facts_by_position(0, "alice");
        assert!(alice_likes.is_some());
        let alice_facts = alice_likes.unwrap();
        assert_eq!(alice_facts.len(), 2); // alice likes pizza and pasta

        // Test retrieval by second position
        let pizza_lovers = index.get_facts_by_position(1, "pizza");
        assert!(pizza_lovers.is_some());
        let pizza_facts = pizza_lovers.unwrap();
        assert_eq!(pizza_facts.len(), 1); // only alice likes pizza

        // Add a rule to test indexed evaluation
        engine.add_rule(Rule {
            head: Atom {
                predicate: "food_lover".to_string(),
                terms: vec![Term::Variable("x".to_string())],
            },
            body: vec![Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Symbol("pizza".to_string()),
                ],
            }],
        });

        engine.run();

        let food_lovers = engine.get_facts("food_lover");
        assert_eq!(food_lovers.len(), 1);
        assert_eq!(food_lovers[0], vec!["alice".to_string()]);
    }

    #[test]
    fn test_index_updates_with_new_facts() {
        let mut engine = DatalogEngine::new();

        // Add initial fact
        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("alice".to_string())],
        });

        // Verify index was created and contains alice
        assert!(engine.indexes_by_pred.contains_key("person"));
        {
            let index = &engine.indexes_by_pred["person"];
            let alice_facts = index.get_facts_by_position(0, "alice").unwrap();
            assert_eq!(alice_facts.len(), 1);
        }

        // Add another fact and verify index is updated
        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("bob".to_string())],
        });

        {
            let index = &engine.indexes_by_pred["person"];
            let bob_facts = index.get_facts_by_position(0, "bob").unwrap();
            assert_eq!(bob_facts.len(), 1);
        }

        // Verify total facts
        let all_facts = engine.get_facts("person");
        assert_eq!(all_facts.len(), 2);
    }

    #[test]
    fn test_query_with_variables() {
        let mut engine = DatalogEngine::new();

        // Add some facts
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pizza".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("bob".to_string()),
                Term::Symbol("burger".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pasta".to_string()),
            ],
        });

        // Query: What does alice like? likes(alice, x)
        let query = Query {
            atom: Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Variable("x".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 2);

        // Extract the foods alice likes
        let mut foods: Vec<String> = results
            .iter()
            .map(|r| r.bindings.get("x").unwrap().clone())
            .collect();
        foods.sort();

        assert_eq!(foods, vec!["pasta".to_string(), "pizza".to_string()]);
    }

    #[test]
    fn test_query_with_multiple_variables() {
        let mut engine = DatalogEngine::new();

        // Add facts
        engine.add_fact(Atom {
            predicate: "parent".to_string(),
            terms: vec![
                Term::Symbol("john".to_string()),
                Term::Symbol("mary".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "parent".to_string(),
            terms: vec![
                Term::Symbol("mary".to_string()),
                Term::Symbol("alice".to_string()),
            ],
        });

        // Query: Who are all the parent-child pairs? parent(x, y)
        let query = Query {
            atom: Atom {
                predicate: "parent".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 2);

        // Check that we got the expected bindings
        let mut pairs: Vec<(String, String)> = results
            .iter()
            .map(|r| {
                (
                    r.bindings.get("x").unwrap().clone(),
                    r.bindings.get("y").unwrap().clone(),
                )
            })
            .collect();
        pairs.sort();

        let expected = vec![
            ("john".to_string(), "mary".to_string()),
            ("mary".to_string(), "alice".to_string()),
        ];

        assert_eq!(pairs, expected);
    }

    #[test]
    fn test_query_with_constants_only() {
        let mut engine = DatalogEngine::new();

        engine.add_fact(Atom {
            predicate: "friend".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("bob".to_string()),
            ],
        });

        // Query: Are alice and bob friends? friend(alice, bob)
        let query = Query {
            atom: Atom {
                predicate: "friend".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Symbol("bob".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 1);
        assert!(results[0].bindings.is_empty()); // No variables to bind
    }

    #[test]
    fn test_ask_method() {
        let mut engine = DatalogEngine::new();

        engine.add_fact(Atom {
            predicate: "student".to_string(),
            terms: vec![Term::Symbol("alice".to_string())],
        });

        // Ask if alice is a student
        let query_true = Query {
            atom: Atom {
                predicate: "student".to_string(),
                terms: vec![Term::Symbol("alice".to_string())],
            },
        };
        assert!(engine.ask(&query_true));

        // Ask if bob is a student (should be false)
        let query_false = Query {
            atom: Atom {
                predicate: "student".to_string(),
                terms: vec![Term::Symbol("bob".to_string())],
            },
        };
        assert!(!engine.ask(&query_false));
    }

    #[test]
    fn test_query_on_derived_facts() {
        let mut engine = DatalogEngine::new();

        // Add base facts
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });
        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("b".to_string()), Term::Symbol("c".to_string())],
        });

        // Add transitive closure rule
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

        // Run evaluation
        engine.run();

        // Query: What can we reach from 'a'? path(a, x)
        let query = Query {
            atom: Atom {
                predicate: "path".to_string(),
                terms: vec![
                    Term::Symbol("a".to_string()),
                    Term::Variable("x".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 2); // Should reach 'b' and 'c'

        let mut reachable: Vec<String> = results
            .iter()
            .map(|r| r.bindings.get("x").unwrap().clone())
            .collect();
        reachable.sort();

        assert_eq!(reachable, vec!["b".to_string(), "c".to_string()]);

        // Test ask method on derived facts
        let query_direct = Query {
            atom: Atom {
                predicate: "path".to_string(),
                terms: vec![Term::Symbol("a".to_string()), Term::Symbol("c".to_string())],
            },
        };
        assert!(engine.ask(&query_direct)); // a can reach c transitively
    }

    #[test]
    fn test_query_nonexistent_predicate() {
        let engine = DatalogEngine::new();

        let query = Query {
            atom: Atom {
                predicate: "nonexistent".to_string(),
                terms: vec![Term::Variable("x".to_string())],
            },
        };

        let results = engine.query(&query);
        assert!(results.is_empty());
        assert!(!engine.ask(&query));
    }

    #[test]
    fn test_query_with_indexed_optimization() {
        let mut engine = DatalogEngine::new();

        // Add many facts to test indexing
        for i in 0..100 {
            engine.add_fact(Atom {
                predicate: "number".to_string(),
                terms: vec![
                    Term::Symbol(format!("group_{}", i % 10)),
                    Term::Symbol(i.to_string()),
                ],
            });
        }

        // Query for all numbers in group_5
        let query = Query {
            atom: Atom {
                predicate: "number".to_string(),
                terms: vec![
                    Term::Symbol("group_5".to_string()),
                    Term::Variable("x".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 10); // Should find 10 numbers in group_5

        // Verify the numbers are correct
        let mut numbers: Vec<i32> = results
            .iter()
            .map(|r| r.bindings.get("x").unwrap().parse::<i32>().unwrap())
            .collect();
        numbers.sort_unstable();

        let expected: Vec<i32> = (0..10).map(|i| i * 10 + 5).collect();
        assert_eq!(numbers, expected);
    }

    #[test]
    fn test_repeated_variable_unification() {
        let mut engine = DatalogEngine::new();

        // Add a fact: same("a", "b") - different values
        engine.add_fact(Atom {
            predicate: "same".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });

        // Add a fact: same("c", "c") - same values
        engine.add_fact(Atom {
            predicate: "same".to_string(),
            terms: vec![Term::Symbol("c".to_string()), Term::Symbol("c".to_string())],
        });

        // Query: same(x, x) - should only match facts where both arguments are identical
        // This should only return x = "c", not x = "a" or x = "b"
        let query = Query {
            atom: Atom {
                predicate: "same".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("x".to_string()), // Same variable repeated
                ],
            },
        };

        let results = engine.query(&query);

        assert_eq!(
            results.len(),
            1,
            "Should only match facts where both arguments are identical"
        );

        if let Some(result) = results.first() {
            assert_eq!(result.bindings.get("x"), Some(&"c".to_string()));
        }
    }

    #[test]
    fn test_repeated_variable_unification_with_rule() {
        let mut engine = DatalogEngine::new();

        // Add facts
        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("alice".to_string())],
        });
        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("bob".to_string())],
        });

        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pizza".to_string()),
            ],
        });
        engine.add_fact(Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("bob".to_string()),
                Term::Symbol("bob".to_string()), // Bob likes himself
            ],
        });

        // Add rule: narcissist(x) <- likes(x, x)
        // This should only match people who like themselves
        engine.add_rule(Rule {
            head: Atom {
                predicate: "narcissist".to_string(),
                terms: vec![Term::Variable("x".to_string())],
            },
            body: vec![Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("x".to_string()), // Same variable repeated
                ],
            }],
        });

        engine.run();

        let narcissists = engine.get_facts("narcissist");

        // Should only have bob as narcissist
        assert_eq!(narcissists.len(), 1, "Should only have one narcissist");
        assert_eq!(narcissists[0], vec!["bob".to_string()]);
    }

    #[test]
    fn test_inconsistent_arity() {
        let mut engine = DatalogEngine::new();

        // Add a fact with arity 2
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });

        // Try to add a fact with different arity (3) for the same predicate
        // This should either be rejected or handled properly
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol("x".to_string()),
                Term::Symbol("y".to_string()),
                Term::Symbol("z".to_string()),
            ],
        });

        // The index was created with arity 2, but now we're trying to add arity 3
        // This could cause issues with indexing
        let facts = engine.get_facts("test");

        // Both facts should be present
        assert_eq!(facts.len(), 2);
        assert!(facts.contains(&vec!["a".to_string(), "b".to_string()]));
        assert!(facts.contains(&vec!["x".to_string(), "y".to_string(), "z".to_string()]));
    }

    #[test]
    fn test_inconsistent_arity_indexing_fix() {
        let mut engine = DatalogEngine::new();

        // Add a fact with arity 2
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });

        // Add a fact with arity 3 for the same predicate
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol("x".to_string()),
                Term::Symbol("y".to_string()),
                Term::Symbol("z".to_string()),
            ],
        });

        // Check the index structure again - should now be recreated with arity 3
        let index = &engine.indexes_by_pred["test"];

        // Now position 2 should be indexed properly
        let pos2_index = index.get_facts_by_position(2, "z");

        // This should now work because the index was recreated
        assert!(
            pos2_index.is_some(),
            "Position 2 should be indexed after recreation"
        );

        // Verify that old facts are still indexed properly
        let pos0_index = index.get_facts_by_position(0, "a");
        assert!(
            pos0_index.is_some(),
            "Position 0 should still work for old facts"
        );
    }

    #[test]
    fn test_empty_string_variable() {
        let mut engine = DatalogEngine::new();

        // Add facts with empty strings
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol(String::new()), // Empty string
                Term::Symbol("value".to_string()),
            ],
        });

        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol("other".to_string()),
                Term::Symbol(String::new()), // Empty string
            ],
        });

        let query = Query {
            atom: Atom {
                predicate: "test".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()), // Unbound variable
                    Term::Symbol("value".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bindings.get("x"), Some(&String::new()));
    }

    #[test]
    #[should_panic(
        expected = "Rule validation failed: Variable 'y' appears in head but not in body"
    )]
    fn test_unbound_variable_in_head_validation() {
        let mut engine = DatalogEngine::new();

        // Add a fact
        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("alice".to_string())],
        });

        // Try to add a rule where the head contains a variable not in the body
        // This should now panic with proper validation
        engine.add_rule(Rule {
            head: Atom {
                predicate: "result".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()), // x is in body
                    Term::Variable("y".to_string()), // y is NOT in body - should cause panic
                ],
            },
            body: vec![Atom {
                predicate: "person".to_string(),
                terms: vec![Term::Variable("x".to_string())],
            }],
        });
    }

    #[test]
    fn test_filter_logic_empty_string_ambiguity() {
        let mut engine = DatalogEngine::new();

        // Add facts with empty strings in different positions
        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol(String::new()), // Empty string at position 0
                Term::Symbol("value1".to_string()),
            ],
        });

        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol("actual".to_string()),
                Term::Symbol("value2".to_string()),
            ],
        });

        engine.add_fact(Atom {
            predicate: "test".to_string(),
            terms: vec![
                Term::Symbol("another".to_string()),
                Term::Symbol(String::new()), // Empty string at position 1
            ],
        });

        // Query 1: test(x, "value1") - x should be bound to ""
        let query1 = Query {
            atom: Atom {
                predicate: "test".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Symbol("value1".to_string()),
                ],
            },
        };

        let results1 = engine.query(&query1);
        assert_eq!(results1.len(), 1);
        assert_eq!(results1[0].bindings.get("x"), Some(&String::new()));

        // Query 2: test("actual", y) - y should be bound to "value2"
        let query2 = Query {
            atom: Atom {
                predicate: "test".to_string(),
                terms: vec![
                    Term::Symbol("actual".to_string()),
                    Term::Variable("y".to_string()),
                ],
            },
        };

        let results2 = engine.query(&query2);
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].bindings.get("y"), Some(&"value2".to_string()));

        // Query 3: test(x, y) - should return all facts
        let query3 = Query {
            atom: Atom {
                predicate: "test".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),
                    Term::Variable("y".to_string()),
                ],
            },
        };

        let results3 = engine.query(&query3);
        // Should return all 3 facts
        assert_eq!(results3.len(), 3);
    }

    #[test]
    fn test_unbound_variable_becomes_empty_string() {
        let mut engine = DatalogEngine::new();

        engine.add_fact(Atom {
            predicate: "person".to_string(),
            terms: vec![Term::Symbol("alice".to_string())],
        });

        let buggy_rule = Rule {
            head: Atom {
                predicate: "result".to_string(),
                terms: vec![
                    Term::Variable("x".to_string()),       // This will be bound
                    Term::Variable("unbound".to_string()), // This will NOT be bound!
                ],
            },
            body: vec![Atom {
                predicate: "person".to_string(),
                terms: vec![Term::Variable("x".to_string())],
            }],
        };

        // Bypass validation by directly pushing to rules vector
        engine.rules.push(buggy_rule);
        engine.run();

        let result_facts = engine.get_facts("result");
        assert_eq!(
            result_facts.len(),
            0,
            "No facts should be generated when head variables are unbound"
        );
    }

    #[test]
    fn test_demonstrate_empty_string_ambiguity() {
        let mut engine = DatalogEngine::new();

        engine.add_fact(Atom {
            predicate: "data".to_string(),
            terms: vec![
                Term::Symbol("key1".to_string()),
                Term::Symbol(String::new()), // Legitimate empty string
            ],
        });

        let query = Query {
            atom: Atom {
                predicate: "data".to_string(),
                terms: vec![
                    Term::Symbol("key1".to_string()),
                    Term::Variable("value".to_string()),
                ],
            },
        };

        let results = engine.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].bindings.get("value"), Some(&String::new()));
    }

    #[test]
    fn test_index_update_during_rule_evaluation() {
        let mut engine = DatalogEngine::new();

        // First, add some initial facts to establish indexes
        engine.add_fact(Atom {
            predicate: "path".to_string(),
            terms: vec![Term::Symbol("x".to_string()), Term::Symbol("y".to_string())],
        });

        // This should create an index for the "path" predicate
        assert!(engine.indexes_by_pred.contains_key("path"));

        engine.add_fact(Atom {
            predicate: "edge".to_string(),
            terms: vec![Term::Symbol("a".to_string()), Term::Symbol("b".to_string())],
        });

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

        // Run evaluation - this will derive path(a, b) from edge(a, b)
        engine.run();

        let path_facts = engine.get_facts("path");
        assert_eq!(path_facts.len(), 2); // x->y and a->b

        let index = &engine.indexes_by_pred["path"];

        let a_facts = index.get_facts_by_position(0, "a");
        assert!(
            a_facts.is_some(),
            "Index was not updated with derived fact path(a, b)"
        );

        let a_facts = a_facts.unwrap();
        let expected_fact = vec!["a".to_string(), "b".to_string()];
        assert!(
            a_facts.contains(&expected_fact),
            "Derived fact path(a, b) is not in the index at position 0 for value 'a'"
        );
    }

    #[test]
    fn test_filter_facts_chooses_most_selective_index_with_constants() {
        let mut engine = DatalogEngine::new();

        // Create a scenario where position 0 has few matches, position 1 has many
        // Add many facts where various people like "pizza"
        for i in 0..100 {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol(format!("person_{i}")),
                    Term::Symbol("pizza".to_string()),
                ],
            });
        }

        // Add a few facts where "alice" likes different things
        for food in ["pizza", "burger", "pasta"] {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Symbol(food.to_string()),
                ],
            });
        }

        let atom = Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("pizza".to_string()),
            ],
        };

        let all_facts = engine.facts_by_pred.get("likes").unwrap();
        let empty_binding = IndexMap::new();

        let filtered_facts = engine.filter_facts_by_bindings(&atom, all_facts, &empty_binding);

        assert_eq!(
            filtered_facts.len(),
            3,
            "Should return all facts where alice is at position 0"
        );

        let expected_fact = vec!["alice".to_string(), "pizza".to_string()];
        assert!(
            filtered_facts.contains(&expected_fact),
            "Should contain the alice-pizza fact"
        );

        // Verify all returned facts have alice at position 0
        for fact in &filtered_facts {
            assert_eq!(
                fact[0], "alice",
                "All facts should have alice at position 0"
            );
        }
    }

    #[test]
    fn test_filter_facts_with_variable_uses_available_index() {
        let mut engine = DatalogEngine::new();

        // Create a scenario where position 0 has few matches, position 1 has many
        for i in 0..100 {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol(format!("person_{i}")),
                    Term::Symbol("pizza".to_string()),
                ],
            });
        }

        // Add a few facts where "alice" likes different things
        for food in ["pizza", "burger", "pasta"] {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Symbol(food.to_string()),
                ],
            });
        }

        let atom_with_var = Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Variable("x".to_string()),
            ],
        };

        let all_facts = engine.facts_by_pred.get("likes").unwrap();
        let empty_binding = IndexMap::new();

        let filtered_with_var =
            engine.filter_facts_by_bindings(&atom_with_var, all_facts, &empty_binding);

        assert_eq!(
            filtered_with_var.len(),
            3,
            "Should return all facts where alice is at position 0"
        );
    }

    #[test]
    fn test_filter_facts_selects_most_selective_among_multiple_indexes() {
        let mut engine = DatalogEngine::new();

        // Add facts where "alice" appears in many relationships
        for i in 0..50 {
            engine.add_fact(Atom {
                predicate: "knows".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Symbol(format!("person_{i}")),
                ],
            });
        }

        // Add facts where few people know "bob"
        for person in ["alice", "charlie"] {
            engine.add_fact(Atom {
                predicate: "knows".to_string(),
                terms: vec![
                    Term::Symbol(person.to_string()),
                    Term::Symbol("bob".to_string()),
                ],
            });
        }

        let atom_selective = Atom {
            predicate: "knows".to_string(),
            terms: vec![
                Term::Symbol("alice".to_string()),
                Term::Symbol("bob".to_string()),
            ],
        };

        let all_facts = engine.facts_by_pred.get("knows").unwrap();
        let empty_binding = IndexMap::new();

        let filtered_selective =
            engine.filter_facts_by_bindings(&atom_selective, all_facts, &empty_binding);

        assert_eq!(
            filtered_selective.len(),
            2,
            "Should return facts from the most selective index (position 1 for 'bob')"
        );

        // Verify all returned facts have bob at position 1
        for fact in &filtered_selective {
            assert_eq!(fact[1], "bob", "All facts should have bob at position 1");
        }
    }

    #[test]
    fn test_filter_facts_with_variable_bindings_from_context() {
        let mut engine = DatalogEngine::new();

        // Create a scenario where position 0 has few matches, position 1 has many
        for i in 0..100 {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol(format!("person_{i}")),
                    Term::Symbol("pizza".to_string()),
                ],
            });
        }

        // Add a few facts where "alice" likes different things
        for food in ["pizza", "burger", "pasta"] {
            engine.add_fact(Atom {
                predicate: "likes".to_string(),
                terms: vec![
                    Term::Symbol("alice".to_string()),
                    Term::Symbol(food.to_string()),
                ],
            });
        }

        let mut binding = IndexMap::new();
        binding.insert("person".to_string(), "alice".to_string());
        binding.insert("food".to_string(), "pizza".to_string());

        let atom_bound_vars = Atom {
            predicate: "likes".to_string(),
            terms: vec![
                Term::Variable("person".to_string()),
                Term::Variable("food".to_string()),
            ],
        };

        let all_facts = engine.facts_by_pred.get("likes").unwrap();
        let filtered_bound = engine.filter_facts_by_bindings(&atom_bound_vars, all_facts, &binding);

        assert_eq!(
            filtered_bound.len(),
            3,
            "Should return facts from most selective index"
        );

        // Verify all facts have alice at position 0
        for fact in &filtered_bound {
            assert_eq!(
                fact[0], "alice",
                "All facts should have alice at position 0"
            );
        }
    }
}
