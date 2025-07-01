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

/// The Datalog evaluation engine
#[derive(Debug)]
pub struct DatalogEngine {
    relations: IndexMap<String, IndexSet<Vec<String>>>,
    rules: Vec<Rule>,
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
            relations: IndexMap::new(),
            rules: Vec::new(),
        }
    }

    /// Add a fact to the knowledge base
    ///
    /// # Panics
    ///
    /// Panics if the atom contains any variables (only symbols are allowed in facts)
    pub fn add_fact(&mut self, atom: Atom) {
        let values = atom
            .terms
            .into_iter()
            .map(|term| match term {
                Term::Symbol(s) => s,
                Term::Variable(_) => panic!("Facts must contain only symbols"),
            })
            .collect();

        self.relations
            .entry(atom.predicate)
            .or_default()
            .insert(values);
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Run the Datalog evaluation to fixed point
    pub fn run(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            for rule in &self.rules {
                let mut new_facts = self.evaluate_rule(rule);
                let relation = self
                    .relations
                    .entry(rule.head.predicate.clone())
                    .or_default();

                for fact in new_facts.drain(..) {
                    if relation.insert(fact) {
                        changed = true;
                    }
                }
            }
        }
    }

    /// Evaluate a single rule
    fn evaluate_rule(&self, rule: &Rule) -> IndexSet<Vec<String>> {
        // Simplified join implementation
        let mut results = IndexSet::new();
        let mut bindings: Vec<IndexMap<String, String>> = vec![IndexMap::new()];

        for atom in &rule.body {
            let mut new_bindings = Vec::new();
            let relation = self
                .relations
                .get(&atom.predicate)
                .cloned()
                .unwrap_or_default();

            for binding in &bindings {
                for tuple in &relation {
                    if let Some(new_binding) = Self::match_atom(atom, tuple, binding) {
                        new_bindings.push(new_binding);
                    }
                }
            }
            bindings = new_bindings;
        }

        for binding in bindings {
            let mut fact = Vec::new();
            for term in &rule.head.terms {
                match term {
                    Term::Variable(var) => fact.push(binding[var].clone()),
                    Term::Symbol(sym) => fact.push(sym.clone()),
                }
            }
            results.insert(fact);
        }

        results
    }

    /// Match an atom against a tuple with current bindings
    fn match_atom(
        atom: &Atom,
        tuple: &[String],
        binding: &IndexMap<String, String>,
    ) -> Option<IndexMap<String, String>> {
        let mut new_binding = binding.clone();

        for (i, term) in atom.terms.iter().enumerate() {
            match term {
                Term::Symbol(sym) => {
                    if sym != &tuple[i] {
                        return None;
                    }
                }
                Term::Variable(var) => {
                    if let Some(bound) = binding.get(var) {
                        if bound != &tuple[i] {
                            return None;
                        }
                    } else {
                        new_binding.insert(var.clone(), tuple[i].clone());
                    }
                }
            }
        }
        Some(new_binding)
    }

    /// Get all facts for a predicate
    #[must_use]
    pub fn get_facts(&self, predicate: &str) -> Vec<Vec<String>> {
        self.relations
            .get(predicate)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
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
}
