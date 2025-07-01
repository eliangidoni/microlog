# microlog

## Key Features
Core Components:
- Term: Represents variables or symbols
- Atom: Predicates with terms (e.g., edge(a, b))
- Rule: Horn clauses (head ← body)
- DatalogEngine: Evaluation core with relations storage

Evaluation Algorithm:
- Naïve evaluation until fixed point
- Join processing through variable binding
- Tuple-matching with unification

Optimizations:
- Hash-based relation storage for deduplication
- Iterative rule application
- Binding propagation during joins

Example Included:
- Computes transitive closure (path finding)
- Demonstrates fact insertion and rule definition

## Implementation Notes
Conciseness:
- Complete in ~150 lines
- Avoids complex dependencies
- Uses standard Rust collections

Limitations:
- No negation or aggregation
- Basic unification without type checking
- Simplified indexing (full scans)

Extensibility Points:
- Add semi-naïve evaluation
- Implement indexing for joins
- Support recursion optimizations

This implementation demonstrates core Datalog concepts while maintaining minimalism. The example computes transitive closure over a graph, showing how new facts are derived through recursive rule application until no new tuples emerge. For production use, consider extending with:

- Magic sets optimization
- Incremental evaluation
- Advanced join strategies

