"""
CSP (Constraint Satisfaction Problem) module for ARC Prize 2025.

This module provides a comprehensive framework for solving constraint satisfaction
problems, with a focus on grid-based puzzles and pattern recognition tasks.
"""

# Core CSP classes
from .core import (
    CSP,
    Variable,
    Domain,
    Constraint,
    UnaryConstraint,
    BinaryConstraint,
    NAryConstraint,
    ConstraintType,
    VariableName,
    DomainValue,
    create_grid_csp,
    create_pattern_csp
)

# Arc consistency algorithms
from .arc_consistency import (
    AC3,
    AC4,
    MAC,
    ArcConsistencyStats,
    achieve_arc_consistency,
    is_arc_consistent,
    PathConsistency
)

# Search algorithms
from .search import (
    BacktrackingSearch,
    ForwardCheckingSearch,
    MACSearch,
    MinConflicts,
    HybridSearch,
    VariableOrdering,
    ValueOrdering,
    SearchStats,
    solve_csp,
    find_all_solutions
)

# ARC-specific constraints
from .constraints import (
    AllDifferentConstraint,
    SamePatternConstraint,
    TransformationConstraint,
    ColorMappingConstraint,
    SymmetryConstraint,
    AdjacentConstraint,
    CountConstraint,
    ConnectedComponentConstraint,
    RegionConstraint,
    create_sudoku_constraints
)

__version__ = "1.0.0"

__all__ = [
    # Core
    'CSP',
    'Variable',
    'Domain',
    'Constraint',
    'UnaryConstraint',
    'BinaryConstraint',
    'NAryConstraint',
    'ConstraintType',
    'VariableName',
    'DomainValue',
    'create_grid_csp',
    'create_pattern_csp',
    
    # Arc consistency
    'AC3',
    'AC4',
    'MAC',
    'ArcConsistencyStats',
    'achieve_arc_consistency',
    'is_arc_consistent',
    'PathConsistency',
    
    # Search
    'BacktrackingSearch',
    'ForwardCheckingSearch',
    'MACSearch',
    'MinConflicts',
    'HybridSearch',
    'VariableOrdering',
    'ValueOrdering',
    'SearchStats',
    'solve_csp',
    'find_all_solutions',
    
    # Constraints
    'AllDifferentConstraint',
    'SamePatternConstraint',
    'TransformationConstraint',
    'ColorMappingConstraint',
    'SymmetryConstraint',
    'AdjacentConstraint',
    'CountConstraint',
    'ConnectedComponentConstraint',
    'RegionConstraint',
    'create_sudoku_constraints'
]