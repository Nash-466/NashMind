"""
Core CSP (Constraint Satisfaction Problem) classes for ARC Prize 2025.

This module provides the fundamental building blocks for defining and solving
constraint satisfaction problems, optimized for grid-based puzzles with up to
30x30 cells and 10 colors.
"""

from typing import Dict, List, Set, Tuple, Any, Optional, Callable, Generic, TypeVar
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Type definitions
T = TypeVar('T')
VariableName = str
DomainValue = Any


class ConstraintType(Enum):
    """Types of constraints supported by the CSP solver."""
    UNARY = 1
    BINARY = 2
    N_ARY = 3
    GLOBAL = 4


@dataclass
class Variable:
    """
    Represents a variable in a CSP.
    
    Attributes:
        name: Unique identifier for the variable
        domain: Initial domain of possible values
        position: Optional (row, col) position for grid-based problems
        metadata: Additional metadata for the variable
    """
    name: VariableName
    domain: Set[DomainValue]
    position: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Variable) and self.name == other.name
    
    def __repr__(self) -> str:
        return f"Variable({self.name}, |domain|={len(self.domain)})"
    
    def copy(self) -> 'Variable':
        """Create a deep copy of the variable."""
        return Variable(
            name=self.name,
            domain=self.domain.copy(),
            position=self.position,
            metadata=self.metadata.copy()
        )


@dataclass
class Domain:
    """
    Manages domains for all variables in a CSP.
    
    Provides efficient domain manipulation and tracking for constraint propagation.
    """
    _domains: Dict[VariableName, Set[DomainValue]] = field(default_factory=dict)
    _initial_domains: Dict[VariableName, Set[DomainValue]] = field(default_factory=dict)
    _domain_changes: List[Tuple[VariableName, Set[DomainValue]]] = field(default_factory=list)
    
    def __init__(self, variables: Optional[List[Variable]] = None):
        """Initialize domains from a list of variables."""
        self._domains = {}
        self._initial_domains = {}
        self._domain_changes = []
        
        if variables:
            for var in variables:
                self._domains[var.name] = var.domain.copy()
                self._initial_domains[var.name] = var.domain.copy()
    
    def get(self, var_name: VariableName) -> Set[DomainValue]:
        """Get the current domain of a variable."""
        return self._domains.get(var_name, set())
    
    def set(self, var_name: VariableName, values: Set[DomainValue]) -> None:
        """Set the domain of a variable."""
        old_domain = self._domains.get(var_name, set()).copy()
        self._domains[var_name] = values.copy()
        self._domain_changes.append((var_name, old_domain))
    
    def remove(self, var_name: VariableName, value: DomainValue) -> bool:
        """
        Remove a value from a variable's domain.
        
        Returns:
            True if the value was removed, False if it wasn't in the domain
        """
        if var_name in self._domains and value in self._domains[var_name]:
            old_domain = self._domains[var_name].copy()
            self._domains[var_name].remove(value)
            self._domain_changes.append((var_name, old_domain))
            return True
        return False
    
    def is_empty(self, var_name: VariableName) -> bool:
        """Check if a variable's domain is empty."""
        return len(self._domains.get(var_name, set())) == 0
    
    def is_singleton(self, var_name: VariableName) -> bool:
        """Check if a variable's domain has exactly one value."""
        return len(self._domains.get(var_name, set())) == 1
    
    def get_singleton_value(self, var_name: VariableName) -> Optional[DomainValue]:
        """Get the single value if domain is a singleton, None otherwise."""
        domain = self._domains.get(var_name, set())
        if len(domain) == 1:
            return next(iter(domain))
        return None
    
    def save_state(self) -> int:
        """Save the current state and return a checkpoint ID."""
        checkpoint = len(self._domain_changes)
        return checkpoint
    
    def restore_state(self, checkpoint: int) -> None:
        """Restore domains to a previous checkpoint."""
        while len(self._domain_changes) > checkpoint:
            var_name, old_domain = self._domain_changes.pop()
            self._domains[var_name] = old_domain
    
    def copy(self) -> 'Domain':
        """Create a deep copy of the domain manager."""
        new_domain = Domain()
        new_domain._domains = {k: v.copy() for k, v in self._domains.items()}
        new_domain._initial_domains = {k: v.copy() for k, v in self._initial_domains.items()}
        new_domain._domain_changes = self._domain_changes.copy()
        return new_domain
    
    def reset(self) -> None:
        """Reset all domains to their initial values."""
        self._domains = {k: v.copy() for k, v in self._initial_domains.items()}
        self._domain_changes.clear()


class Constraint(ABC):
    """
    Abstract base class for all constraints.
    
    A constraint defines a relationship that must hold between variables.
    """
    
    def __init__(self, variables: List[VariableName], name: Optional[str] = None):
        """
        Initialize a constraint.
        
        Args:
            variables: List of variable names involved in this constraint
            name: Optional name for the constraint
        """
        self.variables = variables
        self.name = name or f"Constraint({','.join(variables)})"
    
    @abstractmethod
    def is_satisfied(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """
        Check if the constraint is satisfied by the given assignment.
        
        Args:
            assignment: Current variable assignments
            
        Returns:
            True if the constraint is satisfied, False otherwise
        """
        pass
    
    @abstractmethod
    def prune_domains(self, domains: Domain) -> bool:
        """
        Prune inconsistent values from variable domains.
        
        Args:
            domains: Current domains of all variables
            
        Returns:
            True if any domain was changed, False otherwise
        """
        pass
    
    def get_scope(self) -> List[VariableName]:
        """Get the variables involved in this constraint."""
        return self.variables
    
    def __repr__(self) -> str:
        return self.name


class UnaryConstraint(Constraint):
    """
    A constraint involving a single variable.
    
    Useful for imposing restrictions on individual cells in ARC puzzles.
    """
    
    def __init__(self, variable: VariableName, predicate: Callable[[DomainValue], bool],
                 name: Optional[str] = None):
        """
        Initialize a unary constraint.
        
        Args:
            variable: The variable this constraint applies to
            predicate: Function that returns True for valid values
            name: Optional name for the constraint
        """
        super().__init__([variable], name)
        self.predicate = predicate
    
    def is_satisfied(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """Check if the constraint is satisfied."""
        if self.variables[0] not in assignment:
            return True  # Unassigned variables don't violate constraints
        return self.predicate(assignment[self.variables[0]])
    
    def prune_domains(self, domains: Domain) -> bool:
        """Remove values that don't satisfy the predicate."""
        changed = False
        var_name = self.variables[0]
        current_domain = domains.get(var_name).copy()
        
        for value in current_domain:
            if not self.predicate(value):
                domains.remove(var_name, value)
                changed = True
        
        return changed


class BinaryConstraint(Constraint):
    """
    A constraint involving exactly two variables.
    
    Most common constraint type in CSPs, efficient for arc consistency.
    """
    
    def __init__(self, var1: VariableName, var2: VariableName,
                 relation: Callable[[DomainValue, DomainValue], bool],
                 name: Optional[str] = None):
        """
        Initialize a binary constraint.
        
        Args:
            var1: First variable
            var2: Second variable
            relation: Function that returns True for valid value pairs
            name: Optional name for the constraint
        """
        super().__init__([var1, var2], name)
        self.relation = relation
    
    def is_satisfied(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """Check if the constraint is satisfied."""
        var1, var2 = self.variables
        
        # If either variable is unassigned, the constraint is not violated
        if var1 not in assignment or var2 not in assignment:
            return True
        
        return self.relation(assignment[var1], assignment[var2])
    
    def prune_domains(self, domains: Domain) -> bool:
        """Prune domains using arc consistency."""
        changed = False
        var1, var2 = self.variables
        
        # Prune var1's domain
        domain1 = domains.get(var1).copy()
        domain2 = domains.get(var2)
        
        for val1 in domain1:
            has_support = False
            for val2 in domain2:
                if self.relation(val1, val2):
                    has_support = True
                    break
            
            if not has_support:
                domains.remove(var1, val1)
                changed = True
        
        # Prune var2's domain
        domain1 = domains.get(var1)
        domain2 = domains.get(var2).copy()
        
        for val2 in domain2:
            has_support = False
            for val1 in domain1:
                if self.relation(val1, val2):
                    has_support = True
                    break
            
            if not has_support:
                domains.remove(var2, val2)
                changed = True
        
        return changed
    
    def has_support(self, value: DomainValue, var_idx: int, 
                    other_domain: Set[DomainValue]) -> bool:
        """
        Check if a value has support in the other variable's domain.
        
        Args:
            value: The value to check
            var_idx: Index of the variable (0 or 1)
            other_domain: Domain of the other variable
            
        Returns:
            True if the value has at least one supporting value
        """
        for other_value in other_domain:
            if var_idx == 0:
                if self.relation(value, other_value):
                    return True
            else:
                if self.relation(other_value, value):
                    return True
        return False


class NAryConstraint(Constraint):
    """
    A constraint involving multiple variables (more than 2).
    
    Used for complex relationships in ARC puzzles.
    """
    
    def __init__(self, variables: List[VariableName],
                 predicate: Callable[[Dict[VariableName, DomainValue]], bool],
                 name: Optional[str] = None):
        """
        Initialize an n-ary constraint.
        
        Args:
            variables: List of variables involved
            predicate: Function that validates assignments
            name: Optional name for the constraint
        """
        super().__init__(variables, name)
        self.predicate = predicate
    
    def is_satisfied(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """Check if the constraint is satisfied."""
        # Extract relevant assignments
        relevant = {var: assignment[var] for var in self.variables if var in assignment}
        
        # If not all variables are assigned, we can't violate the constraint yet
        if len(relevant) < len(self.variables):
            return True
        
        return self.predicate(relevant)
    
    def prune_domains(self, domains: Domain) -> bool:
        """
        Prune domains using generalized arc consistency.
        
        This is more expensive than binary constraint propagation.
        """
        changed = False
        
        # For each variable in the constraint
        for i, var in enumerate(self.variables):
            current_domain = domains.get(var).copy()
            other_vars = [v for j, v in enumerate(self.variables) if j != i]
            
            # Check each value in the current variable's domain
            for value in current_domain:
                has_support = self._has_support(var, value, other_vars, domains)
                
                if not has_support:
                    domains.remove(var, value)
                    changed = True
        
        return changed
    
    def _has_support(self, var: VariableName, value: DomainValue,
                     other_vars: List[VariableName], domains: Domain) -> bool:
        """
        Check if a value has support (recursive backtracking).
        
        This performs a mini-search to see if there exists an assignment
        to other variables that satisfies the constraint.
        """
        # Create partial assignment
        assignment = {var: value}
        return self._check_support_recursive(assignment, other_vars, 0, domains)
    
    def _check_support_recursive(self, assignment: Dict[VariableName, DomainValue],
                                other_vars: List[VariableName], idx: int,
                                domains: Domain) -> bool:
        """Recursive helper for checking support."""
        if idx == len(other_vars):
            # All variables assigned, check constraint
            return self.predicate(assignment)
        
        var = other_vars[idx]
        for val in domains.get(var):
            assignment[var] = val
            if self._check_support_recursive(assignment, other_vars, idx + 1, domains):
                del assignment[var]
                return True
            del assignment[var]
        
        return False


class CSP:
    """
    Represents a Constraint Satisfaction Problem.
    
    Manages variables, domains, and constraints for ARC puzzle solving.
    """
    
    def __init__(self, name: str = "CSP"):
        """Initialize an empty CSP."""
        self.name = name
        self.variables: Dict[VariableName, Variable] = {}
        self.constraints: List[Constraint] = []
        self.domains: Domain = Domain()
        
        # Constraint indexing for efficient lookup
        self.constraints_by_var: Dict[VariableName, List[Constraint]] = defaultdict(list)
        self.binary_constraints: Dict[Tuple[VariableName, VariableName], List[BinaryConstraint]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'nodes_explored': 0,
            'backtracks': 0,
            'domain_reductions': 0,
            'constraint_checks': 0
        }
    
    def add_variable(self, variable: Variable) -> None:
        """Add a variable to the CSP."""
        self.variables[variable.name] = variable
        self.domains.set(variable.name, variable.domain)
    
    def add_variables(self, variables: List[Variable]) -> None:
        """Add multiple variables to the CSP."""
        for var in variables:
            self.add_variable(var)
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the CSP."""
        self.constraints.append(constraint)
        
        # Update indices
        for var in constraint.get_scope():
            self.constraints_by_var[var].append(constraint)
        
        # Special indexing for binary constraints
        if isinstance(constraint, BinaryConstraint) and len(constraint.variables) == 2:
            var1, var2 = constraint.variables
            self.binary_constraints[(var1, var2)].append(constraint)
            self.binary_constraints[(var2, var1)].append(constraint)
    
    def add_constraints(self, constraints: List[Constraint]) -> None:
        """Add multiple constraints to the CSP."""
        for constraint in constraints:
            self.add_constraint(constraint)
    
    def get_constraints_for_variable(self, var_name: VariableName) -> List[Constraint]:
        """Get all constraints involving a specific variable."""
        return self.constraints_by_var.get(var_name, [])
    
    def get_binary_constraints(self, var1: VariableName, var2: VariableName) -> List[BinaryConstraint]:
        """Get binary constraints between two specific variables."""
        return self.binary_constraints.get((var1, var2), [])
    
    def is_consistent(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """
        Check if an assignment is consistent with all constraints.
        
        Args:
            assignment: Current variable assignments
            
        Returns:
            True if all constraints are satisfied
        """
        for constraint in self.constraints:
            self.stats['constraint_checks'] += 1
            if not constraint.is_satisfied(assignment):
                return False
        return True
    
    def is_complete(self, assignment: Dict[VariableName, DomainValue]) -> bool:
        """Check if an assignment is complete (all variables assigned)."""
        return len(assignment) == len(self.variables)
    
    def get_unassigned_variables(self, assignment: Dict[VariableName, DomainValue]) -> List[VariableName]:
        """Get list of unassigned variables."""
        return [var for var in self.variables if var not in assignment]
    
    def copy(self) -> 'CSP':
        """Create a deep copy of the CSP."""
        new_csp = CSP(self.name)
        
        # Copy variables
        for var in self.variables.values():
            new_csp.add_variable(var.copy())
        
        # Copy constraints (note: constraints themselves are not deep copied)
        new_csp.constraints = self.constraints.copy()
        new_csp.constraints_by_var = defaultdict(list)
        for var, constraints in self.constraints_by_var.items():
            new_csp.constraints_by_var[var] = constraints.copy()
        
        new_csp.binary_constraints = defaultdict(list)
        for key, constraints in self.binary_constraints.items():
            new_csp.binary_constraints[key] = constraints.copy()
        
        # Copy domains
        new_csp.domains = self.domains.copy()
        
        # Copy stats
        new_csp.stats = self.stats.copy()
        
        return new_csp
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'nodes_explored': 0,
            'backtracks': 0,
            'domain_reductions': 0,
            'constraint_checks': 0
        }
    
    def __repr__(self) -> str:
        return (f"CSP(name={self.name}, "
                f"variables={len(self.variables)}, "
                f"constraints={len(self.constraints)})")


def create_grid_csp(rows: int, cols: int, colors: Set[int],
                    name: str = "GridCSP") -> CSP:
    """
    Create a CSP for a grid-based problem.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        colors: Set of possible colors (usually 0-9 for ARC)
        name: Name for the CSP
        
    Returns:
        A CSP with variables for each grid cell
    """
    csp = CSP(name)
    
    # Create a variable for each grid cell
    for row in range(rows):
        for col in range(cols):
            var_name = f"cell_{row}_{col}"
            var = Variable(
                name=var_name,
                domain=colors.copy(),
                position=(row, col)
            )
            csp.add_variable(var)
    
    return csp


def create_pattern_csp(pattern: np.ndarray, name: str = "PatternCSP") -> CSP:
    """
    Create a CSP from a pattern with some unknown values.
    
    Args:
        pattern: NumPy array where -1 indicates unknown values
        name: Name for the CSP
        
    Returns:
        A CSP representing the pattern completion problem
    """
    rows, cols = pattern.shape
    
    # Determine the set of colors from known values
    known_values = pattern[pattern >= 0]
    if len(known_values) > 0:
        colors = set(np.unique(known_values))
        # Add a few more colors in case new ones are needed
        max_color = max(colors) if colors else 0
        for i in range(max_color + 1, min(max_color + 3, 10)):
            colors.add(i)
    else:
        colors = set(range(10))  # Default ARC colors
    
    csp = create_grid_csp(rows, cols, colors, name)
    
    # Add unary constraints for known values
    for row in range(rows):
        for col in range(cols):
            if pattern[row, col] >= 0:
                var_name = f"cell_{row}_{col}"
                value = int(pattern[row, col])
                constraint = UnaryConstraint(
                    var_name,
                    lambda v, val=value: v == val,
                    name=f"Fixed_{row}_{col}={value}"
                )
                csp.add_constraint(constraint)
    
    return csp