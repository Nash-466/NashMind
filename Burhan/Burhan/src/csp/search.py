"""
Search algorithms for CSP solving in ARC Prize 2025.

This module implements various search strategies for finding solutions
to constraint satisfaction problems.
"""

from typing import Dict, List, Optional, Tuple, Set, Callable
from collections import defaultdict
import random
import time
from dataclasses import dataclass
from enum import Enum

from .core import (
    CSP, Variable, Domain, Constraint,
    VariableName, DomainValue
)
from .arc_consistency import MAC, AC3, achieve_arc_consistency


class VariableOrdering(Enum):
    """Variable ordering heuristics for search."""
    DEFAULT = "default"  # Lexicographic ordering
    MRV = "mrv"  # Minimum Remaining Values (Most Constrained Variable)
    DEGREE = "degree"  # Degree heuristic
    MRV_DEGREE = "mrv_degree"  # MRV with degree as tiebreaker
    RANDOM = "random"  # Random ordering


class ValueOrdering(Enum):
    """Value ordering heuristics for search."""
    DEFAULT = "default"  # Natural ordering
    LCV = "lcv"  # Least Constraining Value
    RANDOM = "random"  # Random ordering
    MIN = "min"  # Minimum value first
    MAX = "max"  # Maximum value first


@dataclass
class SearchStats:
    """Statistics for search algorithms."""
    nodes_explored: int = 0
    backtracks: int = 0
    solutions_found: int = 0
    time_elapsed: float = 0.0
    constraint_checks: int = 0
    domain_reductions: int = 0
    
    def reset(self):
        """Reset all statistics."""
        self.nodes_explored = 0
        self.backtracks = 0
        self.solutions_found = 0
        self.time_elapsed = 0.0
        self.constraint_checks = 0
        self.domain_reductions = 0


class BacktrackingSearch:
    """
    Standard backtracking search algorithm for CSPs.
    
    Systematically explores the search space using depth-first search
    with constraint checking.
    """
    
    def __init__(self, csp: CSP,
                 variable_ordering: VariableOrdering = VariableOrdering.MRV_DEGREE,
                 value_ordering: ValueOrdering = ValueOrdering.LCV):
        """
        Initialize backtracking search.
        
        Args:
            csp: The CSP to solve
            variable_ordering: Heuristic for selecting next variable
            value_ordering: Heuristic for ordering domain values
        """
        self.csp = csp
        self.variable_ordering = variable_ordering
        self.value_ordering = value_ordering
        self.stats = SearchStats()
    
    def solve(self, assignment: Optional[Dict[VariableName, DomainValue]] = None,
              domains: Optional[Domain] = None) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Find a solution to the CSP using backtracking.
        
        Args:
            assignment: Initial partial assignment
            domains: Initial domains (uses CSP's domains if not provided)
            
        Returns:
            Complete assignment if solution found, None otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        if assignment is None:
            assignment = {}
        
        if domains is None:
            domains = self.csp.domains.copy()
        
        result = self._backtrack(assignment, domains)
        
        self.stats.time_elapsed = time.time() - start_time
        return result
    
    def _backtrack(self, assignment: Dict[VariableName, DomainValue],
                   domains: Domain) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Recursive backtracking algorithm.
        
        Args:
            assignment: Current partial assignment
            domains: Current domains
            
        Returns:
            Complete assignment if found, None otherwise
        """
        self.stats.nodes_explored += 1
        
        # Check if assignment is complete
        if self.csp.is_complete(assignment):
            if self.csp.is_consistent(assignment):
                self.stats.solutions_found += 1
                return assignment.copy()
            return None
        
        # Select next variable to assign
        var = self._select_unassigned_variable(assignment, domains)
        if var is None:
            return None
        
        # Try each value in the domain
        for value in self._order_domain_values(var, assignment, domains):
            # Check if value is consistent with current assignment
            assignment[var] = value
            self.stats.constraint_checks += 1
            
            if self._is_consistent_assignment(var, value, assignment):
                # Save current state
                checkpoint = domains.save_state()
                
                # Apply inference (constraint propagation)
                inference_result = self._inference(var, value, assignment, domains)
                
                if inference_result:
                    # Recurse with updated assignment
                    result = self._backtrack(assignment, domains)
                    
                    if result is not None:
                        return result
                
                # Restore domains
                domains.restore_state(checkpoint)
            
            # Remove assignment
            del assignment[var]
            self.stats.backtracks += 1
        
        return None
    
    def _select_unassigned_variable(self, assignment: Dict[VariableName, DomainValue],
                                   domains: Domain) -> Optional[VariableName]:
        """
        Select the next variable to assign using the specified heuristic.
        
        Args:
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            Name of the selected variable, or None if all assigned
        """
        unassigned = self.csp.get_unassigned_variables(assignment)
        
        if not unassigned:
            return None
        
        if self.variable_ordering == VariableOrdering.MRV:
            # Minimum Remaining Values
            return min(unassigned, key=lambda v: len(domains.get(v)))
        
        elif self.variable_ordering == VariableOrdering.DEGREE:
            # Degree heuristic - most constrained variable
            return max(unassigned, key=lambda v: self._get_degree(v, assignment))
        
        elif self.variable_ordering == VariableOrdering.MRV_DEGREE:
            # MRV with degree as tiebreaker
            min_domain_size = min(len(domains.get(v)) for v in unassigned)
            candidates = [v for v in unassigned if len(domains.get(v)) == min_domain_size]
            
            if len(candidates) == 1:
                return candidates[0]
            
            # Use degree as tiebreaker
            return max(candidates, key=lambda v: self._get_degree(v, assignment))
        
        elif self.variable_ordering == VariableOrdering.RANDOM:
            return random.choice(unassigned)
        
        else:  # DEFAULT - lexicographic
            return unassigned[0]
    
    def _get_degree(self, var: VariableName,
                   assignment: Dict[VariableName, DomainValue]) -> int:
        """
        Get the degree of a variable (number of constraints with unassigned variables).
        
        Args:
            var: Variable to check
            assignment: Current assignment
            
        Returns:
            Number of constraints with unassigned variables
        """
        degree = 0
        for constraint in self.csp.get_constraints_for_variable(var):
            for other_var in constraint.get_scope():
                if other_var != var and other_var not in assignment:
                    degree += 1
                    break
        return degree
    
    def _order_domain_values(self, var: VariableName,
                           assignment: Dict[VariableName, DomainValue],
                           domains: Domain) -> List[DomainValue]:
        """
        Order domain values using the specified heuristic.
        
        Args:
            var: Variable whose values to order
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            Ordered list of domain values
        """
        values = list(domains.get(var))
        
        if self.value_ordering == ValueOrdering.LCV:
            # Least Constraining Value - choose value that rules out fewest choices
            def count_conflicts(value):
                conflicts = 0
                for constraint in self.csp.get_constraints_for_variable(var):
                    for other_var in constraint.get_scope():
                        if other_var != var and other_var not in assignment:
                            for other_val in domains.get(other_var):
                                test_assignment = assignment.copy()
                                test_assignment[var] = value
                                test_assignment[other_var] = other_val
                                if not constraint.is_satisfied(test_assignment):
                                    conflicts += 1
                return conflicts
            
            values.sort(key=count_conflicts)
        
        elif self.value_ordering == ValueOrdering.RANDOM:
            random.shuffle(values)
        
        elif self.value_ordering == ValueOrdering.MIN:
            values.sort()
        
        elif self.value_ordering == ValueOrdering.MAX:
            values.sort(reverse=True)
        
        # DEFAULT - natural ordering
        return values
    
    def _is_consistent_assignment(self, var: VariableName, value: DomainValue,
                                 assignment: Dict[VariableName, DomainValue]) -> bool:
        """
        Check if assigning var=value is consistent with current assignment.
        
        Args:
            var: Variable being assigned
            value: Value being assigned
            assignment: Current assignment (including var=value)
            
        Returns:
            True if consistent, False otherwise
        """
        for constraint in self.csp.get_constraints_for_variable(var):
            # Check only if all variables in constraint are assigned
            all_assigned = all(v in assignment for v in constraint.get_scope())
            
            if all_assigned:
                if not constraint.is_satisfied(assignment):
                    return False
        
        return True
    
    def _inference(self, var: VariableName, value: DomainValue,
                  assignment: Dict[VariableName, DomainValue],
                  domains: Domain) -> bool:
        """
        Basic inference - no additional constraint propagation.
        
        Override in subclasses for more sophisticated inference.
        
        Args:
            var: Variable that was just assigned
            value: Value that was assigned
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            True if inference succeeded, False if inconsistency detected
        """
        # Reduce domain of assigned variable to single value
        domains.set(var, {value})
        return True
    
    def get_stats(self) -> SearchStats:
        """Get search statistics."""
        return self.stats


class ForwardCheckingSearch(BacktrackingSearch):
    """
    Backtracking search with forward checking.
    
    Maintains arc consistency for the assigned variable after each assignment.
    """
    
    def _inference(self, var: VariableName, value: DomainValue,
                  assignment: Dict[VariableName, DomainValue],
                  domains: Domain) -> bool:
        """
        Forward checking - remove inconsistent values from unassigned variables.
        
        Args:
            var: Variable that was just assigned
            value: Value that was assigned
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            True if inference succeeded, False if domain wipeout detected
        """
        # First reduce domain of assigned variable
        domains.set(var, {value})
        
        # Forward check: remove inconsistent values from neighbors
        for constraint in self.csp.get_constraints_for_variable(var):
            for other_var in constraint.get_scope():
                if other_var != var and other_var not in assignment:
                    # Remove values from other_var that are inconsistent with var=value
                    other_domain = domains.get(other_var).copy()
                    
                    for other_val in other_domain:
                        test_assignment = {var: value, other_var: other_val}
                        
                        # Add any other assigned variables in the constraint
                        for v in constraint.get_scope():
                            if v in assignment:
                                test_assignment[v] = assignment[v]
                        
                        if not constraint.is_satisfied(test_assignment):
                            domains.remove(other_var, other_val)
                            self.stats.domain_reductions += 1
                    
                    # Check for domain wipeout
                    if domains.is_empty(other_var):
                        return False
        
        return True


class MACSearch(BacktrackingSearch):
    """
    Backtracking search with Maintaining Arc Consistency (MAC).
    
    Maintains full arc consistency after each assignment.
    """
    
    def __init__(self, csp: CSP,
                 variable_ordering: VariableOrdering = VariableOrdering.MRV_DEGREE,
                 value_ordering: ValueOrdering = ValueOrdering.LCV,
                 ac_algorithm: str = 'ac3'):
        """
        Initialize MAC search.
        
        Args:
            csp: The CSP to solve
            variable_ordering: Heuristic for selecting next variable
            value_ordering: Heuristic for ordering domain values
            ac_algorithm: Arc consistency algorithm to use ('ac3' or 'ac4')
        """
        super().__init__(csp, variable_ordering, value_ordering)
        self.mac = MAC(csp, ac_algorithm)
    
    def solve(self, assignment: Optional[Dict[VariableName, DomainValue]] = None,
              domains: Optional[Domain] = None) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Find a solution using MAC.
        
        Args:
            assignment: Initial partial assignment
            domains: Initial domains
            
        Returns:
            Complete assignment if solution found, None otherwise
        """
        if domains is None:
            domains = self.csp.domains.copy()
        
        # Initial arc consistency
        if not achieve_arc_consistency(self.csp, domains=domains):
            return None  # Problem is inconsistent
        
        return super().solve(assignment, domains)
    
    def _inference(self, var: VariableName, value: DomainValue,
                  assignment: Dict[VariableName, DomainValue],
                  domains: Domain) -> bool:
        """
        MAC inference - maintain arc consistency after assignment.
        
        Args:
            var: Variable that was just assigned
            value: Value that was assigned
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            True if arc consistency maintained, False if inconsistency detected
        """
        return self.mac.maintain_arc_consistency(var, value, assignment, domains)


class MinConflicts:
    """
    Min-conflicts local search algorithm.
    
    Starts with a complete (possibly inconsistent) assignment and
    iteratively improves it by minimizing conflicts.
    """
    
    def __init__(self, csp: CSP, max_steps: int = 10000):
        """
        Initialize min-conflicts search.
        
        Args:
            csp: The CSP to solve
            max_steps: Maximum number of improvement steps
        """
        self.csp = csp
        self.max_steps = max_steps
        self.stats = SearchStats()
    
    def solve(self, initial_assignment: Optional[Dict[VariableName, DomainValue]] = None) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Find a solution using min-conflicts local search.
        
        Args:
            initial_assignment: Initial complete assignment (random if not provided)
            
        Returns:
            Solution if found within max_steps, None otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        # Generate initial complete assignment
        if initial_assignment is None:
            assignment = self._random_assignment()
        else:
            assignment = initial_assignment.copy()
            # Complete any missing variables
            for var in self.csp.variables:
                if var not in assignment:
                    domain = list(self.csp.domains.get(var))
                    if domain:
                        assignment[var] = random.choice(domain)
        
        # Main loop
        for step in range(self.max_steps):
            self.stats.nodes_explored += 1
            
            # Check if current assignment is a solution
            conflicts = self._get_conflicted_variables(assignment)
            
            if not conflicts:
                self.stats.solutions_found += 1
                self.stats.time_elapsed = time.time() - start_time
                return assignment
            
            # Select a random conflicted variable
            var = random.choice(conflicts)
            
            # Find value that minimizes conflicts
            min_conflicts_value = self._min_conflicts_value(var, assignment)
            
            if min_conflicts_value is not None:
                assignment[var] = min_conflicts_value
        
        self.stats.time_elapsed = time.time() - start_time
        return None  # No solution found within max_steps
    
    def _random_assignment(self) -> Dict[VariableName, DomainValue]:
        """Generate a random complete assignment."""
        assignment = {}
        
        for var in self.csp.variables:
            domain = list(self.csp.domains.get(var))
            if domain:
                assignment[var] = random.choice(domain)
        
        return assignment
    
    def _get_conflicted_variables(self, assignment: Dict[VariableName, DomainValue]) -> List[VariableName]:
        """
        Get list of variables involved in constraint violations.
        
        Args:
            assignment: Current assignment
            
        Returns:
            List of conflicted variable names
        """
        conflicted = set()
        
        for constraint in self.csp.constraints:
            self.stats.constraint_checks += 1
            
            # Check if all variables in constraint are assigned
            if all(var in assignment for var in constraint.get_scope()):
                if not constraint.is_satisfied(assignment):
                    # Add all variables in this constraint to conflicted set
                    for var in constraint.get_scope():
                        conflicted.add(var)
        
        return list(conflicted)
    
    def _min_conflicts_value(self, var: VariableName,
                            assignment: Dict[VariableName, DomainValue]) -> Optional[DomainValue]:
        """
        Find the value that minimizes conflicts for the given variable.
        
        Args:
            var: Variable to find value for
            assignment: Current assignment
            
        Returns:
            Value that minimizes conflicts, or None if no improvement possible
        """
        min_conflicts = float('inf')
        best_values = []
        
        for value in self.csp.domains.get(var):
            # Count conflicts if we assign var=value
            test_assignment = assignment.copy()
            test_assignment[var] = value
            
            conflicts = 0
            for constraint in self.csp.get_constraints_for_variable(var):
                self.stats.constraint_checks += 1
                
                # Check if all variables in constraint are assigned
                if all(v in test_assignment for v in constraint.get_scope()):
                    if not constraint.is_satisfied(test_assignment):
                        conflicts += 1
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_values = [value]
            elif conflicts == min_conflicts:
                best_values.append(value)
        
        if best_values:
            # Return random choice among best values
            return random.choice(best_values)
        
        return None
    
    def get_stats(self) -> SearchStats:
        """Get search statistics."""
        return self.stats


class HybridSearch:
    """
    Hybrid search combining systematic and local search.
    
    Uses backtracking with constraint propagation, but switches to
    local search when stuck.
    """
    
    def __init__(self, csp: CSP,
                 systematic_timeout: float = 10.0,
                 local_search_steps: int = 1000):
        """
        Initialize hybrid search.
        
        Args:
            csp: The CSP to solve
            systematic_timeout: Time limit for systematic search (seconds)
            local_search_steps: Max steps for local search phase
        """
        self.csp = csp
        self.systematic_timeout = systematic_timeout
        self.local_search_steps = local_search_steps
        self.stats = SearchStats()
    
    def solve(self) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Find a solution using hybrid approach.
        
        Returns:
            Solution if found, None otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        # Phase 1: Try systematic search with MAC
        mac_search = MACSearch(self.csp)
        
        # Set up timeout mechanism
        def timeout_checker():
            return time.time() - start_time < self.systematic_timeout
        
        # Modified backtrack with timeout
        assignment = self._backtrack_with_timeout(mac_search, {}, 
                                                 self.csp.domains.copy(),
                                                 timeout_checker)
        
        if assignment is not None:
            self.stats.solutions_found += 1
            self.stats.time_elapsed = time.time() - start_time
            return assignment
        
        # Phase 2: Switch to local search
        # Use partial assignment from systematic search as starting point
        partial = self._get_best_partial_assignment()
        
        min_conflicts = MinConflicts(self.csp, self.local_search_steps)
        solution = min_conflicts.solve(partial)
        
        if solution is not None:
            self.stats.solutions_found += 1
        
        self.stats.time_elapsed = time.time() - start_time
        
        # Combine stats
        self.stats.nodes_explored += min_conflicts.stats.nodes_explored
        self.stats.constraint_checks += min_conflicts.stats.constraint_checks
        
        return solution
    
    def _backtrack_with_timeout(self, search: MACSearch,
                               assignment: Dict[VariableName, DomainValue],
                               domains: Domain,
                               timeout_checker: Callable[[], bool]) -> Optional[Dict[VariableName, DomainValue]]:
        """
        Backtracking with timeout check.
        
        Args:
            search: Search algorithm instance
            assignment: Current assignment
            domains: Current domains
            timeout_checker: Function that returns False when timeout reached
            
        Returns:
            Solution if found before timeout, None otherwise
        """
        if not timeout_checker():
            return None
        
        self.stats.nodes_explored += 1
        
        if self.csp.is_complete(assignment):
            if self.csp.is_consistent(assignment):
                return assignment.copy()
            return None
        
        var = search._select_unassigned_variable(assignment, domains)
        if var is None:
            return None
        
        for value in search._order_domain_values(var, assignment, domains):
            if not timeout_checker():
                return None
            
            assignment[var] = value
            
            if search._is_consistent_assignment(var, value, assignment):
                checkpoint = domains.save_state()
                
                if search._inference(var, value, assignment, domains):
                    result = self._backtrack_with_timeout(search, assignment, 
                                                         domains, timeout_checker)
                    
                    if result is not None:
                        return result
                
                domains.restore_state(checkpoint)
            
            del assignment[var]
            self.stats.backtracks += 1
        
        return None
    
    def _get_best_partial_assignment(self) -> Dict[VariableName, DomainValue]:
        """
        Get the best partial assignment found so far.
        
        This is a heuristic - we'll use domain reduction information.
        
        Returns:
            Partial assignment based on singleton domains
        """
        assignment = {}
        
        for var in self.csp.variables:
            domain = self.csp.domains.get(var)
            if len(domain) == 1:
                # Variable has been determined
                assignment[var] = next(iter(domain))
            elif len(domain) > 0:
                # Use most constrained value heuristic
                # Choose value that appears in most constraints
                value_counts = defaultdict(int)
                
                for constraint in self.csp.get_constraints_for_variable(var):
                    for val in domain:
                        test_assignment = {var: val}
                        # Count how many other values this is compatible with
                        for other_var in constraint.get_scope():
                            if other_var != var:
                                for other_val in self.csp.domains.get(other_var):
                                    test_assignment[other_var] = other_val
                                    if constraint.is_satisfied(test_assignment):
                                        value_counts[val] += 1
                                    del test_assignment[other_var]
                
                if value_counts:
                    # Choose value with most compatibility
                    best_val = max(value_counts, key=value_counts.get)
                    assignment[var] = best_val
        
        return assignment
    
    def get_stats(self) -> SearchStats:
        """Get search statistics."""
        return self.stats


def solve_csp(csp: CSP, 
              algorithm: str = 'mac',
              **kwargs) -> Optional[Dict[VariableName, DomainValue]]:
    """
    Solve a CSP using the specified algorithm.
    
    Args:
        csp: The CSP to solve
        algorithm: Algorithm to use ('backtrack', 'forward', 'mac', 'min_conflicts', 'hybrid')
        **kwargs: Additional arguments for the specific algorithm
        
    Returns:
        Solution if found, None otherwise
    """
    if algorithm == 'backtrack':
        search = BacktrackingSearch(csp, **kwargs)
        return search.solve()
    
    elif algorithm == 'forward':
        search = ForwardCheckingSearch(csp, **kwargs)
        return search.solve()
    
    elif algorithm == 'mac':
        search = MACSearch(csp, **kwargs)
        return search.solve()
    
    elif algorithm == 'min_conflicts':
        search = MinConflicts(csp, **kwargs)
        return search.solve()
    
    elif algorithm == 'hybrid':
        search = HybridSearch(csp, **kwargs)
        return search.solve()
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def find_all_solutions(csp: CSP,
                      algorithm: str = 'backtrack',
                      max_solutions: int = 100) -> List[Dict[VariableName, DomainValue]]:
    """
    Find all solutions to a CSP (up to max_solutions).
    
    Args:
        csp: The CSP to solve
        algorithm: Algorithm to use
        max_solutions: Maximum number of solutions to find
        
    Returns:
        List of all solutions found
    """
    solutions = []
    
    if algorithm == 'min_conflicts':
        # Min-conflicts can't systematically find all solutions
        # Just run it multiple times with different random starts
        search = MinConflicts(csp)
        
        for _ in range(max_solutions):
            solution = search.solve()
            if solution and solution not in solutions:
                solutions.append(solution)
            
            if len(solutions) >= max_solutions:
                break
    else:
        # Use systematic search with solution collection
        if algorithm == 'forward':
            search = ForwardCheckingSearch(csp)
        elif algorithm == 'mac':
            search = MACSearch(csp)
        else:
            search = BacktrackingSearch(csp)
        
        # Modified backtrack that collects all solutions
        def collect_solutions(assignment, domains):
            nonlocal solutions
            
            if len(solutions) >= max_solutions:
                return
            
            if csp.is_complete(assignment):
                if csp.is_consistent(assignment):
                    solutions.append(assignment.copy())
                return
            
            var = search._select_unassigned_variable(assignment, domains)
            if var is None:
                return
            
            for value in search._order_domain_values(var, assignment, domains):
                assignment[var] = value
                
                if search._is_consistent_assignment(var, value, assignment):
                    checkpoint = domains.save_state()
                    
                    if search._inference(var, value, assignment, domains):
                        collect_solutions(assignment, domains)
                    
                    domains.restore_state(checkpoint)
                
                del assignment[var]
        
        collect_solutions({}, csp.domains.copy())
    
    return solutions