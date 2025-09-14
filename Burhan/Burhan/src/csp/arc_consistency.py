"""
Arc Consistency algorithms for CSP solving in ARC Prize 2025.

This module implements various arc consistency algorithms for efficient
constraint propagation in grid-based puzzles.
"""

from typing import Dict, List, Set, Tuple, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import time

from .core import (
    CSP, Variable, Domain, Constraint, BinaryConstraint,
    VariableName, DomainValue
)


@dataclass
class ArcConsistencyStats:
    """Statistics for arc consistency algorithms."""
    arcs_examined: int = 0
    domain_reductions: int = 0
    time_elapsed: float = 0.0
    iterations: int = 0
    
    def reset(self):
        """Reset all statistics."""
        self.arcs_examined = 0
        self.domain_reductions = 0
        self.time_elapsed = 0.0
        self.iterations = 0


class AC3:
    """
    AC-3 (Arc Consistency Algorithm #3) implementation.
    
    This is the most popular arc consistency algorithm, with O(ed³) complexity
    where e is the number of edges and d is the maximum domain size.
    """
    
    def __init__(self, csp: CSP):
        """
        Initialize AC-3 algorithm.
        
        Args:
            csp: The CSP to apply arc consistency to
        """
        self.csp = csp
        self.stats = ArcConsistencyStats()
    
    def run(self, domains: Optional[Domain] = None) -> bool:
        """
        Run AC-3 algorithm to achieve arc consistency.
        
        Args:
            domains: Domain manager (uses CSP's domains if not provided)
            
        Returns:
            False if inconsistency detected, True otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        if domains is None:
            domains = self.csp.domains
        
        # Initialize queue with all arcs
        queue = self._initialize_queue()
        
        while queue:
            self.stats.iterations += 1
            xi, xj = queue.popleft()
            self.stats.arcs_examined += 1
            
            if self._revise(xi, xj, domains):
                if domains.is_empty(xi):
                    # Domain wipeout - inconsistency detected
                    self.stats.time_elapsed = time.time() - start_time
                    return False
                
                # Add all arcs (xk, xi) for each neighbor xk of xi
                for constraint in self.csp.get_constraints_for_variable(xi):
                    if isinstance(constraint, BinaryConstraint):
                        for var in constraint.variables:
                            if var != xi and var != xj:
                                queue.append((var, xi))
        
        self.stats.time_elapsed = time.time() - start_time
        return True
    
    def _initialize_queue(self) -> Deque[Tuple[VariableName, VariableName]]:
        """
        Initialize the queue with all arcs from binary constraints.
        
        Returns:
            Deque containing all arcs
        """
        queue = deque()
        seen = set()
        
        for constraint in self.csp.constraints:
            if isinstance(constraint, BinaryConstraint):
                var1, var2 = constraint.variables
                
                # Add both directions of the arc
                if (var1, var2) not in seen:
                    queue.append((var1, var2))
                    seen.add((var1, var2))
                
                if (var2, var1) not in seen:
                    queue.append((var2, var1))
                    seen.add((var2, var1))
        
        return queue
    
    def _revise(self, xi: VariableName, xj: VariableName, domains: Domain) -> bool:
        """
        Revise the domain of xi with respect to xj.
        
        Remove values from domain of xi that have no support in domain of xj.
        
        Args:
            xi: Variable whose domain is being revised
            xj: Variable providing support
            domains: Current domains
            
        Returns:
            True if domain of xi was changed
        """
        revised = False
        xi_domain = domains.get(xi).copy()
        xj_domain = domains.get(xj)
        
        # Get constraints between xi and xj
        constraints = self.csp.get_binary_constraints(xi, xj)
        
        for vi in xi_domain:
            has_support = False
            
            # Check if vi has support in xj's domain
            for vj in xj_domain:
                # Check all constraints between xi and xj
                all_satisfied = True
                for constraint in constraints:
                    # Determine order of variables in constraint
                    if constraint.variables[0] == xi:
                        if not constraint.relation(vi, vj):
                            all_satisfied = False
                            break
                    else:
                        if not constraint.relation(vj, vi):
                            all_satisfied = False
                            break
                
                if all_satisfied and constraints:  # Found support
                    has_support = True
                    break
            
            # If no constraints between xi and xj, everything has support
            if not constraints:
                has_support = True
            
            if not has_support:
                domains.remove(xi, vi)
                revised = True
                self.stats.domain_reductions += 1
        
        return revised
    
    def get_stats(self) -> ArcConsistencyStats:
        """Get algorithm statistics."""
        return self.stats


class AC4:
    """
    AC-4 (Arc Consistency Algorithm #4) implementation.
    
    More complex but optimal O(ed²) algorithm, better for dense constraint graphs.
    Uses support counters to avoid redundant checks.
    """
    
    def __init__(self, csp: CSP):
        """
        Initialize AC-4 algorithm.
        
        Args:
            csp: The CSP to apply arc consistency to
        """
        self.csp = csp
        self.stats = ArcConsistencyStats()
    
    def run(self, domains: Optional[Domain] = None) -> bool:
        """
        Run AC-4 algorithm to achieve arc consistency.
        
        Args:
            domains: Domain manager (uses CSP's domains if not provided)
            
        Returns:
            False if inconsistency detected, True otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        if domains is None:
            domains = self.csp.domains
        
        # Initialize support structures
        support = self._initialize_support(domains)
        counter = self._initialize_counter(domains, support)
        
        # Initialize queue with unsupported values
        queue = deque()
        
        for (xi, vi), supporting_pairs in support.items():
            if not supporting_pairs:
                queue.append((xi, vi))
                if not domains.remove(xi, vi):
                    continue  # Value already removed
                self.stats.domain_reductions += 1
        
        # Propagate removals
        while queue:
            self.stats.iterations += 1
            xi, vi = queue.popleft()
            
            # Find all (xj, vj) supported by (xi, vi)
            for constraint in self.csp.get_constraints_for_variable(xi):
                if not isinstance(constraint, BinaryConstraint):
                    continue
                
                xj = None
                for var in constraint.variables:
                    if var != xi:
                        xj = var
                        break
                
                if xj is None:
                    continue
                
                for vj in domains.get(xj).copy():
                    # Check if (xi, vi) supports (xj, vj)
                    if self._is_support(xi, vi, xj, vj, constraint):
                        counter[(xj, vj, xi)] -= 1
                        
                        if counter[(xj, vj, xi)] == 0:
                            # No more support for (xj, vj) from xi
                            if vj in domains.get(xj):
                                queue.append((xj, vj))
                                domains.remove(xj, vj)
                                self.stats.domain_reductions += 1
                                
                                if domains.is_empty(xj):
                                    self.stats.time_elapsed = time.time() - start_time
                                    return False
        
        self.stats.time_elapsed = time.time() - start_time
        return True
    
    def _initialize_support(self, domains: Domain) -> Dict[Tuple[VariableName, DomainValue], 
                                                           Set[Tuple[VariableName, DomainValue]]]:
        """
        Initialize support sets for each (variable, value) pair.
        
        Returns:
            Dictionary mapping (var, val) to set of supporting (var, val) pairs
        """
        support = defaultdict(set)
        
        for constraint in self.csp.constraints:
            if not isinstance(constraint, BinaryConstraint):
                continue
            
            xi, xj = constraint.variables
            
            for vi in domains.get(xi):
                for vj in domains.get(xj):
                    if constraint.relation(vi, vj):
                        support[(xi, vi)].add((xj, vj))
                        support[(xj, vj)].add((xi, vi))
        
        return support
    
    def _initialize_counter(self, domains: Domain, 
                           support: Dict[Tuple[VariableName, DomainValue], 
                                       Set[Tuple[VariableName, DomainValue]]]) -> Dict:
        """
        Initialize counters for support counts.
        
        Returns:
            Dictionary mapping (xi, vi, xj) to count of supports from xj
        """
        counter = defaultdict(int)
        
        for (xi, vi), supporting_pairs in support.items():
            support_by_var = defaultdict(int)
            for xj, vj in supporting_pairs:
                support_by_var[xj] += 1
            
            for xj, count in support_by_var.items():
                counter[(xi, vi, xj)] = count
        
        # Ensure all (xi, vi, xj) combinations have an entry
        for constraint in self.csp.constraints:
            if not isinstance(constraint, BinaryConstraint):
                continue
            
            xi, xj = constraint.variables
            
            for vi in domains.get(xi):
                if (xi, vi, xj) not in counter:
                    counter[(xi, vi, xj)] = 0
            
            for vj in domains.get(xj):
                if (xj, vj, xi) not in counter:
                    counter[(xj, vj, xi)] = 0
        
        return counter
    
    def _is_support(self, xi: VariableName, vi: DomainValue,
                   xj: VariableName, vj: DomainValue,
                   constraint: BinaryConstraint) -> bool:
        """Check if (xi, vi) supports (xj, vj) under the constraint."""
        if constraint.variables[0] == xi:
            return constraint.relation(vi, vj)
        else:
            return constraint.relation(vj, vi)
    
    def get_stats(self) -> ArcConsistencyStats:
        """Get algorithm statistics."""
        return self.stats


class MAC:
    """
    Maintaining Arc Consistency (MAC) algorithm.
    
    Combines search with arc consistency maintenance for efficient solving.
    """
    
    def __init__(self, csp: CSP, ac_algorithm: str = 'ac3'):
        """
        Initialize MAC algorithm.
        
        Args:
            csp: The CSP to solve
            ac_algorithm: Which AC algorithm to use ('ac3' or 'ac4')
        """
        self.csp = csp
        self.ac_algorithm = ac_algorithm
        self.stats = ArcConsistencyStats()
    
    def maintain_arc_consistency(self, var: VariableName, value: DomainValue,
                                assignment: Dict[VariableName, DomainValue],
                                domains: Domain) -> bool:
        """
        Maintain arc consistency after assigning var=value.
        
        Args:
            var: Variable being assigned
            value: Value being assigned
            assignment: Current assignment
            domains: Current domains
            
        Returns:
            False if inconsistency detected, True otherwise
        """
        # Create AC algorithm instance
        if self.ac_algorithm == 'ac4':
            ac = AC4(self.csp)
        else:
            ac = AC3(self.csp)
        
        # First, reduce domain of var to just the assigned value
        domains.set(var, {value})
        
        # Run arc consistency from this variable
        queue = deque()
        
        # Add all arcs (neighbor, var) to queue
        for constraint in self.csp.get_constraints_for_variable(var):
            if isinstance(constraint, BinaryConstraint):
                for other_var in constraint.variables:
                    if other_var != var and other_var not in assignment:
                        queue.append((other_var, var))
        
        # Run limited AC-3 starting from affected arcs
        return self._ac3_from_queue(queue, domains)
    
    def _ac3_from_queue(self, queue: Deque[Tuple[VariableName, VariableName]],
                       domains: Domain) -> bool:
        """
        Run AC-3 starting from a specific queue of arcs.
        
        Args:
            queue: Initial queue of arcs to process
            domains: Current domains
            
        Returns:
            False if inconsistency detected, True otherwise
        """
        processed = set()
        
        while queue:
            self.stats.iterations += 1
            xi, xj = queue.popleft()
            
            # Skip if already processed
            if (xi, xj) in processed:
                continue
            processed.add((xi, xj))
            
            self.stats.arcs_examined += 1
            
            if self._revise(xi, xj, domains):
                if domains.is_empty(xi):
                    return False
                
                # Add all arcs (xk, xi) for neighbors xk != xj
                for constraint in self.csp.get_constraints_for_variable(xi):
                    if isinstance(constraint, BinaryConstraint):
                        for var in constraint.variables:
                            if var != xi and var != xj:
                                if (var, xi) not in processed:
                                    queue.append((var, xi))
        
        return True
    
    def _revise(self, xi: VariableName, xj: VariableName, domains: Domain) -> bool:
        """
        Revise domain of xi with respect to xj (same as AC-3).
        
        Args:
            xi: Variable whose domain is being revised
            xj: Variable providing support
            domains: Current domains
            
        Returns:
            True if domain was changed
        """
        revised = False
        xi_domain = domains.get(xi).copy()
        xj_domain = domains.get(xj)
        
        constraints = self.csp.get_binary_constraints(xi, xj)
        
        for vi in xi_domain:
            has_support = False
            
            for vj in xj_domain:
                all_satisfied = True
                for constraint in constraints:
                    if constraint.variables[0] == xi:
                        if not constraint.relation(vi, vj):
                            all_satisfied = False
                            break
                    else:
                        if not constraint.relation(vj, vi):
                            all_satisfied = False
                            break
                
                if all_satisfied and constraints:
                    has_support = True
                    break
            
            if not constraints:
                has_support = True
            
            if not has_support:
                domains.remove(xi, vi)
                revised = True
                self.stats.domain_reductions += 1
        
        return revised
    
    def get_stats(self) -> ArcConsistencyStats:
        """Get algorithm statistics."""
        return self.stats


def achieve_arc_consistency(csp: CSP, algorithm: str = 'ac3',
                           domains: Optional[Domain] = None) -> bool:
    """
    Achieve arc consistency using specified algorithm.
    
    Args:
        csp: The CSP to make arc consistent
        algorithm: Algorithm to use ('ac3' or 'ac4')
        domains: Domain manager (uses CSP's domains if not provided)
        
    Returns:
        False if inconsistency detected, True otherwise
    """
    if algorithm == 'ac4':
        ac = AC4(csp)
    else:
        ac = AC3(csp)
    
    return ac.run(domains)


def is_arc_consistent(csp: CSP, domains: Optional[Domain] = None) -> bool:
    """
    Check if a CSP is already arc consistent.
    
    Args:
        csp: The CSP to check
        domains: Domain manager (uses CSP's domains if not provided)
        
    Returns:
        True if arc consistent, False otherwise
    """
    if domains is None:
        domains = csp.domains
    
    for constraint in csp.constraints:
        if not isinstance(constraint, BinaryConstraint):
            continue
        
        xi, xj = constraint.variables
        
        # Check if every value in xi has support in xj
        for vi in domains.get(xi):
            has_support = False
            for vj in domains.get(xj):
                if constraint.relation(vi, vj):
                    has_support = True
                    break
            
            if not has_support:
                return False
        
        # Check if every value in xj has support in xi
        for vj in domains.get(xj):
            has_support = False
            for vi in domains.get(xi):
                if constraint.relation(vi, vj):
                    has_support = True
                    break
            
            if not has_support:
                return False
    
    return True


class PathConsistency:
    """
    Path Consistency algorithm for stronger consistency than arc consistency.
    
    Ensures that for any consistent assignment to two variables,
    there exists a consistent assignment to any third variable.
    """
    
    def __init__(self, csp: CSP):
        """Initialize path consistency algorithm."""
        self.csp = csp
        self.stats = ArcConsistencyStats()
    
    def run(self, domains: Optional[Domain] = None) -> bool:
        """
        Achieve path consistency.
        
        Note: This is computationally expensive and typically not used
        for large problems like 30x30 grids.
        
        Args:
            domains: Domain manager
            
        Returns:
            False if inconsistency detected, True otherwise
        """
        start_time = time.time()
        self.stats.reset()
        
        if domains is None:
            domains = self.csp.domains
        
        variables = list(self.csp.variables.keys())
        n = len(variables)
        
        # Initialize allowed tuples for each pair of variables
        allowed = {}
        for i in range(n):
            for j in range(i + 1, n):
                xi, xj = variables[i], variables[j]
                allowed[(xi, xj)] = set()
                
                # Find all consistent value pairs
                for vi in domains.get(xi):
                    for vj in domains.get(xj):
                        # Check if this pair satisfies all binary constraints
                        assignment = {xi: vi, xj: vj}
                        if self._check_binary_constraints(xi, xj, assignment):
                            allowed[(xi, xj)].add((vi, vj))
        
        # Path consistency main loop
        changed = True
        while changed:
            changed = False
            self.stats.iterations += 1
            
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        
                        xi, xj, xk = variables[i], variables[j], variables[k]
                        
                        # Revise allowed[(xi, xj)] based on xk
                        old_size = len(allowed.get((xi, xj), set()))
                        self._revise_path(xi, xj, xk, allowed)
                        
                        if len(allowed.get((xi, xj), set())) < old_size:
                            changed = True
                            self.stats.domain_reductions += old_size - len(allowed.get((xi, xj), set()))
                            
                            if not allowed.get((xi, xj), set()):
                                # No consistent pairs left
                                self.stats.time_elapsed = time.time() - start_time
                                return False
        
        # Update domains based on path consistency
        for i in range(n):
            xi = variables[i]
            new_domain = set()
            
            for vi in domains.get(xi):
                has_support = True
                
                for j in range(n):
                    if i == j:
                        continue
                    
                    xj = variables[j]
                    pair_key = (xi, xj) if i < j else (xj, xi)
                    
                    # Check if vi participates in any allowed tuple with xj
                    found = False
                    for tuple_val in allowed.get(pair_key, set()):
                        if i < j and tuple_val[0] == vi:
                            found = True
                            break
                        elif i > j and tuple_val[1] == vi:
                            found = True
                            break
                    
                    if not found:
                        has_support = False
                        break
                
                if has_support:
                    new_domain.add(vi)
            
            domains.set(xi, new_domain)
        
        self.stats.time_elapsed = time.time() - start_time
        return True
    
    def _check_binary_constraints(self, xi: VariableName, xj: VariableName,
                                 assignment: Dict[VariableName, DomainValue]) -> bool:
        """Check if assignment satisfies all binary constraints between xi and xj."""
        constraints = self.csp.get_binary_constraints(xi, xj)
        
        for constraint in constraints:
            if not constraint.is_satisfied(assignment):
                return False
        
        return True
    
    def _revise_path(self, xi: VariableName, xj: VariableName, xk: VariableName,
                    allowed: Dict) -> None:
        """Revise allowed tuples for (xi, xj) based on xk."""
        pair_key = (xi, xj)
        ik_key = (xi, xk) if xi < xk else (xk, xi)
        jk_key = (xj, xk) if xj < xk else (xk, xj)
        
        new_allowed = set()
        
        for vi, vj in allowed.get(pair_key, set()):
            # Check if there exists vk such that (vi, vk) and (vj, vk) are allowed
            found_support = False
            
            for vk in self.csp.domains.get(xk):
                # Check (xi, xk) compatibility
                if xi < xk:
                    ik_compatible = (vi, vk) in allowed.get(ik_key, set())
                else:
                    ik_compatible = (vk, vi) in allowed.get(ik_key, set())
                
                # Check (xj, xk) compatibility
                if xj < xk:
                    jk_compatible = (vj, vk) in allowed.get(jk_key, set())
                else:
                    jk_compatible = (vk, vj) in allowed.get(jk_key, set())
                
                if ik_compatible and jk_compatible:
                    found_support = True
                    break
            
            if found_support:
                new_allowed.add((vi, vj))
        
        allowed[pair_key] = new_allowed