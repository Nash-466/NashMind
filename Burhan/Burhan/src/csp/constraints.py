"""
ARC-specific constraints for CSP solving in ARC Prize 2025.

This module provides specialized constraints for grid-based puzzles
and pattern recognition tasks.
"""

from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations, permutations

from .core import (
    Constraint, BinaryConstraint, NAryConstraint,
    Variable, Domain, VariableName, DomainValue
)


class FunctionConstraint(NAryConstraint):
    """
    General-purpose constraint defined by a custom function.
    
    Useful for complex constraints that don't fit standard patterns.
    """
    
    def __init__(self, variables: List[VariableName], 
                 function: Callable[[Dict[VariableName, DomainValue]], bool],
                 name: Optional[str] = None):
        """
        Initialize function constraint.
        
        Args:
            variables: Variables involved in the constraint
            function: Function that checks if assignment is valid
            name: Optional name for the constraint
        """
        super().__init__(variables, function, name or "FunctionConstraint")


class AllDifferentConstraint(NAryConstraint):
    """
    Constraint requiring all variables to have different values.
    
    Useful for puzzles where each cell in a region must have a unique color.
    """
    
    def __init__(self, variables: List[VariableName], name: Optional[str] = None):
        """
        Initialize all-different constraint.
        
        Args:
            variables: List of variables that must all be different
            name: Optional name for the constraint
        """
        def all_different_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            values = list(assignment.values())
            return len(values) == len(set(values))
        
        super().__init__(variables, all_different_predicate, 
                        name or f"AllDifferent({','.join(variables[:3])}...)")
    
    def prune_domains(self, domains: Domain) -> bool:
        """
        Specialized pruning for all-different constraint.
        
        Uses the pigeonhole principle and singleton propagation.
        """
        changed = False
        
        # Collect singleton values (variables with domain size 1)
        singleton_values = set()
        for var in self.variables:
            if domains.is_singleton(var):
                singleton_values.add(domains.get_singleton_value(var))
        
        # Remove singleton values from other variables' domains
        for var in self.variables:
            if not domains.is_singleton(var):
                for value in singleton_values:
                    if domains.remove(var, value):
                        changed = True
        
        # Check for pigeonhole principle violations
        remaining_vars = [v for v in self.variables if not domains.is_singleton(v)]
        if len(remaining_vars) > 0:
            # Collect all possible values
            all_values = set()
            for var in remaining_vars:
                all_values.update(domains.get(var))
            
            # If there are more variables than values, no solution exists
            if len(remaining_vars) > len(all_values):
                # Create domain wipeout
                for var in remaining_vars:
                    domains.set(var, set())
                return True
        
        # Hidden singles: if a value appears in only one domain, assign it
        value_to_vars = defaultdict(list)
        for var in self.variables:
            if not domains.is_singleton(var):
                for value in domains.get(var):
                    value_to_vars[value].append(var)
        
        for value, vars_with_value in value_to_vars.items():
            if len(vars_with_value) == 1:
                var = vars_with_value[0]
                if len(domains.get(var)) > 1:
                    domains.set(var, {value})
                    changed = True
        
        return changed


class SamePatternConstraint(NAryConstraint):
    """
    Constraint requiring variables to follow the same pattern as a reference.
    
    Useful for pattern matching and repetition detection in ARC puzzles.
    """
    
    def __init__(self, variables: List[VariableName], 
                 reference_pattern: List[Any],
                 allow_rotation: bool = False,
                 allow_reflection: bool = False,
                 name: Optional[str] = None):
        """
        Initialize same-pattern constraint.
        
        Args:
            variables: Variables that should match the pattern
            reference_pattern: The pattern to match
            allow_rotation: Whether to allow rotated versions
            allow_reflection: Whether to allow reflected versions
            name: Optional name for the constraint
        """
        self.reference_pattern = reference_pattern
        self.allow_rotation = allow_rotation
        self.allow_reflection = allow_reflection
        
        # Generate all valid transformations of the pattern
        self.valid_patterns = self._generate_pattern_variants(reference_pattern)
        
        def pattern_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            if len(assignment) != len(variables):
                return True  # Not all assigned yet
            
            # Extract values in order
            values = [assignment[var] for var in variables]
            
            # Check if values match any valid pattern
            return tuple(values) in self.valid_patterns
        
        super().__init__(variables, pattern_predicate,
                        name or f"SamePattern({len(variables)} vars)")
    
    def _generate_pattern_variants(self, pattern: List[Any]) -> Set[Tuple[Any, ...]]:
        """Generate all valid transformations of the pattern."""
        variants = {tuple(pattern)}
        
        if self.allow_rotation:
            # For 1D patterns, rotation means reversal
            variants.add(tuple(reversed(pattern)))
        
        if self.allow_reflection:
            # For 1D patterns, reflection is same as reversal
            variants.add(tuple(reversed(pattern)))
        
        return variants


class TransformationConstraint(NAryConstraint):
    """
    Constraint for grid transformations (rotation, reflection, scaling).
    
    Ensures that one grid region is a transformation of another.
    """
    
    def __init__(self, source_vars: List[VariableName],
                 target_vars: List[VariableName],
                 transformation_type: str,
                 grid_shape: Tuple[int, int],
                 name: Optional[str] = None):
        """
        Initialize transformation constraint.
        
        Args:
            source_vars: Variables in the source region
            target_vars: Variables in the target region
            transformation_type: Type of transformation ('rotate90', 'rotate180', 
                                'rotate270', 'fliph', 'flipv', 'transpose')
            grid_shape: Shape of the grid (rows, cols)
            name: Optional name for the constraint
        """
        self.source_vars = source_vars
        self.target_vars = target_vars
        self.transformation_type = transformation_type
        self.grid_shape = grid_shape
        
        # Create mapping from source to target positions
        self.position_mapping = self._create_position_mapping()
        
        def transformation_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            # Check if source and target match according to transformation
            for src_var, tgt_var in self.position_mapping.items():
                if src_var in assignment and tgt_var in assignment:
                    if assignment[src_var] != assignment[tgt_var]:
                        return False
            return True
        
        all_vars = list(set(source_vars + target_vars))
        super().__init__(all_vars, transformation_predicate,
                        name or f"Transform_{transformation_type}")
    
    def _create_position_mapping(self) -> Dict[VariableName, VariableName]:
        """Create mapping from source to target variables based on transformation."""
        mapping = {}
        rows, cols = self.grid_shape
        
        # Parse variable positions from names (assuming format "cell_row_col")
        def get_position(var_name):
            parts = var_name.split('_')
            if len(parts) >= 3:
                return int(parts[-2]), int(parts[-1])
            return None
        
        # Create position arrays
        source_positions = {var: get_position(var) for var in self.source_vars}
        target_positions = {get_position(var): var for var in self.target_vars}
        
        for src_var, src_pos in source_positions.items():
            if src_pos is None:
                continue
            
            row, col = src_pos
            
            # Apply transformation
            if self.transformation_type == 'rotate90':
                new_row, new_col = col, rows - 1 - row
            elif self.transformation_type == 'rotate180':
                new_row, new_col = rows - 1 - row, cols - 1 - col
            elif self.transformation_type == 'rotate270':
                new_row, new_col = cols - 1 - col, row
            elif self.transformation_type == 'fliph':
                new_row, new_col = row, cols - 1 - col
            elif self.transformation_type == 'flipv':
                new_row, new_col = rows - 1 - row, col
            elif self.transformation_type == 'transpose':
                new_row, new_col = col, row
            else:
                new_row, new_col = row, col
            
            # Find corresponding target variable
            if (new_row, new_col) in target_positions:
                mapping[src_var] = target_positions[(new_row, new_col)]
        
        return mapping
    
    def prune_domains(self, domains: Domain) -> bool:
        """Prune domains based on transformation mapping."""
        changed = False
        
        for src_var, tgt_var in self.position_mapping.items():
            src_domain = domains.get(src_var)
            tgt_domain = domains.get(tgt_var)
            
            # Intersect domains
            common = src_domain.intersection(tgt_domain)
            
            if len(common) < len(src_domain):
                domains.set(src_var, common)
                changed = True
            
            if len(common) < len(tgt_domain):
                domains.set(tgt_var, common)
                changed = True
        
        return changed


class ColorMappingConstraint(NAryConstraint):
    """
    Constraint for color mapping relationships.
    
    Ensures that colors are mapped consistently (e.g., all 1s become 2s).
    """
    
    def __init__(self, source_vars: List[VariableName],
                 target_vars: List[VariableName],
                 mapping: Optional[Dict[int, int]] = None,
                 allow_permutation: bool = False,
                 name: Optional[str] = None):
        """
        Initialize color mapping constraint.
        
        Args:
            source_vars: Variables in the source region
            target_vars: Variables in the target region  
            mapping: Fixed color mapping (None for any consistent mapping)
            allow_permutation: Whether to allow any permutation of colors
            name: Optional name for the constraint
        """
        self.source_vars = source_vars
        self.target_vars = target_vars
        self.fixed_mapping = mapping
        self.allow_permutation = allow_permutation
        
        def mapping_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            # Build mapping from assigned values
            inferred_mapping = {}
            
            for src_var, tgt_var in zip(source_vars, target_vars):
                if src_var in assignment and tgt_var in assignment:
                    src_val = assignment[src_var]
                    tgt_val = assignment[tgt_var]
                    
                    if self.fixed_mapping:
                        # Check against fixed mapping
                        if src_val in self.fixed_mapping:
                            if self.fixed_mapping[src_val] != tgt_val:
                                return False
                    else:
                        # Infer mapping
                        if src_val in inferred_mapping:
                            if inferred_mapping[src_val] != tgt_val:
                                return False
                        else:
                            inferred_mapping[src_val] = tgt_val
            
            # Check if mapping is valid (injective if required)
            if not self.allow_permutation:
                # Mapping should be one-to-one
                if len(set(inferred_mapping.values())) != len(inferred_mapping):
                    return False
            
            return True
        
        all_vars = list(set(source_vars + target_vars))
        super().__init__(all_vars, mapping_predicate,
                        name or "ColorMapping")


class SymmetryConstraint(NAryConstraint):
    """
    Constraint for enforcing symmetry in grids.
    
    Ensures that a grid region has specified symmetry properties.
    """
    
    def __init__(self, variables: List[VariableName],
                 grid_shape: Tuple[int, int],
                 symmetry_type: str,
                 name: Optional[str] = None):
        """
        Initialize symmetry constraint.
        
        Args:
            variables: Variables in the grid region
            grid_shape: Shape of the grid (rows, cols)
            symmetry_type: Type of symmetry ('horizontal', 'vertical', 
                          'diagonal', 'anti-diagonal', 'rotational')
            name: Optional name for the constraint
        """
        self.grid_shape = grid_shape
        self.symmetry_type = symmetry_type
        
        # Create symmetry pairs
        self.symmetry_pairs = self._create_symmetry_pairs(variables)
        
        def symmetry_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            for var1, var2 in self.symmetry_pairs:
                if var1 in assignment and var2 in assignment:
                    if assignment[var1] != assignment[var2]:
                        return False
            return True
        
        super().__init__(variables, symmetry_predicate,
                        name or f"Symmetry_{symmetry_type}")
    
    def _create_symmetry_pairs(self, variables: List[VariableName]) -> List[Tuple[VariableName, VariableName]]:
        """Create pairs of variables that must be equal for symmetry."""
        pairs = []
        rows, cols = self.grid_shape
        
        # Parse positions
        var_positions = {}
        position_vars = {}
        
        for var in variables:
            parts = var.split('_')
            if len(parts) >= 3:
                row, col = int(parts[-2]), int(parts[-1])
                var_positions[var] = (row, col)
                position_vars[(row, col)] = var
        
        # Create pairs based on symmetry type
        for var, (row, col) in var_positions.items():
            sym_row, sym_col = row, col
            
            if self.symmetry_type == 'horizontal':
                sym_col = cols - 1 - col
            elif self.symmetry_type == 'vertical':
                sym_row = rows - 1 - row
            elif self.symmetry_type == 'diagonal':
                sym_row, sym_col = col, row
            elif self.symmetry_type == 'anti-diagonal':
                sym_row = cols - 1 - col
                sym_col = rows - 1 - row
            elif self.symmetry_type == 'rotational':
                sym_row = rows - 1 - row
                sym_col = cols - 1 - col
            
            if (sym_row, sym_col) in position_vars:
                sym_var = position_vars[(sym_row, sym_col)]
                if var != sym_var and (sym_var, var) not in pairs:
                    pairs.append((var, sym_var))
        
        return pairs
    
    def prune_domains(self, domains: Domain) -> bool:
        """Prune domains based on symmetry pairs."""
        changed = False
        
        for var1, var2 in self.symmetry_pairs:
            domain1 = domains.get(var1)
            domain2 = domains.get(var2)
            
            # Intersect domains for symmetric positions
            common = domain1.intersection(domain2)
            
            if len(common) < len(domain1):
                domains.set(var1, common)
                changed = True
            
            if len(common) < len(domain2):
                domains.set(var2, common)
                changed = True
        
        return changed


class AdjacentConstraint(BinaryConstraint):
    """
    Constraint for adjacent cells in a grid.
    
    Ensures that adjacent cells satisfy a given relationship.
    """
    
    def __init__(self, var1: VariableName, var2: VariableName,
                 relation_type: str = 'different',
                 name: Optional[str] = None):
        """
        Initialize adjacent constraint.
        
        Args:
            var1: First variable
            var2: Second variable (adjacent to first)
            relation_type: Type of relation ('different', 'same', 'sum', 'product')
            name: Optional name for the constraint
        """
        if relation_type == 'different':
            relation = lambda v1, v2: v1 != v2
        elif relation_type == 'same':
            relation = lambda v1, v2: v1 == v2
        elif relation_type == 'sum':
            # For sum constraint, values should sum to a specific target
            # This would need to be parameterized
            relation = lambda v1, v2: True  # Placeholder
        elif relation_type == 'product':
            relation = lambda v1, v2: True  # Placeholder
        else:
            relation = lambda v1, v2: True
        
        super().__init__(var1, var2, relation,
                        name or f"Adjacent_{relation_type}({var1},{var2})")


class CountConstraint(NAryConstraint):
    """
    Constraint on the count of specific values.
    
    Ensures that certain values appear a specific number of times.
    """
    
    def __init__(self, variables: List[VariableName],
                 value_counts: Dict[DomainValue, int],
                 exact: bool = True,
                 name: Optional[str] = None):
        """
        Initialize count constraint.
        
        Args:
            variables: Variables to count over
            value_counts: Required counts for each value
            exact: Whether counts must be exact (vs minimum)
            name: Optional name for the constraint
        """
        self.value_counts = value_counts
        self.exact = exact
        
        def count_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            # Count assigned values
            counts = Counter(assignment.values())
            
            # Check against required counts
            for value, required_count in value_counts.items():
                actual_count = counts.get(value, 0)
                
                if exact:
                    # For exact counts, check if we haven't exceeded
                    # (final check happens when all vars assigned)
                    if actual_count > required_count:
                        return False
                    
                    # If all variables assigned, check exact match
                    if len(assignment) == len(variables):
                        if actual_count != required_count:
                            return False
                else:
                    # For minimum counts, check when all assigned
                    if len(assignment) == len(variables):
                        if actual_count < required_count:
                            return False
            
            return True
        
        super().__init__(variables, count_predicate,
                        name or f"Count({len(value_counts)} values)")
    
    def prune_domains(self, domains: Domain) -> bool:
        """Prune based on counting constraints."""
        changed = False
        
        if not self.exact:
            return super().prune_domains(domains)
        
        # Count current assignments
        assigned_counts = defaultdict(int)
        unassigned_vars = []
        
        for var in self.variables:
            if domains.is_singleton(var):
                value = domains.get_singleton_value(var)
                assigned_counts[value] += 1
            else:
                unassigned_vars.append(var)
        
        # Check if any value has reached its limit
        for value, max_count in self.value_counts.items():
            if assigned_counts[value] >= max_count:
                # Remove this value from all unassigned variables
                for var in unassigned_vars:
                    if domains.remove(var, value):
                        changed = True
        
        # Check if any value must be assigned to remaining variables
        for value, required_count in self.value_counts.items():
            remaining_needed = required_count - assigned_counts[value]
            
            if remaining_needed > 0:
                # Count how many unassigned variables can take this value
                possible_vars = [v for v in unassigned_vars if value in domains.get(v)]
                
                if len(possible_vars) == remaining_needed:
                    # These variables must all take this value
                    for var in possible_vars:
                        if len(domains.get(var)) > 1:
                            domains.set(var, {value})
                            changed = True
                elif len(possible_vars) < remaining_needed:
                    # Impossible to satisfy constraint
                    for var in unassigned_vars:
                        domains.set(var, set())
                    return True
        
        return changed


class ConnectedComponentConstraint(NAryConstraint):
    """
    Constraint for connected components in a grid.
    
    Ensures that cells of the same color form connected regions.
    """
    
    def __init__(self, variables: List[VariableName],
                 grid_shape: Tuple[int, int],
                 connectivity: str = '4',  # '4' or '8' connectivity
                 name: Optional[str] = None):
        """
        Initialize connected component constraint.
        
        Args:
            variables: Variables in the grid
            grid_shape: Shape of the grid (rows, cols)
            connectivity: Type of connectivity ('4' or '8')
            name: Optional name for the constraint
        """
        self.grid_shape = grid_shape
        self.connectivity = connectivity
        
        # Build adjacency structure
        self.adjacency = self._build_adjacency(variables)
        
        def connected_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            if len(assignment) < len(variables):
                return True  # Can't check connectivity until all assigned
            
            # Group variables by color
            color_groups = defaultdict(list)
            for var, color in assignment.items():
                color_groups[color].append(var)
            
            # Check if each color group is connected
            for color, vars_list in color_groups.items():
                if not self._is_connected(vars_list):
                    return False
            
            return True
        
        super().__init__(variables, connected_predicate,
                        name or f"ConnectedComponents_{connectivity}")
    
    def _build_adjacency(self, variables: List[VariableName]) -> Dict[VariableName, List[VariableName]]:
        """Build adjacency list for the grid."""
        adjacency = defaultdict(list)
        
        # Parse positions
        var_positions = {}
        position_vars = {}
        
        for var in variables:
            parts = var.split('_')
            if len(parts) >= 3:
                row, col = int(parts[-2]), int(parts[-1])
                var_positions[var] = (row, col)
                position_vars[(row, col)] = var
        
        # Build adjacency based on connectivity
        for var, (row, col) in var_positions.items():
            neighbors = []
            
            # 4-connectivity
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                if (new_row, new_col) in position_vars:
                    neighbors.append(position_vars[(new_row, new_col)])
            
            # Additional neighbors for 8-connectivity
            if self.connectivity == '8':
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    new_row, new_col = row + dr, col + dc
                    if (new_row, new_col) in position_vars:
                        neighbors.append(position_vars[(new_row, new_col)])
            
            adjacency[var] = neighbors
        
        return adjacency
    
    def _is_connected(self, variables: List[VariableName]) -> bool:
        """Check if a set of variables forms a connected component."""
        if not variables:
            return True
        
        # BFS from first variable
        visited = {variables[0]}
        queue = [variables[0]]
        var_set = set(variables)
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in self.adjacency[current]:
                if neighbor in var_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(variables)


class RegionConstraint(NAryConstraint):
    """
    Constraint for rectangular regions in a grid.
    
    Ensures that a rectangular region satisfies certain properties.
    """
    
    def __init__(self, variables: List[VariableName],
                 region_property: str,
                 target_value: Optional[Any] = None,
                 name: Optional[str] = None):
        """
        Initialize region constraint.
        
        Args:
            variables: Variables in the region
            region_property: Property to enforce ('uniform', 'sum', 'product', 'min', 'max')
            target_value: Target value for the property
            name: Optional name for the constraint
        """
        self.region_property = region_property
        self.target_value = target_value
        
        def region_predicate(assignment: Dict[VariableName, DomainValue]) -> bool:
            if not assignment:
                return True
            
            values = list(assignment.values())
            
            if region_property == 'uniform':
                # All values in region must be the same
                return len(set(values)) <= 1
            
            elif region_property == 'sum' and target_value is not None:
                # Sum of values must equal target
                if len(assignment) == len(variables):
                    return sum(values) == target_value
                else:
                    # Partial sum shouldn't exceed target
                    return sum(values) <= target_value
            
            elif region_property == 'product' and target_value is not None:
                # Product of values must equal target
                if len(assignment) == len(variables):
                    product = 1
                    for v in values:
                        product *= v
                    return product == target_value
                return True
            
            elif region_property == 'min' and target_value is not None:
                # Minimum value must equal target
                return min(values) >= target_value
            
            elif region_property == 'max' and target_value is not None:
                # Maximum value must equal target
                return max(values) <= target_value
            
            return True
        
        super().__init__(variables, region_predicate,
                        name or f"Region_{region_property}")


def create_sudoku_constraints(csp, grid_size: int = 9, box_size: int = 3) -> None:
    """
    Add Sudoku constraints to a CSP.
    
    Args:
        csp: The CSP to add constraints to
        grid_size: Size of the Sudoku grid (usually 9)
        box_size: Size of each box (usually 3)
    """
    # Row constraints
    for row in range(grid_size):
        row_vars = [f"cell_{row}_{col}" for col in range(grid_size)]
        csp.add_constraint(AllDifferentConstraint(row_vars, f"Row_{row}"))
    
    # Column constraints
    for col in range(grid_size):
        col_vars = [f"cell_{row}_{col}" for row in range(grid_size)]
        csp.add_constraint(AllDifferentConstraint(col_vars, f"Col_{col}"))
    
    # Box constraints
    for box_row in range(0, grid_size, box_size):
        for box_col in range(0, grid_size, box_size):
            box_vars = []
            for r in range(box_row, box_row + box_size):
                for c in range(box_col, box_col + box_size):
                    box_vars.append(f"cell_{r}_{c}")
            csp.add_constraint(AllDifferentConstraint(box_vars, 
                                                     f"Box_{box_row}_{box_col}"))