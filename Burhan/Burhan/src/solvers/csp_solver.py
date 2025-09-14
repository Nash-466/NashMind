"""
CSP-based Solver for ARC Prize 2025
Uses Constraint Satisfaction Problems to solve ARC tasks
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import copy

from ..arc.grid_operations import Grid
from ..arc.pattern_detector import PatternDetector
from ..csp.core import CSP, Variable, Domain, Constraint
from ..csp.constraints import AllDifferentConstraint, FunctionConstraint
from ..csp.arc_consistency import AC3
from ..csp.search import BacktrackingSearch, VariableOrdering, ValueOrdering


@dataclass
class GridVariable(Variable):
    """Variable representing a cell or region in the grid"""
    position: Tuple[int, int] = (0, 0)
    region_id: Optional[int] = None
    color_constraint: Optional[Set[int]] = None
    
    def __post_init__(self):
        super().__init__(self.name, self.domain)
        
    def is_neighbor(self, other: 'GridVariable') -> bool:
        """Check if two variables are neighbors"""
        if not isinstance(other, GridVariable):
            return False
        
        y1, x1 = self.position
        y2, x2 = other.position
        
        # Adjacent cells (4-connected)
        return abs(y1 - y2) + abs(x1 - x2) == 1


@dataclass
class ColorConstraint(Constraint):
    """Constraint on cell colors"""
    allowed_colors: Set[int]
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        for var in self.variables:
            if var in assignment:
                if assignment[var] not in self.allowed_colors:
                    return False
        return True


@dataclass
class PatternConstraint(Constraint):
    """Constraint enforcing pattern relationships"""
    pattern_type: str  # 'repeat', 'mirror', 'rotate', etc.
    source_vars: List[str]
    target_vars: List[str]
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        # Check if all variables are assigned
        all_assigned = all(v in assignment for v in self.variables)
        if not all_assigned:
            return True  # Can't check partial assignment
        
        if self.pattern_type == 'repeat':
            # Target should repeat source
            source_values = [assignment[v] for v in self.source_vars]
            target_values = [assignment[v] for v in self.target_vars]
            
            if len(target_values) >= len(source_values):
                # Check if target starts with source pattern
                for i in range(0, len(target_values), len(source_values)):
                    chunk = target_values[i:i+len(source_values)]
                    if chunk != source_values[:len(chunk)]:
                        return False
            return True
        
        elif self.pattern_type == 'mirror':
            # Target should be mirror of source
            source_values = [assignment[v] for v in self.source_vars]
            target_values = [assignment[v] for v in self.target_vars]
            return target_values == source_values[::-1]
        
        elif self.pattern_type == 'equal':
            # All variables should have same value
            values = [assignment[v] for v in self.variables]
            return len(set(values)) == 1
        
        return True


@dataclass
class SpatialConstraint(Constraint):
    """Constraint on spatial relationships between cells"""
    constraint_type: str  # 'connected', 'separated', 'aligned', etc.
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        if self.constraint_type == 'connected':
            # Check if assigned cells form connected component
            return self._check_connectivity(assignment)
        
        elif self.constraint_type == 'separated':
            # Check if regions are separated
            return self._check_separation(assignment)
        
        elif self.constraint_type == 'aligned':
            # Check if cells are aligned (row/column)
            return self._check_alignment(assignment)
        
        return True
    
    def _check_connectivity(self, assignment: Dict[str, Any]) -> bool:
        """Check if assigned cells form connected component"""
        # Implementation would check 4-connectivity
        return True
    
    def _check_separation(self, assignment: Dict[str, Any]) -> bool:
        """Check if regions are properly separated"""
        return True
    
    def _check_alignment(self, assignment: Dict[str, Any]) -> bool:
        """Check if cells are aligned"""
        return True


class ARCtoCSPConverter:
    """Converts ARC tasks to CSP problems"""
    
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.grid_shape = None
    
    def convert_task(self, 
                     input_grid: Grid,
                     output_grid: Optional[Grid] = None,
                     examples: Optional[List[Tuple[Grid, Grid]]] = None) -> CSP:
        """Convert ARC task to CSP problem"""
        
        self.grid_shape = output_grid.shape if output_grid else input_grid.shape
        
        # Create variables for each cell
        self._create_cell_variables(input_grid, output_grid)
        
        # Add constraints based on patterns
        if examples:
            self._infer_constraints_from_examples(examples)
        else:
            self._infer_constraints_from_pair(input_grid, output_grid)
        
        # Create and return CSP
        return CSP(self.variables, self.constraints)
    
    def _create_cell_variables(self, 
                              input_grid: Grid,
                              output_grid: Optional[Grid] = None):
        """Create variables for each cell in the grid"""
        self.variables = []
        
        height, width = self.grid_shape
        
        for y in range(height):
            for x in range(width):
                var_name = f"cell_{y}_{x}"
                
                # Determine domain based on input and output
                if output_grid and y < output_grid.height and x < output_grid.width:
                    # If we know the output, domain is just that value
                    domain = {int(output_grid.data[y, x])}
                else:
                    # Otherwise, domain is all possible colors
                    domain = set(range(10))
                
                var = GridVariable(
                    name=var_name,
                    domain=domain,
                    position=(y, x)
                )
                
                self.variables.append(var)
    
    def _infer_constraints_from_examples(self, 
                                        examples: List[Tuple[Grid, Grid]]):
        """Infer constraints from training examples"""
        
        # Analyze patterns across examples
        for input_grid, output_grid in examples:
            # Detect patterns in output
            detector = PatternDetector(output_grid)
            
            # Check for symmetries
            symmetries = detector.get_symmetries()
            if symmetries['horizontal']:
                self._add_symmetry_constraint('horizontal')
            if symmetries['vertical']:
                self._add_symmetry_constraint('vertical')
            
            # Check for repeating patterns
            patterns = detector.find_repeating_patterns()
            if patterns:
                self._add_pattern_constraints(patterns)
            
            # Check for color relationships
            self._add_color_constraints(input_grid, output_grid)
            
            # Check for spatial relationships
            self._add_spatial_constraints(input_grid, output_grid)
    
    def _infer_constraints_from_pair(self, 
                                    input_grid: Grid,
                                    output_grid: Grid):
        """Infer constraints from single input/output pair"""
        
        # Basic color mapping constraint
        input_colors = input_grid.unique_colors
        output_colors = output_grid.unique_colors
        
        # If colors are preserved, add color preservation constraint
        if input_colors == output_colors:
            for y in range(min(input_grid.height, output_grid.height)):
                for x in range(min(input_grid.width, output_grid.width)):
                    if input_grid.data[y, x] == output_grid.data[y, x]:
                        # This cell preserves its color
                        var_name = f"cell_{y}_{x}"
                        var = next((v for v in self.variables if v.name == var_name), None)
                        if var:
                            var.domain = Domain([input_grid.data[y, x]])
    
    def _add_symmetry_constraint(self, symmetry_type: str):
        """Add symmetry constraint"""
        height, width = self.grid_shape
        
        if symmetry_type == 'horizontal':
            # Left-right symmetry
            for y in range(height):
                for x in range(width // 2):
                    left_var = f"cell_{y}_{x}"
                    right_var = f"cell_{y}_{width - 1 - x}"
                    
                    constraint = PatternConstraint(
                        variables=[left_var, right_var],
                        pattern_type='equal',
                        source_vars=[left_var],
                        target_vars=[right_var]
                    )
                    self.constraints.append(constraint)
        
        elif symmetry_type == 'vertical':
            # Top-bottom symmetry
            for y in range(height // 2):
                for x in range(width):
                    top_var = f"cell_{y}_{x}"
                    bottom_var = f"cell_{height - 1 - y}_{x}"
                    
                    constraint = PatternConstraint(
                        variables=[top_var, bottom_var],
                        pattern_type='equal',
                        source_vars=[top_var],
                        target_vars=[bottom_var]
                    )
                    self.constraints.append(constraint)
    
    def _add_pattern_constraints(self, patterns: List[Tuple[Grid, int]]):
        """Add constraints for repeating patterns"""
        if not patterns:
            return
        
        # Take the most frequent pattern
        pattern_grid, frequency = patterns[0]
        pattern_h, pattern_w = pattern_grid.shape
        
        height, width = self.grid_shape
        
        # Add constraints for pattern repetition
        for y in range(0, height - pattern_h + 1, pattern_h):
            for x in range(0, width - pattern_w + 1, pattern_w):
                # Create constraint for this pattern instance
                pattern_vars = []
                for py in range(pattern_h):
                    for px in range(pattern_w):
                        var_name = f"cell_{y + py}_{x + px}"
                        pattern_vars.append(var_name)
                
                # Constraint that enforces the pattern
                def pattern_checker(assignment, pvars=pattern_vars, pgrid=pattern_grid):
                    for i, var in enumerate(pvars):
                        if var in assignment:
                            py = i // pgrid.width
                            px = i % pgrid.width
                            if assignment[var] != pgrid.data[py, px]:
                                return False
                    return True
                
                constraint = FunctionConstraint(
                    variables=pattern_vars,
                    function=pattern_checker
                )
                self.constraints.append(constraint)
    
    def _add_color_constraints(self, input_grid: Grid, output_grid: Grid):
        """Add color-based constraints"""
        input_colors = input_grid.unique_colors
        output_colors = output_grid.unique_colors
        
        # If certain colors are never used, exclude them from domains
        unused_colors = set(range(10)) - output_colors
        
        for var in self.variables:
            if isinstance(var, GridVariable):
                var.domain = var.domain - unused_colors
        
        # Add color mapping constraints if there's a clear mapping
        color_map = self._infer_color_mapping(input_grid, output_grid)
        if color_map:
            for y in range(min(input_grid.height, output_grid.height)):
                for x in range(min(input_grid.width, output_grid.width)):
                    input_color = input_grid.data[y, x]
                    if input_color in color_map:
                        var_name = f"cell_{y}_{x}"
                        var = next((v for v in self.variables if v.name == var_name), None)
                        if var:
                            var.domain = {color_map[input_color]}
    
    def _infer_color_mapping(self, 
                           input_grid: Grid,
                           output_grid: Grid) -> Optional[Dict[int, int]]:
        """Infer color mapping between input and output"""
        if input_grid.shape != output_grid.shape:
            return None
        
        color_map = {}
        
        for color in input_grid.unique_colors:
            input_positions = set(input_grid.get_color_positions(color))
            
            # Find which color these positions map to in output
            output_colors_at_positions = set()
            for y, x in input_positions:
                if y < output_grid.height and x < output_grid.width:
                    output_colors_at_positions.add(output_grid.data[y, x])
            
            # If all map to same color, we have a mapping
            if len(output_colors_at_positions) == 1:
                color_map[color] = output_colors_at_positions.pop()
        
        return color_map if color_map else None
    
    def _add_spatial_constraints(self, input_grid: Grid, output_grid: Grid):
        """Add spatial relationship constraints"""
        
        # Check for connected components
        detector = PatternDetector(output_grid)
        components = detector.get_connected_components()
        
        # Add connectivity constraints for each component
        for component in components:
            if component['color'] == 0:  # Skip background
                continue
            
            component_vars = []
            positions = component.get('positions', component.get('pixels', []))
            for y, x in positions:
                var_name = f"cell_{y}_{x}"
                component_vars.append(var_name)
            
            if len(component_vars) > 1:
                constraint = SpatialConstraint(
                    variables=component_vars,
                    constraint_type='connected'
                )
                self.constraints.append(constraint)


class CSPSolver:
    """CSP-based solver for ARC tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.converter = ARCtoCSPConverter()
        # BacktrackingSearch will be initialized with a CSP in solve method
        self.search_config = {
            'variable_ordering': VariableOrdering.MRV_DEGREE,
            'value_ordering': ValueOrdering.LCV
        }
        
        self.max_iterations = config.get('max_iterations', 10000)
        self.timeout = config.get('timeout', 5.0)
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve ARC task using CSP approach"""
        
        # Convert to Grid objects
        train_input_grids = [Grid(inp) for inp in train_inputs]
        train_output_grids = [Grid(out) for out in train_outputs]
        test_input_grid = Grid(test_input)
        
        # Convert to CSP
        examples = list(zip(train_input_grids, train_output_grids))
        
        # Determine output shape (assume same as input for now)
        output_shape = self._infer_output_shape(
            train_input_grids, 
            train_output_grids,
            test_input_grid
        )
        
        # Create template output grid
        template_output = Grid(np.zeros(output_shape, dtype=np.int8))
        
        # Convert to CSP
        csp = self.converter.convert_task(
            test_input_grid,
            template_output,
            examples
        )
        
        # Apply arc consistency
        ac3 = AC3(csp)
        ac3.run()
        
        # Create search with the CSP
        search = BacktrackingSearch(
            csp,
            variable_ordering=self.search_config['variable_ordering'],
            value_ordering=self.search_config['value_ordering']
        )
        
        # Search for solution
        solution = search.solve()
        
        if solution:
            # Convert solution to grid
            return self._solution_to_grid(solution, output_shape)
        
        return None
    
    def _infer_output_shape(self,
                          train_inputs: List[Grid],
                          train_outputs: List[Grid],
                          test_input: Grid) -> Tuple[int, int]:
        """Infer the output shape based on examples"""
        
        # Check if output shape is consistent across examples
        output_shapes = [out.shape for out in train_outputs]
        
        if len(set(output_shapes)) == 1:
            # All outputs have same shape
            output_shape = output_shapes[0]
            
            # Check if it's a fixed shape or relative to input
            input_shapes = [inp.shape for inp in train_inputs]
            
            if len(set(input_shapes)) == 1 and input_shapes[0] == output_shape:
                # Same shape as input
                return test_input.shape
            else:
                # Fixed output shape
                return output_shape
        
        # Try to infer scaling relationship
        scale_factors = []
        for inp, out in zip(train_inputs, train_outputs):
            if out.height % inp.height == 0 and out.width % inp.width == 0:
                h_scale = out.height // inp.height
                w_scale = out.width // inp.width
                if h_scale == w_scale:
                    scale_factors.append(h_scale)
        
        if scale_factors and len(set(scale_factors)) == 1:
            # Consistent scaling
            scale = scale_factors[0]
            return (test_input.height * scale, test_input.width * scale)
        
        # Default: same as input
        return test_input.shape
    
    def _solution_to_grid(self, 
                        solution: Dict[str, Any],
                        shape: Tuple[int, int]) -> np.ndarray:
        """Convert CSP solution to grid"""
        height, width = shape
        grid = np.zeros((height, width), dtype=np.int8)
        
        for y in range(height):
            for x in range(width):
                var_name = f"cell_{y}_{x}"
                if var_name in solution:
                    grid[y, x] = solution[var_name]
        
        return grid
    
    def solve_with_constraints(self,
                              input_grid: Grid,
                              constraints: List[Constraint]) -> Optional[Grid]:
        """Solve with explicit constraints"""
        
        # Create CSP with given constraints
        height, width = input_grid.shape
        variables = []
        
        for y in range(height):
            for x in range(width):
                var = GridVariable(
                    name=f"cell_{y}_{x}",
                    domain=Domain(list(range(10))),
                    position=(y, x)
                )
                variables.append(var)
        
        csp = CSP(variables, constraints)
        
        # Apply arc consistency
        ac3 = AC3(csp)
        ac3.run()
        
        # Create search with the CSP
        search = BacktrackingSearch(
            csp,
            variable_ordering=self.search_config['variable_ordering'],
            value_ordering=self.search_config['value_ordering']
        )
        
        # Search for solution
        solution = search.solve()
        
        if solution:
            # Convert to grid
            result_data = self._solution_to_grid(solution, input_grid.shape)
            return Grid(result_data)
        
        return None