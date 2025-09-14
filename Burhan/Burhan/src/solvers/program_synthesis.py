"""
Program Synthesis Engine for ARC Prize 2025
Implements program synthesis with DSL, beam search, and program optimization
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import product, combinations
import time
import copy

from ..arc.grid_operations import Grid
from ..arc.pattern_detector import PatternDetector
from ..arc.transformation_rules import TransformationType, TransformationRule


class DSLOperation(Enum):
    """Domain-Specific Language operations for grid transformations"""
    # Geometric operations
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"
    TRANSPOSE = "transpose"
    
    # Scaling operations
    SCALE_2X = "scale_2x"
    SCALE_3X = "scale_3x"
    DOWNSCALE_2X = "downscale_2x"
    
    # Tiling operations
    TILE_2X2 = "tile_2x2"
    TILE_3X3 = "tile_3x3"
    MIRROR_H = "mirror_h"
    MIRROR_V = "mirror_v"
    
    # Color operations
    REPLACE_COLOR = "replace_color"
    FILTER_COLOR = "filter_color"
    INVERT_COLORS = "invert_colors"
    MAP_COLORS = "map_colors"
    
    # Pattern operations
    EXTRACT_PATTERN = "extract_pattern"
    REPEAT_PATTERN = "repeat_pattern"
    OVERLAY = "overlay"
    MASK = "mask"
    
    # Structural operations
    CROP_TO_CONTENT = "crop_to_content"
    PAD = "pad"
    EXTRACT_SUBGRID = "extract_subgrid"
    FILL_BACKGROUND = "fill_background"
    
    # Connected components
    EXTRACT_LARGEST = "extract_largest"
    EXTRACT_SMALLEST = "extract_smallest"
    EXTRACT_BY_COLOR = "extract_by_color"
    
    # Conditional operations
    IF_SYMMETRIC = "if_symmetric"
    IF_CONTAINS_COLOR = "if_contains_color"
    IF_SIZE = "if_size"
    
    # Composite operations
    COMPOSE = "compose"
    LOOP = "loop"
    APPLY_TO_EACH = "apply_to_each"


@dataclass
class DSLInstruction:
    """Single instruction in the DSL program"""
    operation: DSLOperation
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        param_tuple = tuple(sorted(self.parameters.items()))
        return hash((self.operation, param_tuple))


@dataclass
class Program:
    """Represents a transformation program"""
    instructions: List[DSLInstruction]
    score: float = 0.0
    complexity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.complexity = len(self.instructions)
        for inst in self.instructions:
            # Add complexity for parameterized operations
            if inst.parameters:
                self.complexity += len(inst.parameters)
    
    def execute(self, grid: Grid) -> Grid:
        """Execute the program on a grid"""
        result = grid.copy()
        
        for instruction in self.instructions:
            try:
                result = self._execute_instruction(result, instruction)
            except Exception as e:
                # Program execution failed
                raise ProgramExecutionError(f"Failed at {instruction.operation}: {str(e)}")
        
        return result
    
    def _execute_instruction(self, grid: Grid, instruction: DSLInstruction) -> Grid:
        """Execute a single instruction"""
        op = instruction.operation
        params = instruction.parameters
        
        # Geometric operations
        if op == DSLOperation.ROTATE_90:
            return grid.rotate(90)
        elif op == DSLOperation.ROTATE_180:
            return grid.rotate(180)
        elif op == DSLOperation.ROTATE_270:
            return grid.rotate(270)
        elif op == DSLOperation.FLIP_H:
            return grid.flip_horizontal()
        elif op == DSLOperation.FLIP_V:
            return grid.flip_vertical()
        elif op == DSLOperation.TRANSPOSE:
            return grid.transpose()
        
        # Scaling operations
        elif op == DSLOperation.SCALE_2X:
            return grid.scale(2)
        elif op == DSLOperation.SCALE_3X:
            return grid.scale(3)
        elif op == DSLOperation.DOWNSCALE_2X:
            return self._downscale(grid, 2)
        
        # Tiling operations
        elif op == DSLOperation.TILE_2X2:
            return grid.tile(2, 2)
        elif op == DSLOperation.TILE_3X3:
            return grid.tile(3, 3)
        elif op == DSLOperation.MIRROR_H:
            return grid.mirror('horizontal')
        elif op == DSLOperation.MIRROR_V:
            return grid.mirror('vertical')
        
        # Color operations
        elif op == DSLOperation.REPLACE_COLOR:
            return grid.replace_color(params['old_color'], params['new_color'])
        elif op == DSLOperation.FILTER_COLOR:
            return grid.filter_color(params['color'], params.get('background', 0))
        elif op == DSLOperation.INVERT_COLORS:
            return self._invert_colors(grid)
        elif op == DSLOperation.MAP_COLORS:
            return grid.map_colors(params['color_map'])
        
        # Pattern operations
        elif op == DSLOperation.EXTRACT_PATTERN:
            return self._extract_pattern(grid, params)
        elif op == DSLOperation.REPEAT_PATTERN:
            return self._repeat_pattern(grid, params)
        elif op == DSLOperation.OVERLAY:
            other = params['other']
            return grid.overlay(other, params.get('x', 0), params.get('y', 0))
        elif op == DSLOperation.MASK:
            return grid.mask(params['mask'], params.get('mask_value', 0))
        
        # Structural operations
        elif op == DSLOperation.CROP_TO_CONTENT:
            return grid.crop_to_content(params.get('background', 0))
        elif op == DSLOperation.PAD:
            return grid.pad(params.get('padding', 1), params.get('value', 0))
        elif op == DSLOperation.EXTRACT_SUBGRID:
            return grid.extract_subgrid(
                params['y'], params['x'], 
                params['height'], params['width']
            )
        elif op == DSLOperation.FILL_BACKGROUND:
            return self._fill_background(grid, params.get('color', 0))
        
        # Connected components
        elif op == DSLOperation.EXTRACT_LARGEST:
            return grid.extract_largest_component(params.get('background', 0))
        elif op == DSLOperation.EXTRACT_SMALLEST:
            return grid.extract_smallest_component(params.get('background', 0))
        elif op == DSLOperation.EXTRACT_BY_COLOR:
            return grid.filter_color(params['color'])
        
        # Conditional operations
        elif op == DSLOperation.IF_SYMMETRIC:
            if self._check_symmetry(grid, params.get('type', 'horizontal')):
                return self._execute_instruction(grid, params['then'])
            elif 'else' in params:
                return self._execute_instruction(grid, params['else'])
            return grid
        
        # Composite operations
        elif op == DSLOperation.COMPOSE:
            result = grid
            for sub_inst in params['instructions']:
                result = self._execute_instruction(result, sub_inst)
            return result
        
        elif op == DSLOperation.LOOP:
            result = grid
            for _ in range(params.get('times', 1)):
                result = self._execute_instruction(result, params['instruction'])
            return result
        
        else:
            raise ValueError(f"Unknown operation: {op}")
    
    def _downscale(self, grid: Grid, factor: int) -> Grid:
        """Downscale grid by taking every nth pixel"""
        data = grid.data[::factor, ::factor]
        return Grid(data)
    
    def _invert_colors(self, grid: Grid) -> Grid:
        """Invert colors (swap non-zero colors)"""
        color_map = {}
        unique_colors = sorted(grid.unique_colors)
        if 0 in unique_colors:
            unique_colors.remove(0)
        
        # Create inversion mapping
        for i, color in enumerate(unique_colors):
            color_map[color] = unique_colors[-(i+1)]
        
        return grid.map_colors(color_map)
    
    def _extract_pattern(self, grid: Grid, params: Dict) -> Grid:
        """Extract a repeating pattern from the grid"""
        detector = PatternDetector(grid)
        patterns = detector.find_repeating_patterns(
            min_size=params.get('min_size', 2),
            max_size=params.get('max_size', None)
        )
        
        if patterns:
            return patterns[0][0]  # Return most frequent pattern
        return grid
    
    def _repeat_pattern(self, grid: Grid, params: Dict) -> Grid:
        """Repeat a pattern to fill a larger grid"""
        rows = params.get('rows', 2)
        cols = params.get('cols', 2)
        return grid.tile(rows, cols)
    
    def _fill_background(self, grid: Grid, color: int) -> Grid:
        """Fill background (0) with specified color"""
        result = grid.copy()
        result.data[result.data == 0] = color
        return result
    
    def _check_symmetry(self, grid: Grid, sym_type: str) -> bool:
        """Check if grid has specified symmetry"""
        detector = PatternDetector(grid)
        if sym_type == 'horizontal':
            return detector.has_horizontal_symmetry()
        elif sym_type == 'vertical':
            return detector.has_vertical_symmetry()
        elif sym_type == 'diagonal':
            return detector.has_diagonal_symmetry()
        return False
    
    def to_string(self) -> str:
        """Convert program to string representation"""
        lines = []
        for inst in self.instructions:
            if inst.parameters:
                param_str = ', '.join(f"{k}={v}" for k, v in inst.parameters.items())
                lines.append(f"{inst.operation.value}({param_str})")
            else:
                lines.append(inst.operation.value)
        return '\n'.join(lines)


class ProgramExecutionError(Exception):
    """Raised when program execution fails"""
    pass


class ProgramGenerator:
    """Generates candidate programs for solving ARC tasks"""
    
    def __init__(self, max_length: int = 5, allow_composite: bool = True):
        self.max_length = max_length
        self.allow_composite = allow_composite
        self.operation_templates = self._create_operation_templates()
    
    def _create_operation_templates(self) -> Dict[DSLOperation, List[Dict]]:
        """Create templates for parameterized operations"""
        templates = {
            # Simple geometric operations (no parameters)
            DSLOperation.ROTATE_90: [{}],
            DSLOperation.ROTATE_180: [{}],
            DSLOperation.ROTATE_270: [{}],
            DSLOperation.FLIP_H: [{}],
            DSLOperation.FLIP_V: [{}],
            DSLOperation.TRANSPOSE: [{}],
            
            # Scaling operations
            DSLOperation.SCALE_2X: [{}],
            DSLOperation.SCALE_3X: [{}],
            DSLOperation.DOWNSCALE_2X: [{}],
            
            # Tiling operations
            DSLOperation.TILE_2X2: [{}],
            DSLOperation.TILE_3X3: [{}],
            DSLOperation.MIRROR_H: [{}],
            DSLOperation.MIRROR_V: [{}],
            
            # Color operations (need parameters)
            DSLOperation.REPLACE_COLOR: [
                {'old_color': old, 'new_color': new}
                for old in range(10) for new in range(10) if old != new
            ][:20],  # Limit to avoid explosion
            
            DSLOperation.FILTER_COLOR: [
                {'color': c, 'background': 0} for c in range(1, 10)
            ],
            
            DSLOperation.INVERT_COLORS: [{}],
            
            # Structural operations
            DSLOperation.CROP_TO_CONTENT: [{'background': 0}],
            DSLOperation.PAD: [{'padding': p, 'value': 0} for p in [1, 2]],
            DSLOperation.FILL_BACKGROUND: [{'color': c} for c in range(1, 10)],
            
            # Component operations
            DSLOperation.EXTRACT_LARGEST: [{'background': 0}],
            DSLOperation.EXTRACT_SMALLEST: [{'background': 0}],
        }
        
        return templates
    
    def generate_atomic_programs(self, input_grid: Grid, output_grid: Grid) -> List[Program]:
        """Generate single-instruction programs"""
        programs = []
        
        # Analyze grids to guide generation
        input_colors = input_grid.unique_colors
        output_colors = output_grid.unique_colors
        
        for op, param_templates in self.operation_templates.items():
            for params in param_templates:
                # Filter color operations based on actual colors
                if op in [DSLOperation.REPLACE_COLOR, DSLOperation.FILTER_COLOR]:
                    if 'old_color' in params and params['old_color'] not in input_colors:
                        continue
                    if 'color' in params and params['color'] not in input_colors:
                        continue
                
                instruction = DSLInstruction(op, params)
                program = Program([instruction])
                programs.append(program)
        
        return programs
    
    def generate_composite_programs(self, 
                                  atomic_programs: List[Program],
                                  max_length: int = None) -> List[Program]:
        """Generate composite programs by combining atomic programs"""
        if max_length is None:
            max_length = self.max_length
        
        composite_programs = []
        
        # Generate programs of length 2
        for prog1 in atomic_programs:
            for prog2 in atomic_programs:
                if len(prog1.instructions) + len(prog2.instructions) <= max_length:
                    combined = Program(prog1.instructions + prog2.instructions)
                    composite_programs.append(combined)
        
        # Generate programs of length 3+ (limited to avoid explosion)
        if max_length >= 3:
            for prog in composite_programs[:100]:  # Limit for efficiency
                for atomic in atomic_programs[:20]:
                    if len(prog.instructions) + len(atomic.instructions) <= max_length:
                        extended = Program(prog.instructions + atomic.instructions)
                        composite_programs.append(extended)
        
        return composite_programs
    
    def generate_from_examples(self, 
                              train_inputs: List[Grid],
                              train_outputs: List[Grid]) -> List[Program]:
        """Generate programs based on training examples"""
        all_programs = []
        
        # Generate for each training pair
        for input_grid, output_grid in zip(train_inputs, train_outputs):
            # Analyze transformation characteristics
            size_changed = input_grid.shape != output_grid.shape
            colors_changed = input_grid.unique_colors != output_grid.unique_colors
            
            # Generate atomic programs
            atomic = self.generate_atomic_programs(input_grid, output_grid)
            all_programs.extend(atomic)
            
            # Generate composite programs if needed
            if self.allow_composite:
                composite = self.generate_composite_programs(atomic, self.max_length)
                all_programs.extend(composite)
        
        # Remove duplicates
        unique_programs = []
        seen = set()
        for prog in all_programs:
            prog_str = prog.to_string()
            if prog_str not in seen:
                seen.add(prog_str)
                unique_programs.append(prog)
        
        return unique_programs


class ProgramEvaluator:
    """Evaluates programs on ARC examples"""
    
    def __init__(self, timeout: float = 1.0):
        self.timeout = timeout
    
    def evaluate(self, 
                program: Program,
                input_grids: List[Grid],
                output_grids: List[Grid]) -> float:
        """Evaluate program on examples, return score"""
        if not input_grids or not output_grids:
            return 0.0
        
        total_score = 0.0
        
        for input_grid, expected_output in zip(input_grids, output_grids):
            try:
                # Execute program with timeout
                result = self._execute_with_timeout(program, input_grid)
                
                # Calculate similarity score
                score = self._calculate_similarity(result, expected_output)
                total_score += score
                
            except (ProgramExecutionError, TimeoutError, Exception):
                # Program failed on this example
                total_score += 0.0
        
        # Average score across examples
        avg_score = total_score / len(input_grids)
        
        # Apply complexity penalty
        complexity_penalty = 1.0 / (1.0 + 0.1 * program.complexity)
        
        return avg_score * complexity_penalty
    
    def _execute_with_timeout(self, program: Program, grid: Grid) -> Grid:
        """Execute program with timeout (simplified version)"""
        # In production, use proper timeout mechanism
        start_time = time.time()
        result = program.execute(grid)
        
        if time.time() - start_time > self.timeout:
            raise TimeoutError("Program execution timeout")
        
        return result
    
    def _calculate_similarity(self, predicted: Grid, expected: Grid) -> float:
        """Calculate similarity between predicted and expected grids"""
        # Exact match
        if predicted == expected:
            return 1.0
        
        # Shape match bonus
        shape_score = 1.0 if predicted.shape == expected.shape else 0.5
        
        # If shapes don't match, can't compare directly
        if predicted.shape != expected.shape:
            # Check if one is scaled version of other
            if self._is_scaled_version(predicted, expected):
                return 0.7 * shape_score
            return 0.0
        
        # Pixel-wise accuracy
        matches = np.sum(predicted.data == expected.data)
        total = predicted.data.size
        pixel_accuracy = matches / total
        
        # Color distribution similarity
        pred_colors = predicted.count_colors()
        exp_colors = expected.count_colors()
        color_similarity = self._color_distribution_similarity(pred_colors, exp_colors)
        
        # Weighted combination
        return shape_score * (0.7 * pixel_accuracy + 0.3 * color_similarity)
    
    def _is_scaled_version(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if one grid is a scaled version of the other"""
        h1, w1 = grid1.shape
        h2, w2 = grid2.shape
        
        # Check for integer scaling
        if h1 % h2 == 0 and w1 % w2 == 0:
            scale = h1 // h2
            if w1 // w2 == scale:
                # Check if grid1 is scaled version of grid2
                scaled = grid2.scale(scale)
                return scaled == grid1
        
        if h2 % h1 == 0 and w2 % w1 == 0:
            scale = h2 // h1
            if w2 // w1 == scale:
                # Check if grid2 is scaled version of grid1
                scaled = grid1.scale(scale)
                return scaled == grid2
        
        return False
    
    def _color_distribution_similarity(self, 
                                     dist1: Dict[int, int],
                                     dist2: Dict[int, int]) -> float:
        """Calculate similarity between color distributions"""
        all_colors = set(dist1.keys()) | set(dist2.keys())
        
        if not all_colors:
            return 1.0
        
        total_diff = 0
        total_count = sum(dist1.values()) + sum(dist2.values())
        
        for color in all_colors:
            count1 = dist1.get(color, 0)
            count2 = dist2.get(color, 0)
            total_diff += abs(count1 - count2)
        
        if total_count == 0:
            return 1.0
        
        return 1.0 - (total_diff / total_count)


class BeamSearchSynthesizer:
    """Beam search for program synthesis"""
    
    def __init__(self, 
                 beam_width: int = 10,
                 max_iterations: int = 100,
                 generator: Optional[ProgramGenerator] = None,
                 evaluator: Optional[ProgramEvaluator] = None):
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.generator = generator or ProgramGenerator()
        self.evaluator = evaluator or ProgramEvaluator()
    
    def synthesize(self,
                  train_inputs: List[Grid],
                  train_outputs: List[Grid],
                  test_input: Optional[Grid] = None) -> Optional[Program]:
        """Synthesize program using beam search"""
        
        # Generate initial candidate programs
        candidates = self.generator.generate_from_examples(train_inputs, train_outputs)
        
        if not candidates:
            return None
        
        # Evaluate and score candidates
        scored_programs = []
        for program in candidates:
            score = self.evaluator.evaluate(program, train_inputs, train_outputs)
            program.score = score
            scored_programs.append(program)
        
        # Sort by score and keep top beam_width
        scored_programs.sort(key=lambda p: p.score, reverse=True)
        beam = scored_programs[:self.beam_width]
        
        # Check if we found perfect solution
        for program in beam:
            if program.score >= 0.99:
                return program
        
        # Beam search iterations
        for iteration in range(self.max_iterations):
            new_beam = []
            
            for program in beam:
                # Generate variations of this program
                variations = self._generate_variations(program, train_inputs[0], train_outputs[0])
                
                for variant in variations:
                    score = self.evaluator.evaluate(variant, train_inputs, train_outputs)
                    variant.score = score
                    new_beam.append(variant)
            
            # Combine old beam and new candidates
            all_candidates = beam + new_beam
            all_candidates.sort(key=lambda p: p.score, reverse=True)
            
            # Keep top beam_width
            beam = all_candidates[:self.beam_width]
            
            # Check for perfect solution
            if beam[0].score >= 0.99:
                return beam[0]
            
            # Check for convergence
            if iteration > 0 and len(new_beam) == 0:
                break
        
        # Return best program found
        return beam[0] if beam else None
    
    def _generate_variations(self, 
                           program: Program,
                           input_grid: Grid,
                           output_grid: Grid) -> List[Program]:
        """Generate variations of a program"""
        variations = []
        
        # Add single instruction
        atomic = self.generator.generate_atomic_programs(input_grid, output_grid)
        for atomic_prog in atomic[:10]:  # Limit for efficiency
            if len(program.instructions) + 1 <= self.generator.max_length:
                # Prepend
                new_prog = Program(atomic_prog.instructions + program.instructions)
                variations.append(new_prog)
                
                # Append
                new_prog = Program(program.instructions + atomic_prog.instructions)
                variations.append(new_prog)
        
        # Remove instruction
        if len(program.instructions) > 1:
            for i in range(len(program.instructions)):
                new_instructions = (program.instructions[:i] + 
                                  program.instructions[i+1:])
                variations.append(Program(new_instructions))
        
        # Modify parameters
        for i, instruction in enumerate(program.instructions):
            if instruction.parameters:
                # Try different parameter values
                param_variations = self._generate_parameter_variations(instruction)
                for variant_inst in param_variations[:5]:  # Limit
                    new_instructions = program.instructions.copy()
                    new_instructions[i] = variant_inst
                    variations.append(Program(new_instructions))
        
        return variations
    
    def _generate_parameter_variations(self, 
                                      instruction: DSLInstruction) -> List[DSLInstruction]:
        """Generate variations of instruction parameters"""
        variations = []
        
        if instruction.operation == DSLOperation.REPLACE_COLOR:
            # Try different color mappings
            for old_color in range(10):
                for new_color in range(10):
                    if old_color != new_color:
                        params = {'old_color': old_color, 'new_color': new_color}
                        variations.append(DSLInstruction(instruction.operation, params))
        
        elif instruction.operation == DSLOperation.PAD:
            # Try different padding sizes
            for padding in [1, 2, 3]:
                for value in [0, 1]:
                    params = {'padding': padding, 'value': value}
                    variations.append(DSLInstruction(instruction.operation, params))
        
        # Add more parameter variations for other operations...
        
        return variations


class ProgramSynthesisEngine:
    """Main engine for program synthesis-based solving"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.generator = ProgramGenerator(
            max_length=config.get('max_program_length', 5),
            allow_composite=config.get('allow_composite', True)
        )
        
        self.evaluator = ProgramEvaluator(
            timeout=config.get('execution_timeout', 1.0)
        )
        
        self.synthesizer = BeamSearchSynthesizer(
            beam_width=config.get('beam_width', 10),
            max_iterations=config.get('max_iterations', 100),
            generator=self.generator,
            evaluator=self.evaluator
        )
    
    def solve(self, 
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve ARC task using program synthesis"""
        
        # Convert to Grid objects
        train_input_grids = [Grid(inp) for inp in train_inputs]
        train_output_grids = [Grid(out) for out in train_outputs]
        test_input_grid = Grid(test_input)
        
        # Synthesize program
        program = self.synthesizer.synthesize(
            train_input_grids,
            train_output_grids,
            test_input_grid
        )
        
        if program is None:
            return None
        
        try:
            # Apply program to test input
            result_grid = program.execute(test_input_grid)
            return result_grid.data
        except Exception:
            return None
    
    def get_program(self,
                   train_inputs: List[np.ndarray],
                   train_outputs: List[np.ndarray]) -> Optional[Program]:
        """Get the synthesized program without applying to test"""
        
        # Convert to Grid objects
        train_input_grids = [Grid(inp) for inp in train_inputs]
        train_output_grids = [Grid(out) for out in train_outputs]
        
        # Synthesize program
        return self.synthesizer.synthesize(
            train_input_grids,
            train_output_grids
        )