"""
Pattern-based Solver for ARC Prize 2025
Solves ARC tasks by detecting and applying common patterns
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..arc.grid_operations import Grid
from ..arc.pattern_detector import PatternDetector


class PatternType(Enum):
    """Types of patterns commonly found in ARC tasks"""
    TILING = "tiling"
    SCALING = "scaling"
    SYMMETRY = "symmetry"
    ROTATION = "rotation"
    COLOR_MAPPING = "color_mapping"
    REPETITION = "repetition"
    MIRRORING = "mirroring"
    OVERLAY = "overlay"
    EXTRACTION = "extraction"
    FILLING = "filling"
    CONNECTIVITY = "connectivity"
    COUNTING = "counting"


@dataclass
class DetectedPattern:
    """Represents a detected pattern in the task"""
    pattern_type: PatternType
    confidence: float
    parameters: Dict[str, Any]
    transformation: Optional[Callable] = None


class PatternSolver:
    """Solves ARC tasks by detecting and applying patterns"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_patterns = config.get('max_patterns', 10)
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve ARC task using pattern detection"""
        
        # Convert to Grid objects
        train_input_grids = [Grid(inp) for inp in train_inputs]
        train_output_grids = [Grid(out) for out in train_outputs]
        test_input_grid = Grid(test_input)
        
        # Detect patterns from training examples
        patterns = self._detect_patterns(train_input_grids, train_output_grids)
        
        if not patterns:
            return None
        
        # Sort patterns by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Try applying patterns
        for pattern in patterns[:self.max_patterns]:
            if pattern.confidence < self.min_confidence:
                break
            
            try:
                result = self._apply_pattern(test_input_grid, pattern)
                
                # Validate result on training examples
                if self._validate_pattern(pattern, train_input_grids, train_output_grids):
                    return result.data
            except Exception:
                continue
        
        return None
    
    def _detect_patterns(self, 
                        inputs: List[Grid],
                        outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect patterns from training examples"""
        patterns = []
        
        # Check each type of pattern
        patterns.extend(self._detect_tiling_patterns(inputs, outputs))
        patterns.extend(self._detect_scaling_patterns(inputs, outputs))
        patterns.extend(self._detect_symmetry_patterns(inputs, outputs))
        patterns.extend(self._detect_color_patterns(inputs, outputs))
        patterns.extend(self._detect_repetition_patterns(inputs, outputs))
        patterns.extend(self._detect_mirroring_patterns(inputs, outputs))
        patterns.extend(self._detect_extraction_patterns(inputs, outputs))
        patterns.extend(self._detect_filling_patterns(inputs, outputs))
        patterns.extend(self._detect_connectivity_patterns(inputs, outputs))
        
        return patterns
    
    def _detect_tiling_patterns(self, 
                               inputs: List[Grid],
                               outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect tiling patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            # Check if output is tiled version of input
            if out.height % inp.height == 0 and out.width % inp.width == 0:
                rows = out.height // inp.height
                cols = out.width // inp.width
                
                # Verify tiling
                expected = inp.tile(rows, cols)
                if expected == out:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.TILING,
                        confidence=1.0,
                        parameters={'rows': rows, 'cols': cols}
                    )
                    patterns.append(pattern)
                    break
            
            # Check if output contains repeated subpattern
            detector = PatternDetector(out)
            repeating = detector.find_repeating_patterns()
            
            if repeating:
                pattern_grid, count = repeating[0]
                if count > 2:
                    # Check if input matches the pattern
                    if pattern_grid.data.shape == inp.data.shape:
                        if np.array_equal(pattern_grid.data, inp.data):
                            pattern = DetectedPattern(
                                pattern_type=PatternType.TILING,
                                confidence=0.9,
                                parameters={'pattern': pattern_grid, 'count': count}
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_scaling_patterns(self,
                                inputs: List[Grid],
                                outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect scaling patterns"""
        patterns = []
        
        scale_factors = []
        for inp, out in zip(inputs, outputs):
            # Check for uniform scaling
            if out.height % inp.height == 0 and out.width % inp.width == 0:
                h_scale = out.height // inp.height
                w_scale = out.width // inp.width
                
                if h_scale == w_scale:
                    # Verify scaling
                    expected = inp.scale(h_scale)
                    if expected == out:
                        scale_factors.append(h_scale)
        
        if scale_factors and len(set(scale_factors)) == 1:
            pattern = DetectedPattern(
                pattern_type=PatternType.SCALING,
                confidence=1.0,
                parameters={'factor': scale_factors[0]}
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_symmetry_patterns(self,
                                 inputs: List[Grid],
                                 outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect symmetry-based patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            detector_out = PatternDetector(out)
            symmetries = detector_out.get_symmetries()
            
            # Check if output is symmetric version of input
            if symmetries['horizontal']:
                # Check if it's horizontally mirrored input
                mirrored = inp.mirror('horizontal')
                if mirrored == out:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.SYMMETRY,
                        confidence=1.0,
                        parameters={'type': 'horizontal_mirror'}
                    )
                    patterns.append(pattern)
                    break
            
            if symmetries['vertical']:
                # Check if it's vertically mirrored input
                mirrored = inp.mirror('vertical')
                if mirrored == out:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.SYMMETRY,
                        confidence=1.0,
                        parameters={'type': 'vertical_mirror'}
                    )
                    patterns.append(pattern)
                    break
            
            # Check for rotation
            for degrees in [90, 180, 270]:
                rotated = inp.rotate(degrees)
                if rotated == out:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.ROTATION,
                        confidence=1.0,
                        parameters={'degrees': degrees}
                    )
                    patterns.append(pattern)
                    break
        
        return patterns
    
    def _detect_color_patterns(self,
                              inputs: List[Grid],
                              outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect color mapping patterns"""
        patterns = []
        
        # Collect all color mappings
        all_mappings = []
        
        for inp, out in zip(inputs, outputs):
            if inp.shape != out.shape:
                continue
            
            # Build color mapping
            color_map = {}
            for color in inp.unique_colors:
                positions = inp.get_color_positions(color)
                if positions:
                    # Get colors at these positions in output
                    out_colors = set()
                    for y, x in positions:
                        if y < out.height and x < out.width:
                            out_colors.add(out.data[y, x])
                    
                    # If all map to same color
                    if len(out_colors) == 1:
                        color_map[color] = out_colors.pop()
            
            if color_map:
                all_mappings.append(color_map)
        
        # Check for consistent mapping
        if all_mappings and all(m == all_mappings[0] for m in all_mappings):
            pattern = DetectedPattern(
                pattern_type=PatternType.COLOR_MAPPING,
                confidence=1.0,
                parameters={'color_map': all_mappings[0]}
            )
            patterns.append(pattern)
        
        # Check for color inversion
        for inp, out in zip(inputs, outputs):
            if inp.shape == out.shape:
                # Check if colors are inverted/swapped
                inp_colors = sorted(inp.unique_colors - {0})
                out_colors = sorted(out.unique_colors - {0})
                
                if inp_colors == out_colors:
                    # Check for systematic inversion
                    is_inverted = True
                    for i, color in enumerate(inp_colors):
                        expected_color = out_colors[-(i+1)]
                        positions = inp.get_color_positions(color)
                        
                        for y, x in positions:
                            if out.data[y, x] != expected_color:
                                is_inverted = False
                                break
                        
                        if not is_inverted:
                            break
                    
                    if is_inverted:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.COLOR_MAPPING,
                            confidence=0.9,
                            parameters={'type': 'inversion'}
                        )
                        patterns.append(pattern)
                        break
        
        return patterns
    
    def _detect_repetition_patterns(self,
                                  inputs: List[Grid],
                                  outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect repetition patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            detector = PatternDetector(out)
            
            # Check for periodic patterns
            periodicity = detector.detect_periodicity()
            
            if periodicity['horizontal']:
                period = periodicity['horizontal']
                # Check if input matches the period
                if inp.width == period:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.REPETITION,
                        confidence=0.9,
                        parameters={'direction': 'horizontal', 'period': period}
                    )
                    patterns.append(pattern)
            
            if periodicity['vertical']:
                period = periodicity['vertical']
                # Check if input matches the period
                if inp.height == period:
                    pattern = DetectedPattern(
                        pattern_type=PatternType.REPETITION,
                        confidence=0.9,
                        parameters={'direction': 'vertical', 'period': period}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_mirroring_patterns(self,
                                  inputs: List[Grid],
                                  outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect mirroring patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            # Check horizontal mirroring
            if out.width == inp.width * 2:
                left_half = out.extract_subgrid(0, 0, out.height, inp.width)
                right_half = out.extract_subgrid(0, inp.width, out.height, inp.width)
                
                if left_half == inp and right_half == inp.flip_horizontal():
                    pattern = DetectedPattern(
                        pattern_type=PatternType.MIRRORING,
                        confidence=1.0,
                        parameters={'axis': 'horizontal'}
                    )
                    patterns.append(pattern)
                    break
            
            # Check vertical mirroring
            if out.height == inp.height * 2:
                top_half = out.extract_subgrid(0, 0, inp.height, out.width)
                bottom_half = out.extract_subgrid(inp.height, 0, inp.height, out.width)
                
                if top_half == inp and bottom_half == inp.flip_vertical():
                    pattern = DetectedPattern(
                        pattern_type=PatternType.MIRRORING,
                        confidence=1.0,
                        parameters={'axis': 'vertical'}
                    )
                    patterns.append(pattern)
                    break
        
        return patterns
    
    def _detect_extraction_patterns(self,
                                  inputs: List[Grid],
                                  outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect extraction patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            # Check if output is a subgrid of input
            if out.height <= inp.height and out.width <= inp.width:
                found = False
                for y in range(inp.height - out.height + 1):
                    for x in range(inp.width - out.width + 1):
                        subgrid = inp.extract_subgrid(y, x, out.height, out.width)
                        if subgrid == out:
                            pattern = DetectedPattern(
                                pattern_type=PatternType.EXTRACTION,
                                confidence=0.8,
                                parameters={'y': y, 'x': x, 
                                          'height': out.height, 'width': out.width}
                            )
                            patterns.append(pattern)
                            found = True
                            break
                    if found:
                        break
            
            # Check if output is largest/smallest component
            detector = PatternDetector(inp)
            components = detector.get_connected_components()
            if components:
                largest = max(components, key=lambda c: c['area'])
                smallest = min(components, key=lambda c: c['area'])
                
                # Check if output matches largest component
                if self._component_matches_grid(largest, out):
                    pattern = DetectedPattern(
                        pattern_type=PatternType.EXTRACTION,
                        confidence=0.9,
                        parameters={'type': 'largest_component'}
                    )
                    patterns.append(pattern)
                
                # Check if output matches smallest component
                elif self._component_matches_grid(smallest, out):
                    pattern = DetectedPattern(
                        pattern_type=PatternType.EXTRACTION,
                        confidence=0.9,
                        parameters={'type': 'smallest_component'}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_filling_patterns(self,
                                inputs: List[Grid],
                                outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect filling patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            if inp.shape != out.shape:
                continue
            
            # Check if background is filled
            if 0 in inp.unique_colors and 0 not in out.unique_colors:
                # Find what color replaced background
                bg_positions = inp.get_color_positions(0)
                if bg_positions:
                    fill_colors = set()
                    for y, x in bg_positions:
                        fill_colors.add(out.data[y, x])
                    
                    if len(fill_colors) == 1:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.FILLING,
                            confidence=0.9,
                            parameters={'fill_color': fill_colors.pop()}
                        )
                        patterns.append(pattern)
            
            # Check for flood fill patterns
            detector = PatternDetector(inp)
            components = detector.get_connected_components(include_background=True)
            
            for component in components:
                if component['color'] == 0:  # Background component
                    # Check if this region is filled in output
                    region_colors = set()
                    for y, x in component['pixels']:
                        region_colors.add(out.data[y, x])
                    
                    if len(region_colors) == 1 and 0 not in region_colors:
                        pattern = DetectedPattern(
                            pattern_type=PatternType.FILLING,
                            confidence=0.8,
                            parameters={'type': 'flood_fill', 
                                      'fill_color': region_colors.pop()}
                        )
                        patterns.append(pattern)
                        break
        
        return patterns
    
    def _detect_connectivity_patterns(self,
                                    inputs: List[Grid],
                                    outputs: List[Grid]) -> List[DetectedPattern]:
        """Detect connectivity-based patterns"""
        patterns = []
        
        for inp, out in zip(inputs, outputs):
            # Check if output shows connected components
            inp_detector = PatternDetector(inp)
            inp_components = inp_detector.get_connected_components()
            
            out_detector = PatternDetector(out)
            out_components = out_detector.get_connected_components()
            
            # Check if number of components matches
            if len(inp_components) == len(out_components):
                # Check if components are transformed consistently
                component_transforms = []
                
                for inp_comp in inp_components:
                    # Find matching component in output
                    for out_comp in out_components:
                        if self._components_match(inp_comp, out_comp):
                            component_transforms.append({
                                'input': inp_comp,
                                'output': out_comp
                            })
                            break
                
                if len(component_transforms) == len(inp_components):
                    pattern = DetectedPattern(
                        pattern_type=PatternType.CONNECTIVITY,
                        confidence=0.8,
                        parameters={'type': 'component_transform',
                                  'transforms': component_transforms}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _apply_pattern(self, 
                      grid: Grid,
                      pattern: DetectedPattern) -> Grid:
        """Apply detected pattern to grid"""
        
        if pattern.pattern_type == PatternType.TILING:
            params = pattern.parameters
            if 'rows' in params and 'cols' in params:
                return grid.tile(params['rows'], params['cols'])
            elif 'pattern' in params:
                # Tile the pattern
                return params['pattern'].tile(3, 3)  # Default tiling
        
        elif pattern.pattern_type == PatternType.SCALING:
            return grid.scale(pattern.parameters['factor'])
        
        elif pattern.pattern_type == PatternType.SYMMETRY:
            if pattern.parameters['type'] == 'horizontal_mirror':
                return grid.mirror('horizontal')
            elif pattern.parameters['type'] == 'vertical_mirror':
                return grid.mirror('vertical')
        
        elif pattern.pattern_type == PatternType.ROTATION:
            return grid.rotate(pattern.parameters['degrees'])
        
        elif pattern.pattern_type == PatternType.COLOR_MAPPING:
            if 'color_map' in pattern.parameters:
                return grid.map_colors(pattern.parameters['color_map'])
            elif pattern.parameters.get('type') == 'inversion':
                return self._invert_colors(grid)
        
        elif pattern.pattern_type == PatternType.REPETITION:
            direction = pattern.parameters['direction']
            if direction == 'horizontal':
                return grid.tile(1, 3)  # Repeat horizontally
            else:
                return grid.tile(3, 1)  # Repeat vertically
        
        elif pattern.pattern_type == PatternType.MIRRORING:
            return grid.mirror(pattern.parameters['axis'])
        
        elif pattern.pattern_type == PatternType.EXTRACTION:
            if pattern.parameters.get('type') == 'largest_component':
                return grid.extract_largest_component()
            elif pattern.parameters.get('type') == 'smallest_component':
                return grid.extract_smallest_component()
            else:
                # Extract subgrid
                return grid.extract_subgrid(
                    pattern.parameters['y'],
                    pattern.parameters['x'],
                    pattern.parameters['height'],
                    pattern.parameters['width']
                )
        
        elif pattern.pattern_type == PatternType.FILLING:
            if pattern.parameters.get('type') == 'flood_fill':
                # Flood fill background
                result = grid.copy()
                result.data[result.data == 0] = pattern.parameters['fill_color']
                return result
            else:
                # Fill background
                result = grid.copy()
                result.data[result.data == 0] = pattern.parameters['fill_color']
                return result
        
        elif pattern.pattern_type == PatternType.CONNECTIVITY:
            # Apply component transformations
            return self._apply_component_transforms(grid, pattern.parameters)
        
        return grid
    
    def _validate_pattern(self,
                        pattern: DetectedPattern,
                        inputs: List[Grid],
                        outputs: List[Grid]) -> bool:
        """Validate pattern on training examples"""
        
        correct = 0
        total = len(inputs)
        
        for inp, expected_out in zip(inputs, outputs):
            try:
                result = self._apply_pattern(inp, pattern)
                if result == expected_out:
                    correct += 1
            except Exception:
                continue
        
        # Pattern should work on most examples
        return correct / total >= 0.8
    
    def _invert_colors(self, grid: Grid) -> Grid:
        """Invert non-background colors"""
        colors = sorted(grid.unique_colors - {0})
        if not colors:
            return grid
        
        color_map = {}
        for i, color in enumerate(colors):
            color_map[color] = colors[-(i+1)]
        
        return grid.map_colors(color_map)
    
    def _component_matches_grid(self, 
                               component: Dict[str, Any],
                               grid: Grid) -> bool:
        """Check if a component matches a grid"""
        # Get bounding box of component
        pixels = component.get('positions', component.get('pixels', []))
        if not pixels:
            return False
        
        ys = [p[0] for p in pixels]
        xs = [p[1] for p in pixels]
        
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        
        # Check if grid size matches bounding box
        if grid.height != (max_y - min_y + 1) or grid.width != (max_x - min_x + 1):
            return False
        
        # Check if pixels match
        for y, x in pixels:
            grid_y = y - min_y
            grid_x = x - min_x
            if grid.data[grid_y, grid_x] != component['color']:
                return False
        
        return True
    
    def _components_match(self,
                        comp1: Dict[str, Any],
                        comp2: Dict[str, Any]) -> bool:
        """Check if two components match (same shape/structure)"""
        pixels1 = comp1.get('positions', comp1.get('pixels', []))
        pixels2 = comp2.get('positions', comp2.get('pixels', []))
        
        if len(pixels1) != len(pixels2):
            return False
        
        # Normalize positions
        
        if not pixels1 or not pixels2:
            return False
        
        # Get relative positions
        min_y1 = min(p[0] for p in pixels1)
        min_x1 = min(p[1] for p in pixels1)
        
        min_y2 = min(p[0] for p in pixels2)
        min_x2 = min(p[1] for p in pixels2)
        
        norm_pixels1 = set((y - min_y1, x - min_x1) for y, x in pixels1)
        norm_pixels2 = set((y - min_y2, x - min_x2) for y, x in pixels2)
        
        return norm_pixels1 == norm_pixels2
    
    def _apply_component_transforms(self,
                                   grid: Grid,
                                   parameters: Dict) -> Grid:
        """Apply transformations to components"""
        # This would apply component-specific transformations
        # For now, return the grid as-is
        return grid