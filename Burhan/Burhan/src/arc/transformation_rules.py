"""
ARC Transformation Rules Module
Defines and applies transformation rules for ARC tasks
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from .grid_operations import Grid
from .pattern_detector import PatternDetector


class TransformationType(Enum):
    """Types of transformations"""
    # Geometric
    ROTATE = "rotate"
    FLIP = "flip"
    TRANSPOSE = "transpose"
    SCALE = "scale"
    TILE = "tile"
    MIRROR = "mirror"
    
    # Color
    COLOR_MAP = "color_map"
    COLOR_REPLACE = "color_replace"
    COLOR_FILTER = "color_filter"
    
    # Pattern
    PATTERN_REPEAT = "pattern_repeat"
    PATTERN_REPLACE = "pattern_replace"
    OVERLAY = "overlay"
    MASK = "mask"
    
    # Structural
    CROP = "crop"
    PAD = "pad"
    RESIZE = "resize"
    EXTRACT = "extract"
    
    # Complex
    COMPOSITE = "composite"
    CONDITIONAL = "conditional"
    CUSTOM = "custom"


@dataclass
class TransformationRule:
    """Represents a transformation rule"""
    name: str
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    description: str = ""
    
    def apply(self, grid: Grid) -> Grid:
        """Apply the transformation to a grid"""
        if self.transformation_type == TransformationType.ROTATE:
            return grid.rotate(self.parameters['degrees'])
        
        elif self.transformation_type == TransformationType.FLIP:
            if self.parameters['direction'] == 'horizontal':
                return grid.flip_horizontal()
            else:
                return grid.flip_vertical()
        
        elif self.transformation_type == TransformationType.TRANSPOSE:
            return grid.transpose()
        
        elif self.transformation_type == TransformationType.SCALE:
            return grid.scale(self.parameters['factor'])
        
        elif self.transformation_type == TransformationType.TILE:
            return grid.tile(self.parameters['rows'], self.parameters['cols'])
        
        elif self.transformation_type == TransformationType.MIRROR:
            return grid.mirror(self.parameters.get('axis', 'horizontal'))
        
        elif self.transformation_type == TransformationType.COLOR_MAP:
            return grid.map_colors(self.parameters['color_map'])
        
        elif self.transformation_type == TransformationType.COLOR_REPLACE:
            return grid.replace_color(
                self.parameters['old_color'], 
                self.parameters['new_color']
            )
        
        elif self.transformation_type == TransformationType.COLOR_FILTER:
            return grid.filter_color(
                self.parameters['color'],
                self.parameters.get('background', 0)
            )
        
        elif self.transformation_type == TransformationType.CROP:
            return grid.crop_to_content(self.parameters.get('background', 0))
        
        elif self.transformation_type == TransformationType.PAD:
            return grid.pad(
                self.parameters.get('top', 0),
                self.parameters.get('bottom', 0),
                self.parameters.get('left', 0),
                self.parameters.get('right', 0),
                self.parameters.get('fill_value', 0)
            )
        
        elif self.transformation_type == TransformationType.RESIZE:
            return grid.resize(
                self.parameters['height'],
                self.parameters['width'],
                self.parameters.get('fill_value', 0)
            )
        
        elif self.transformation_type == TransformationType.CUSTOM:
            # Custom transformation using a provided function
            func = self.parameters.get('function')
            if func and callable(func):
                return func(grid)
        
        else:
            raise NotImplementedError(f"Transformation {self.transformation_type} not implemented")
    
    def validate(self, grid: Grid) -> bool:
        """Check if this rule can be applied to the given grid"""
        if self.transformation_type == TransformationType.ROTATE:
            return grid.height == grid.width or self.parameters['degrees'] == 180
        
        elif self.transformation_type == TransformationType.TRANSPOSE:
            return True  # Always valid
        
        elif self.transformation_type == TransformationType.SCALE:
            factor = self.parameters['factor']
            return factor > 0 and isinstance(factor, int)
        
        return True  # Default to valid


class RuleChain:
    """Chain multiple transformation rules"""
    
    def __init__(self, rules: List[TransformationRule]):
        self.rules = rules
    
    def apply(self, grid: Grid) -> Grid:
        """Apply all rules in sequence"""
        result = grid.copy()
        for rule in self.rules:
            if rule.validate(result):
                result = rule.apply(result)
        return result
    
    def add_rule(self, rule: TransformationRule):
        """Add a rule to the chain"""
        self.rules.append(rule)
    
    def remove_rule(self, index: int):
        """Remove a rule from the chain"""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)


class TransformationInference:
    """Infer transformation rules from examples"""
    
    @staticmethod
    def infer_simple_transformation(input_grid: Grid, output_grid: Grid) -> Optional[TransformationRule]:
        """Try to infer a simple transformation between input and output"""
        
        # Check for rotation
        for degrees in [90, 180, 270]:
            if input_grid.rotate(degrees) == output_grid:
                return TransformationRule(
                    name="rotation",
                    transformation_type=TransformationType.ROTATE,
                    parameters={'degrees': degrees},
                    description=f"Rotate {degrees} degrees"
                )
        
        # Check for flips
        if input_grid.flip_horizontal() == output_grid:
            return TransformationRule(
                name="horizontal_flip",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'horizontal'},
                description="Flip horizontally"
            )
        
        if input_grid.flip_vertical() == output_grid:
            return TransformationRule(
                name="vertical_flip",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'vertical'},
                description="Flip vertically"
            )
        
        # Check for transpose
        if input_grid.transpose() == output_grid:
            return TransformationRule(
                name="transpose",
                transformation_type=TransformationType.TRANSPOSE,
                parameters={},
                description="Transpose matrix"
            )
        
        # Check for scaling
        if output_grid.height % input_grid.height == 0 and output_grid.width % input_grid.width == 0:
            if output_grid.height // input_grid.height == output_grid.width // input_grid.width:
                factor = output_grid.height // input_grid.height
                if input_grid.scale(factor) == output_grid:
                    return TransformationRule(
                        name="scale",
                        transformation_type=TransformationType.SCALE,
                        parameters={'factor': factor},
                        description=f"Scale by factor {factor}"
                    )
        
        # Check for tiling
        if output_grid.height >= input_grid.height and output_grid.width >= input_grid.width:
            if output_grid.height % input_grid.height == 0 and output_grid.width % input_grid.width == 0:
                rows = output_grid.height // input_grid.height
                cols = output_grid.width // input_grid.width
                if input_grid.tile(rows, cols) == output_grid:
                    return TransformationRule(
                        name="tile",
                        transformation_type=TransformationType.TILE,
                        parameters={'rows': rows, 'cols': cols},
                        description=f"Tile {rows}x{cols}"
                    )
        
        # Check for color mapping
        input_colors = input_grid.unique_colors
        output_colors = output_grid.unique_colors
        
        if input_grid.shape == output_grid.shape:
            # Try to find color mapping
            color_map = {}
            is_color_map = True
            
            for y in range(input_grid.height):
                for x in range(input_grid.width):
                    input_color = input_grid.data[y, x]
                    output_color = output_grid.data[y, x]
                    
                    if input_color in color_map:
                        if color_map[input_color] != output_color:
                            is_color_map = False
                            break
                    else:
                        color_map[input_color] = output_color
                
                if not is_color_map:
                    break
            
            if is_color_map and color_map:
                return TransformationRule(
                    name="color_map",
                    transformation_type=TransformationType.COLOR_MAP,
                    parameters={'color_map': color_map},
                    description=f"Map colors: {color_map}"
                )
        
        return None
    
    @staticmethod
    def infer_from_examples(examples: List[Tuple[Grid, Grid]]) -> List[TransformationRule]:
        """Infer transformation rules from multiple examples"""
        rules = []
        
        # Try to find consistent simple transformations
        for input_grid, output_grid in examples:
            rule = TransformationInference.infer_simple_transformation(input_grid, output_grid)
            if rule:
                # Check if this rule works for all examples
                is_consistent = True
                for test_input, test_output in examples:
                    if rule.apply(test_input) != test_output:
                        is_consistent = False
                        break
                
                if is_consistent:
                    rules.append(rule)
                    return rules  # Found a single consistent rule
        
        # If no single rule works, try to find composite rules
        rules = TransformationInference._infer_composite_rules(examples)
        
        return rules
    
    @staticmethod
    def _infer_composite_rules(examples: List[Tuple[Grid, Grid]]) -> List[TransformationRule]:
        """Try to infer composite transformation rules"""
        rules = []
        
        # Check for common patterns across examples
        for input_grid, output_grid in examples:
            detector_in = PatternDetector(input_grid)
            detector_out = PatternDetector(output_grid)
            
            comparison = detector_in.compare_with(output_grid)
            
            # Build composite rule based on comparison
            if comparison['is_scaled'] and comparison['scale_factor']:
                rules.append(TransformationRule(
                    name="scale",
                    transformation_type=TransformationType.SCALE,
                    parameters={'factor': comparison['scale_factor']},
                    description=f"Scale by {comparison['scale_factor']}"
                ))
            
            if comparison['is_rotated'] and comparison['rotation_angle']:
                rules.append(TransformationRule(
                    name="rotate",
                    transformation_type=TransformationType.ROTATE,
                    parameters={'degrees': comparison['rotation_angle']},
                    description=f"Rotate {comparison['rotation_angle']} degrees"
                ))
            
            if comparison['is_flipped'] and comparison['flip_type']:
                rules.append(TransformationRule(
                    name="flip",
                    transformation_type=TransformationType.FLIP,
                    parameters={'direction': comparison['flip_type']},
                    description=f"Flip {comparison['flip_type']}"
                ))
            
            # Only return rules if they're consistent across all examples
            if rules:
                chain = RuleChain(rules)
                is_consistent = all(
                    chain.apply(inp) == out 
                    for inp, out in examples
                )
                if is_consistent:
                    return rules
        
        return []


class RuleLibrary:
    """Library of common transformation rules"""
    
    @staticmethod
    def get_rotation_rules() -> List[TransformationRule]:
        """Get all rotation rules"""
        return [
            TransformationRule(
                name=f"rotate_{degrees}",
                transformation_type=TransformationType.ROTATE,
                parameters={'degrees': degrees},
                description=f"Rotate {degrees} degrees clockwise"
            )
            for degrees in [90, 180, 270]
        ]
    
    @staticmethod
    def get_flip_rules() -> List[TransformationRule]:
        """Get all flip rules"""
        return [
            TransformationRule(
                name="flip_horizontal",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'horizontal'},
                description="Flip horizontally (left-right)"
            ),
            TransformationRule(
                name="flip_vertical",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'vertical'},
                description="Flip vertically (top-bottom)"
            )
        ]
    
    @staticmethod
    def get_scale_rule(factor: int) -> TransformationRule:
        """Get a scale rule with specified factor"""
        return TransformationRule(
            name=f"scale_{factor}x",
            transformation_type=TransformationType.SCALE,
            parameters={'factor': factor},
            description=f"Scale by factor of {factor}"
        )
    
    @staticmethod
    def get_tile_rule(rows: int, cols: int) -> TransformationRule:
        """Get a tile rule with specified dimensions"""
        return TransformationRule(
            name=f"tile_{rows}x{cols}",
            transformation_type=TransformationType.TILE,
            parameters={'rows': rows, 'cols': cols},
            description=f"Tile in a {rows}x{cols} pattern"
        )
    
    @staticmethod
    def get_color_swap_rule(color1: int, color2: int) -> RuleChain:
        """Get a rule chain that swaps two colors"""
        temp_color = 10  # Temporary color outside normal range
        
        return RuleChain([
            TransformationRule(
                name=f"swap_{color1}_{color2}_step1",
                transformation_type=TransformationType.COLOR_REPLACE,
                parameters={'old_color': color1, 'new_color': temp_color},
                description=f"Replace {color1} with temp"
            ),
            TransformationRule(
                name=f"swap_{color1}_{color2}_step2",
                transformation_type=TransformationType.COLOR_REPLACE,
                parameters={'old_color': color2, 'new_color': color1},
                description=f"Replace {color2} with {color1}"
            ),
            TransformationRule(
                name=f"swap_{color1}_{color2}_step3",
                transformation_type=TransformationType.COLOR_REPLACE,
                parameters={'old_color': temp_color, 'new_color': color2},
                description=f"Replace temp with {color2}"
            )
        ])
    
    @staticmethod
    def get_common_rules() -> Dict[str, TransformationRule]:
        """Get a dictionary of common transformation rules"""
        rules = {}
        
        # Add rotation rules
        for rule in RuleLibrary.get_rotation_rules():
            rules[rule.name] = rule
        
        # Add flip rules
        for rule in RuleLibrary.get_flip_rules():
            rules[rule.name] = rule
        
        # Add transpose
        rules['transpose'] = TransformationRule(
            name="transpose",
            transformation_type=TransformationType.TRANSPOSE,
            parameters={},
            description="Transpose (swap rows and columns)"
        )
        
        # Add common scale rules
        for factor in [2, 3, 4]:
            rule = RuleLibrary.get_scale_rule(factor)
            rules[rule.name] = rule
        
        # Add crop rule
        rules['crop'] = TransformationRule(
            name="crop",
            transformation_type=TransformationType.CROP,
            parameters={'background': 0},
            description="Crop to content (remove background borders)"
        )
        
        return rules