from __future__ import annotations
#!/usr/bin/env python3
"""
ADVANCED PATTERN ENGINE - Universal ARC Pattern Recognition
==========================================================
Extracts and applies complex patterns from any ARC task
Handles multi-level transformations and compositions
"""

import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, Counter
import itertools
from functools import lru_cache

@dataclass
class ComplexPattern:
    """Complex pattern with multi-level transformations"""
    name: str
    transformations: List[Callable]
    confidence: float
    complexity_level: int
    success_rate: float = 0.0
    applicable_contexts: List[str] = None

class AdvancedPatternEngine:
    """Advanced pattern recognition engine"""
    
    def __init__(self):
        self.pattern_library = self._build_comprehensive_library()
        self.learned_patterns = {}
        self.pattern_success_rates = defaultdict(list)
        self.context_patterns = defaultdict(list)
        
    def _build_comprehensive_library(self) -> Dict[str, ComplexPattern]:
        """Build comprehensive pattern library"""
        patterns = {}
        
        # Level 1: Basic transformations
        patterns.update(self._create_basic_patterns())
        
        # Level 2: Composite transformations
        patterns.update(self._create_composite_patterns())
        
        # Level 3: Context-sensitive patterns
        patterns.update(self._create_context_patterns())
        
        # Level 4: Adaptive patterns
        patterns.update(self._create_adaptive_patterns())
        
        return patterns
    
    def _create_basic_patterns(self) -> Dict[str, ComplexPattern]:
        """Create basic transformation patterns"""
        patterns = {}
        
        # Spatial transformations
        patterns['rotate_90'] = ComplexPattern(
            name='rotate_90',
            transformations=[lambda g: np.rot90(g, 1)],
            confidence=0.9,
            complexity_level=1
        )
        
        patterns['flip_lr'] = ComplexPattern(
            name='flip_lr',
            transformations=[np.fliplr],
            confidence=0.9,
            complexity_level=1
        )
        
        patterns['flip_ud'] = ComplexPattern(
            name='flip_ud',
            transformations=[np.flipud],
            confidence=0.9,
            complexity_level=1
        )
        
        # Size transformations
        patterns['tile_2x2'] = ComplexPattern(
            name='tile_2x2',
            transformations=[lambda g: np.tile(g, (2, 2))],
            confidence=0.8,
            complexity_level=2
        )
        
        patterns['tile_3x3'] = ComplexPattern(
            name='tile_3x3',
            transformations=[lambda g: np.tile(g, (3, 3))],
            confidence=0.8,
            complexity_level=2
        )
        
        # Color transformations
        patterns['color_shift'] = ComplexPattern(
            name='color_shift',
            transformations=[lambda g: (g + 1) % 10],
            confidence=0.7,
            complexity_level=2
        )
        
        patterns['color_invert'] = ComplexPattern(
            name='color_invert',
            transformations=[lambda g: 9 - g],
            confidence=0.7,
            complexity_level=2
        )
        
        return patterns
    
    def _create_composite_patterns(self) -> Dict[str, ComplexPattern]:
        """Create composite transformation patterns"""
        patterns = {}
        
        # Rotate then flip
        patterns['rotate_flip'] = ComplexPattern(
            name='rotate_flip',
            transformations=[
                lambda g: np.rot90(g, 1),
                np.fliplr
            ],
            confidence=0.8,
            complexity_level=3
        )
        
        # Tile then rotate
        patterns['tile_rotate'] = ComplexPattern(
            name='tile_rotate',
            transformations=[
                lambda g: np.tile(g, (2, 2)),
                lambda g: np.rot90(g, 1)
            ],
            confidence=0.7,
            complexity_level=3
        )
        
        # Color shift then spatial transform
        patterns['color_spatial'] = ComplexPattern(
            name='color_spatial',
            transformations=[
                lambda g: (g + 1) % 10,
                lambda g: np.rot90(g, 2)
            ],
            confidence=0.6,
            complexity_level=3
        )
        
        return patterns
    
    def _create_context_patterns(self) -> Dict[str, ComplexPattern]:
        """Create context-sensitive patterns"""
        patterns = {}
        
        # Conditional transformations based on grid properties
        patterns['conditional_rotate'] = ComplexPattern(
            name='conditional_rotate',
            transformations=[self._conditional_rotate],
            confidence=0.8,
            complexity_level=4,
            applicable_contexts=['square_grid', 'symmetric']
        )
        
        patterns['size_dependent'] = ComplexPattern(
            name='size_dependent',
            transformations=[self._size_dependent_transform],
            confidence=0.7,
            complexity_level=4,
            applicable_contexts=['variable_size']
        )
        
        return patterns
    
    def _create_adaptive_patterns(self) -> Dict[str, ComplexPattern]:
        """Create adaptive patterns that learn"""
        patterns = {}
        
        patterns['learned_mapping'] = ComplexPattern(
            name='learned_mapping',
            transformations=[self._apply_learned_mapping],
            confidence=0.9,
            complexity_level=5,
            applicable_contexts=['any']
        )
        
        return patterns
    
    def extract_pattern_from_examples(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Extract the best pattern from training examples"""
        candidate_patterns = []
        
        # Test all patterns in library
        for pattern_name, pattern in self.pattern_library.items():
            confidence = self._test_pattern_on_examples(pattern, train_pairs)
            if confidence > 0.5:
                pattern.confidence = confidence
                candidate_patterns.append(pattern)
        
        # Generate new patterns if needed
        if not candidate_patterns:
            candidate_patterns = self._generate_new_patterns(train_pairs)
        
        # Sort by confidence
        candidate_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return candidate_patterns[:5]
    
    def _test_pattern_on_examples(self, pattern: ComplexPattern, train_pairs: List[Dict]) -> float:
        """Test pattern on all training examples"""
        matches = 0
        total = len(train_pairs)
        
        for pair in train_pairs:
            try:
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                
                # Apply all transformations in sequence
                result = input_grid.copy()
                for transform in pattern.transformations:
                    result = transform(result)
                
                if result.shape == output_grid.shape and np.array_equal(result, output_grid):
                    matches += 1
            except:
                continue
        
        return matches / max(total, 1)
    
    def _generate_new_patterns(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Generate new patterns when existing ones fail"""
        new_patterns = []
        
        # Analyze the transformation type
        transform_type = self._analyze_transformation_type(train_pairs)
        
        if transform_type == 'size_change':
            new_patterns.extend(self._generate_size_patterns(train_pairs))
        elif transform_type == 'color_change':
            new_patterns.extend(self._generate_color_patterns(train_pairs))
        elif transform_type == 'spatial_change':
            new_patterns.extend(self._generate_spatial_patterns(train_pairs))
        elif transform_type == 'complex_change':
            new_patterns.extend(self._generate_complex_patterns(train_pairs))
        
        return new_patterns
    
    def _analyze_transformation_type(self, train_pairs: List[Dict]) -> str:
        """Analyze what type of transformation is happening"""
        if not train_pairs:
            return 'unknown'
        
        pair = train_pairs[0]
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        # Check size change
        if input_grid.shape != output_grid.shape:
            return 'size_change'
        
        # Check color change
        if not np.array_equal(np.unique(input_grid), np.unique(output_grid)):
            return 'color_change'
        
        # Check spatial change
        if not np.array_equal(input_grid, output_grid):
            return 'spatial_change'
        
        return 'complex_change'
    
    def _generate_size_patterns(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Generate patterns for size changes"""
        patterns = []
        
        # Analyze size ratios
        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            h_ratio = output_grid.shape[0] / input_grid.shape[0]
            w_ratio = output_grid.shape[1] / input_grid.shape[1]
            
            if h_ratio == w_ratio and h_ratio.is_integer():
                factor = int(h_ratio)
                patterns.append(ComplexPattern(
                    name=f'tile_{factor}x{factor}',
                    transformations=[lambda g, f=factor: np.tile(g, (f, f))],
                    confidence=0.8,
                    complexity_level=2
                ))
        
        return patterns
    
    def _generate_color_patterns(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Generate patterns for color changes"""
        patterns = []
        
        # Extract color mappings
        color_mappings = []
        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            if input_grid.shape == output_grid.shape:
                mapping = self._extract_color_mapping(input_grid, output_grid)
                if mapping:
                    color_mappings.append(mapping)
        
        # Create pattern from consistent mapping
        if color_mappings and all(m == color_mappings[0] for m in color_mappings):
            mapping = color_mappings[0]
            
            def apply_mapping(grid):
                result = grid.copy()
                for old_color, new_color in mapping.items():
                    result[grid == old_color] = new_color
                return result
            
            patterns.append(ComplexPattern(
                name='color_mapping',
                transformations=[apply_mapping],
                confidence=0.9,
                complexity_level=2
            ))
        
        return patterns
    
    def _generate_spatial_patterns(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Generate patterns for spatial changes"""
        patterns = []
        
        # Try different spatial transformations
        spatial_transforms = [
            ('rotate_90', lambda g: np.rot90(g, 1)),
            ('rotate_180', lambda g: np.rot90(g, 2)),
            ('rotate_270', lambda g: np.rot90(g, 3)),
            ('flip_lr', np.fliplr),
            ('flip_ud', np.flipud),
            ('transpose', np.transpose)
        ]
        
        for name, transform in spatial_transforms:
            confidence = self._test_single_transform(train_pairs, transform)
            if confidence > 0.7:
                patterns.append(ComplexPattern(
                    name=name,
                    transformations=[transform],
                    confidence=confidence,
                    complexity_level=1
                ))
        
        return patterns
    
    def _generate_complex_patterns(self, train_pairs: List[Dict]) -> List[ComplexPattern]:
        """Generate complex multi-step patterns"""
        patterns = []
        
        # Try combinations of transformations
        basic_transforms = [
            lambda g: np.rot90(g, 1),
            lambda g: np.rot90(g, 2),
            np.fliplr,
            np.flipud,
            lambda g: (g + 1) % 10
        ]
        
        # Try pairs of transformations
        for t1, t2 in itertools.combinations(basic_transforms, 2):
            def combined_transform(grid, transform1=t1, transform2=t2):
                intermediate = transform1(grid)
                return transform2(intermediate)
            
            confidence = self._test_single_transform(train_pairs, combined_transform)
            if confidence > 0.6:
                patterns.append(ComplexPattern(
                    name='complex_combined',
                    transformations=[t1, t2],
                    confidence=confidence,
                    complexity_level=3
                ))
        
        return patterns
    
    def _test_single_transform(self, train_pairs: List[Dict], transform: Callable) -> float:
        """Test a single transformation on training pairs"""
        matches = 0
        total = len(train_pairs)
        
        for pair in train_pairs:
            try:
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                result = transform(input_grid)
                
                if result.shape == output_grid.shape and np.array_equal(result, output_grid):
                    matches += 1
            except:
                continue
        
        return matches / max(total, 1)
    
    def _extract_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """Extract color mapping between input and output"""
        if input_grid.shape != output_grid.shape:
            return {}
        
        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                input_color = input_grid[i, j]
                output_color = output_grid[i, j]
                
                if input_color in mapping:
                    if mapping[input_color] != output_color:
                        return {}  # Inconsistent mapping
                else:
                    mapping[input_color] = output_color
        
        return mapping
    
    # Context-sensitive transformation methods
    def _conditional_rotate(self, grid: np.ndarray) -> np.ndarray:
        """Rotate based on grid properties"""
        if grid.shape[0] == grid.shape[1]:  # Square grid
            return np.rot90(grid, 1)
        return grid
    
    def _size_dependent_transform(self, grid: np.ndarray) -> np.ndarray:
        """Transform based on grid size"""
        if grid.shape[0] <= 3 and grid.shape[1] <= 3:
            return np.tile(grid, (2, 2))
        return np.rot90(grid, 1)
    
    def _apply_learned_mapping(self, grid: np.ndarray) -> np.ndarray:
        """Apply learned transformation mapping"""
        # This would use patterns learned from previous tasks
        return grid  # Placeholder for now

# Test the engine
if __name__ == "__main__":
    engine = AdvancedPatternEngine()
    print(f"🧠 Advanced Pattern Engine initialized with {len(engine.pattern_library)} patterns")
    print("🎯 Ready to extract complex patterns from any ARC task!")
