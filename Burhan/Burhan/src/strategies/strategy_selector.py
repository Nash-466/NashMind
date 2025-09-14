"""
Strategy Selection Module for ARC Prize 2025
Analyzes tasks and selects appropriate solving strategies
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..arc.grid_operations import Grid
from ..arc.pattern_detector import PatternDetector


class TaskCategory(Enum):
    """Categories of ARC tasks"""
    GEOMETRIC = "geometric"
    COLOR_MANIPULATION = "color_manipulation"
    PATTERN_COMPLETION = "pattern_completion"
    COUNTING = "counting"
    SYMMETRY = "symmetry"
    OBJECT_MANIPULATION = "object_manipulation"
    LOGICAL = "logical"
    SPATIAL_REASONING = "spatial_reasoning"
    UNKNOWN = "unknown"


class ComplexityLevel(Enum):
    """Task complexity levels"""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


@dataclass
class TaskFeatures:
    """Features extracted from ARC task"""
    # Size features
    input_sizes: List[Tuple[int, int]]
    output_sizes: List[Tuple[int, int]]
    size_change_ratio: float
    consistent_size_change: bool
    
    # Color features
    input_color_counts: List[Dict[int, int]]
    output_color_counts: List[Dict[int, int]]
    unique_colors_input: Set[int]
    unique_colors_output: Set[int]
    color_mapping_exists: bool
    color_reduction: bool
    
    # Pattern features
    has_repetition: bool
    has_symmetry: bool
    symmetry_types: List[str]
    has_periodicity: bool
    pattern_count: int
    
    # Structural features
    has_connected_components: bool
    component_count: int
    has_grid_structure: bool
    has_boundaries: bool
    sparsity: float
    
    # Transformation features
    is_rotation: bool
    is_reflection: bool
    is_scaling: bool
    is_translation: bool
    is_color_swap: bool
    
    # Complexity metrics
    pixel_accuracy_variance: float
    transformation_consistency: float
    rule_complexity: int
    
    # Additional metadata
    num_examples: int
    avg_input_size: float
    avg_output_size: float


@dataclass
class StrategyRecommendation:
    """Recommended solving strategy"""
    primary_strategy: str
    alternative_strategies: List[str]
    confidence: float
    reasoning: str
    expected_difficulty: ComplexityLevel
    suggested_timeout: float


class FeatureExtractor:
    """Extracts features from ARC tasks"""
    
    def extract_features(self,
                        train_inputs: List[np.ndarray],
                        train_outputs: List[np.ndarray]) -> TaskFeatures:
        """Extract comprehensive features from training examples"""
        
        # Convert to Grid objects
        input_grids = [Grid(inp) for inp in train_inputs]
        output_grids = [Grid(out) for out in train_outputs]
        
        # Size features
        input_sizes = [g.shape for g in input_grids]
        output_sizes = [g.shape for g in output_grids]
        size_changes = self._analyze_size_changes(input_sizes, output_sizes)
        
        # Color features
        color_features = self._analyze_colors(input_grids, output_grids)
        
        # Pattern features
        pattern_features = self._analyze_patterns(input_grids, output_grids)
        
        # Structural features
        structural_features = self._analyze_structure(input_grids, output_grids)
        
        # Transformation features
        transformation_features = self._analyze_transformations(input_grids, output_grids)
        
        # Complexity metrics
        complexity_metrics = self._analyze_complexity(input_grids, output_grids)
        
        return TaskFeatures(
            # Size features
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            size_change_ratio=size_changes['ratio'],
            consistent_size_change=size_changes['consistent'],
            
            # Color features
            input_color_counts=color_features['input_counts'],
            output_color_counts=color_features['output_counts'],
            unique_colors_input=color_features['unique_input'],
            unique_colors_output=color_features['unique_output'],
            color_mapping_exists=color_features['mapping_exists'],
            color_reduction=color_features['reduction'],
            
            # Pattern features
            has_repetition=pattern_features['repetition'],
            has_symmetry=pattern_features['symmetry'],
            symmetry_types=pattern_features['symmetry_types'],
            has_periodicity=pattern_features['periodicity'],
            pattern_count=pattern_features['pattern_count'],
            
            # Structural features
            has_connected_components=structural_features['has_components'],
            component_count=structural_features['component_count'],
            has_grid_structure=structural_features['grid_structure'],
            has_boundaries=structural_features['boundaries'],
            sparsity=structural_features['sparsity'],
            
            # Transformation features
            is_rotation=transformation_features['rotation'],
            is_reflection=transformation_features['reflection'],
            is_scaling=transformation_features['scaling'],
            is_translation=transformation_features['translation'],
            is_color_swap=transformation_features['color_swap'],
            
            # Complexity metrics
            pixel_accuracy_variance=complexity_metrics['accuracy_variance'],
            transformation_consistency=complexity_metrics['consistency'],
            rule_complexity=complexity_metrics['rule_complexity'],
            
            # Metadata
            num_examples=len(train_inputs),
            avg_input_size=np.mean([g.data.size for g in input_grids]),
            avg_output_size=np.mean([g.data.size for g in output_grids])
        )
    
    def _analyze_size_changes(self,
                            input_sizes: List[Tuple[int, int]],
                            output_sizes: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze size changes between input and output"""
        
        ratios = []
        for (ih, iw), (oh, ow) in zip(input_sizes, output_sizes):
            if ih > 0 and iw > 0:
                h_ratio = oh / ih
                w_ratio = ow / iw
                ratios.append((h_ratio, w_ratio))
        
        # Check consistency
        consistent = len(set(ratios)) == 1 if ratios else False
        
        # Average ratio
        avg_ratio = 1.0
        if ratios:
            avg_h = np.mean([r[0] for r in ratios])
            avg_w = np.mean([r[1] for r in ratios])
            avg_ratio = (avg_h + avg_w) / 2
        
        return {
            'ratio': avg_ratio,
            'consistent': consistent,
            'ratios': ratios
        }
    
    def _analyze_colors(self,
                       input_grids: List[Grid],
                       output_grids: List[Grid]) -> Dict[str, Any]:
        """Analyze color usage and transformations"""
        
        input_counts = [g.count_colors() for g in input_grids]
        output_counts = [g.count_colors() for g in output_grids]
        
        unique_input = set()
        unique_output = set()
        
        for g in input_grids:
            unique_input.update(g.unique_colors)
        for g in output_grids:
            unique_output.update(g.unique_colors)
        
        # Check for color mapping
        mapping_exists = self._check_color_mapping(input_grids, output_grids)
        
        # Check for color reduction
        reduction = len(unique_output) < len(unique_input)
        
        return {
            'input_counts': input_counts,
            'output_counts': output_counts,
            'unique_input': unique_input,
            'unique_output': unique_output,
            'mapping_exists': mapping_exists,
            'reduction': reduction
        }
    
    def _check_color_mapping(self,
                           input_grids: List[Grid],
                           output_grids: List[Grid]) -> bool:
        """Check if there's a consistent color mapping"""
        
        if len(input_grids) != len(output_grids):
            return False
        
        mappings = []
        for inp, out in zip(input_grids, output_grids):
            if inp.shape != out.shape:
                continue
            
            mapping = {}
            for color in inp.unique_colors:
                positions = inp.get_color_positions(color)
                if positions:
                    out_colors = set()
                    for y, x in positions:
                        if y < out.height and x < out.width:
                            out_colors.add(out.data[y, x])
                    
                    if len(out_colors) == 1:
                        mapping[color] = out_colors.pop()
            
            if mapping:
                mappings.append(mapping)
        
        # Check consistency
        if mappings and all(m == mappings[0] for m in mappings):
            return True
        
        return False
    
    def _analyze_patterns(self,
                        input_grids: List[Grid],
                        output_grids: List[Grid]) -> Dict[str, Any]:
        """Analyze pattern-related features"""
        
        has_repetition = False
        has_symmetry = False
        symmetry_types = set()
        has_periodicity = False
        total_patterns = 0
        
        for grid in output_grids:
            detector = PatternDetector(grid)
            
            # Check repetition
            patterns = detector.find_repeating_patterns()
            if patterns:
                has_repetition = True
                total_patterns += len(patterns)
            
            # Check symmetry
            symmetries = detector.get_symmetries()
            if any(symmetries.values()):
                has_symmetry = True
                for sym_type, present in symmetries.items():
                    if present:
                        symmetry_types.add(sym_type)
            
            # Check periodicity
            periodicity = detector.detect_periodicity()
            if periodicity['horizontal'] or periodicity['vertical']:
                has_periodicity = True
        
        return {
            'repetition': has_repetition,
            'symmetry': has_symmetry,
            'symmetry_types': list(symmetry_types),
            'periodicity': has_periodicity,
            'pattern_count': total_patterns
        }
    
    def _analyze_structure(self,
                         input_grids: List[Grid],
                         output_grids: List[Grid]) -> Dict[str, Any]:
        """Analyze structural features"""
        
        has_components = False
        total_components = 0
        has_grid = False
        has_boundaries = False
        sparsity_values = []
        
        for grid in output_grids:
            # Check connected components
            components = grid.get_connected_components()
            if components:
                has_components = True
                total_components += len(components)
            
            # Check grid structure (regular spacing)
            detector = PatternDetector(grid)
            lines = detector.find_lines()
            if lines:
                # Check for regular grid pattern
                h_lines = [l for l in lines if l['type'] == 'horizontal']
                v_lines = [l for l in lines if l['type'] == 'vertical']
                
                if len(h_lines) > 2 and len(v_lines) > 2:
                    has_grid = True
            
            # Check boundaries (using edge detection - simplified for now)
            # edges = detector.find_edges()  # This method doesn't exist yet
            # if edges:
            #     has_boundaries = True
            # For now, check if there are non-zero values at grid edges
            if np.any(grid.data[0, :]) or np.any(grid.data[-1, :]) or \
               np.any(grid.data[:, 0]) or np.any(grid.data[:, -1]):
                has_boundaries = True
            
            # Calculate sparsity
            non_zero = np.count_nonzero(grid.data)
            total = grid.data.size
            sparsity = 1.0 - (non_zero / total) if total > 0 else 0.0
            sparsity_values.append(sparsity)
        
        avg_sparsity = np.mean(sparsity_values) if sparsity_values else 0.0
        
        return {
            'has_components': has_components,
            'component_count': total_components,
            'grid_structure': has_grid,
            'boundaries': has_boundaries,
            'sparsity': avg_sparsity
        }
    
    def _analyze_transformations(self,
                                input_grids: List[Grid],
                                output_grids: List[Grid]) -> Dict[str, Any]:
        """Analyze transformation types"""
        
        is_rotation = False
        is_reflection = False
        is_scaling = False
        is_translation = False
        is_color_swap = False
        
        for inp, out in zip(input_grids, output_grids):
            # Check rotation
            for degrees in [90, 180, 270]:
                if inp.rotate(degrees) == out:
                    is_rotation = True
                    break
            
            # Check reflection
            if inp.flip_horizontal() == out or inp.flip_vertical() == out:
                is_reflection = True
            
            # Check scaling
            for factor in [2, 3]:
                if inp.scale(factor) == out:
                    is_scaling = True
                    break
            
            # Check translation (simplified)
            if inp.shape == out.shape:
                # Check if pattern is shifted
                inp_data = inp.data
                out_data = out.data
                
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        
                        shifted = np.roll(inp_data, (dy, dx), axis=(0, 1))
                        if np.array_equal(shifted, out_data):
                            is_translation = True
                            break
            
            # Check color swap
            if inp.shape == out.shape:
                inp_colors = inp.unique_colors
                out_colors = out.unique_colors
                
                if inp_colors == out_colors and len(inp_colors) > 1:
                    # Check if colors are systematically swapped
                    color_map = {}
                    for color in inp_colors:
                        inp_positions = set(inp.get_color_positions(color))
                        
                        for out_color in out_colors:
                            out_positions = set(out.get_color_positions(out_color))
                            
                            if inp_positions == out_positions:
                                color_map[color] = out_color
                                break
                    
                    if len(color_map) == len(inp_colors) and any(k != v for k, v in color_map.items()):
                        is_color_swap = True
        
        return {
            'rotation': is_rotation,
            'reflection': is_reflection,
            'scaling': is_scaling,
            'translation': is_translation,
            'color_swap': is_color_swap
        }
    
    def _analyze_complexity(self,
                          input_grids: List[Grid],
                          output_grids: List[Grid]) -> Dict[str, Any]:
        """Analyze task complexity"""
        
        # Calculate pixel accuracy variance
        accuracies = []
        for inp, out in zip(input_grids, output_grids):
            if inp.shape == out.shape:
                matches = np.sum(inp.data == out.data)
                total = inp.data.size
                accuracy = matches / total if total > 0 else 0.0
                accuracies.append(accuracy)
        
        accuracy_variance = statistics.variance(accuracies) if len(accuracies) > 1 else 0.0
        
        # Transformation consistency
        consistency = 1.0
        if len(output_grids) > 1:
            # Check if outputs follow similar patterns
            pattern_similarities = []
            for i in range(len(output_grids) - 1):
                for j in range(i + 1, len(output_grids)):
                    sim = self._pattern_similarity(output_grids[i], output_grids[j])
                    pattern_similarities.append(sim)
            
            if pattern_similarities:
                consistency = np.mean(pattern_similarities)
        
        # Rule complexity (heuristic)
        rule_complexity = 1
        
        # Increase complexity for various factors
        if len(input_grids[0].unique_colors) > 3:
            rule_complexity += 1
        if input_grids[0].shape != output_grids[0].shape:
            rule_complexity += 1
        if not self._check_color_mapping(input_grids, output_grids):
            rule_complexity += 1
        if accuracy_variance > 0.2:
            rule_complexity += 1
        
        return {
            'accuracy_variance': accuracy_variance,
            'consistency': consistency,
            'rule_complexity': rule_complexity
        }
    
    def _pattern_similarity(self, grid1: Grid, grid2: Grid) -> float:
        """Calculate pattern similarity between two grids"""
        
        detector1 = PatternDetector(grid1)
        detector2 = PatternDetector(grid2)
        
        # Compare symmetries
        sym1 = detector1.get_symmetries()
        sym2 = detector2.get_symmetries()
        
        sym_matches = sum(1 for k in sym1 if sym1[k] == sym2.get(k, False))
        sym_similarity = sym_matches / len(sym1) if sym1 else 0.0
        
        # Compare color distribution
        colors1 = grid1.count_colors()
        colors2 = grid2.count_colors()
        
        all_colors = set(colors1.keys()) | set(colors2.keys())
        if all_colors:
            color_diff = 0
            for color in all_colors:
                count1 = colors1.get(color, 0)
                count2 = colors2.get(color, 0)
                color_diff += abs(count1 - count2)
            
            total_pixels = grid1.data.size + grid2.data.size
            color_similarity = 1.0 - (color_diff / total_pixels) if total_pixels > 0 else 0.0
        else:
            color_similarity = 1.0
        
        return (sym_similarity + color_similarity) / 2


class TaskClassifier:
    """Classifies ARC tasks into categories"""
    
    def classify(self, features: TaskFeatures) -> TaskCategory:
        """Classify task based on features"""
        
        scores = {
            TaskCategory.GEOMETRIC: self._score_geometric(features),
            TaskCategory.COLOR_MANIPULATION: self._score_color_manipulation(features),
            TaskCategory.PATTERN_COMPLETION: self._score_pattern_completion(features),
            TaskCategory.COUNTING: self._score_counting(features),
            TaskCategory.SYMMETRY: self._score_symmetry(features),
            TaskCategory.OBJECT_MANIPULATION: self._score_object_manipulation(features),
            TaskCategory.LOGICAL: self._score_logical(features),
            TaskCategory.SPATIAL_REASONING: self._score_spatial_reasoning(features)
        }
        
        # Return category with highest score
        best_category = max(scores, key=scores.get)
        
        if scores[best_category] < 0.3:
            return TaskCategory.UNKNOWN
        
        return best_category
    
    def _score_geometric(self, features: TaskFeatures) -> float:
        """Score for geometric transformation tasks"""
        score = 0.0
        
        if features.is_rotation:
            score += 0.4
        if features.is_reflection:
            score += 0.4
        if features.is_scaling:
            score += 0.3
        if features.is_translation:
            score += 0.3
        if features.consistent_size_change:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_color_manipulation(self, features: TaskFeatures) -> float:
        """Score for color manipulation tasks"""
        score = 0.0
        
        if features.color_mapping_exists:
            score += 0.5
        if features.is_color_swap:
            score += 0.4
        if features.color_reduction:
            score += 0.3
        if len(features.unique_colors_output) != len(features.unique_colors_input):
            score += 0.2
        
        return min(1.0, score)
    
    def _score_pattern_completion(self, features: TaskFeatures) -> float:
        """Score for pattern completion tasks"""
        score = 0.0
        
        if features.has_repetition:
            score += 0.4
        if features.has_periodicity:
            score += 0.4
        if features.pattern_count > 2:
            score += 0.3
        if features.has_grid_structure:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_counting(self, features: TaskFeatures) -> float:
        """Score for counting tasks"""
        score = 0.0
        
        # Check if output size correlates with component count
        if features.component_count > 0:
            score += 0.3
        
        # Check if output is much smaller than input (possible count encoding)
        if features.avg_output_size < features.avg_input_size * 0.1:
            score += 0.4
        
        return min(1.0, score)
    
    def _score_symmetry(self, features: TaskFeatures) -> float:
        """Score for symmetry-based tasks"""
        score = 0.0
        
        if features.has_symmetry:
            score += 0.5
        if len(features.symmetry_types) > 1:
            score += 0.3
        if features.is_reflection:
            score += 0.3
        
        return min(1.0, score)
    
    def _score_object_manipulation(self, features: TaskFeatures) -> float:
        """Score for object manipulation tasks"""
        score = 0.0
        
        if features.has_connected_components:
            score += 0.4
        if features.component_count > 1:
            score += 0.3
        if features.has_boundaries:
            score += 0.2
        
        return min(1.0, score)
    
    def _score_logical(self, features: TaskFeatures) -> float:
        """Score for logical reasoning tasks"""
        score = 0.0
        
        if features.rule_complexity > 3:
            score += 0.4
        if features.transformation_consistency < 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _score_spatial_reasoning(self, features: TaskFeatures) -> float:
        """Score for spatial reasoning tasks"""
        score = 0.0
        
        if features.is_translation:
            score += 0.3
        if features.has_boundaries:
            score += 0.2
        if features.sparsity > 0.5:
            score += 0.2
        
        return min(1.0, score)


class StrategySelector:
    """Selects optimal solving strategy for ARC tasks"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.task_classifier = TaskClassifier()
    
    def select_strategy(self,
                       train_inputs: List[np.ndarray],
                       train_outputs: List[np.ndarray]) -> StrategyRecommendation:
        """Select optimal strategy for the task"""
        
        # Extract features
        features = self.feature_extractor.extract_features(train_inputs, train_outputs)
        
        # Classify task
        category = self.task_classifier.classify(features)
        
        # Determine complexity
        complexity = self._assess_complexity(features)
        
        # Select strategy based on category and features
        strategy = self._determine_strategy(category, features, complexity)
        
        return strategy
    
    def _assess_complexity(self, features: TaskFeatures) -> ComplexityLevel:
        """Assess task complexity"""
        
        score = 0
        
        # Size complexity
        if features.avg_output_size > 100:
            score += 2
        elif features.avg_output_size > 25:
            score += 1
        
        # Color complexity
        if len(features.unique_colors_output) > 5:
            score += 1
        
        # Pattern complexity
        if features.pattern_count > 5:
            score += 1
        
        # Rule complexity
        score += min(2, features.rule_complexity // 2)
        
        # Transformation complexity
        if features.pixel_accuracy_variance > 0.3:
            score += 1
        
        # Map score to complexity level
        if score <= 1:
            return ComplexityLevel.TRIVIAL
        elif score <= 3:
            return ComplexityLevel.SIMPLE
        elif score <= 5:
            return ComplexityLevel.MODERATE
        elif score <= 7:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    def _determine_strategy(self,
                          category: TaskCategory,
                          features: TaskFeatures,
                          complexity: ComplexityLevel) -> StrategyRecommendation:
        """Determine solving strategy based on analysis"""
        
        # Strategy mappings for each category
        strategy_map = {
            TaskCategory.GEOMETRIC: {
                'primary': 'pattern_solver',
                'alternatives': ['program_synthesis'],
                'reasoning': 'Geometric transformations detected'
            },
            TaskCategory.COLOR_MANIPULATION: {
                'primary': 'csp_solver',
                'alternatives': ['pattern_solver', 'program_synthesis'],
                'reasoning': 'Color mapping patterns detected'
            },
            TaskCategory.PATTERN_COMPLETION: {
                'primary': 'pattern_solver',
                'alternatives': ['program_synthesis'],
                'reasoning': 'Repeating patterns detected'
            },
            TaskCategory.COUNTING: {
                'primary': 'program_synthesis',
                'alternatives': ['csp_solver'],
                'reasoning': 'Counting or enumeration task detected'
            },
            TaskCategory.SYMMETRY: {
                'primary': 'pattern_solver',
                'alternatives': ['program_synthesis'],
                'reasoning': 'Symmetry-based transformations detected'
            },
            TaskCategory.OBJECT_MANIPULATION: {
                'primary': 'program_synthesis',
                'alternatives': ['pattern_solver', 'csp_solver'],
                'reasoning': 'Object manipulation patterns detected'
            },
            TaskCategory.LOGICAL: {
                'primary': 'csp_solver',
                'alternatives': ['program_synthesis'],
                'reasoning': 'Logical constraints detected'
            },
            TaskCategory.SPATIAL_REASONING: {
                'primary': 'program_synthesis',
                'alternatives': ['pattern_solver'],
                'reasoning': 'Spatial reasoning required'
            },
            TaskCategory.UNKNOWN: {
                'primary': 'ensemble',
                'alternatives': ['program_synthesis', 'pattern_solver', 'csp_solver'],
                'reasoning': 'Task category unclear, using ensemble approach'
            }
        }
        
        strategy_info = strategy_map.get(category, strategy_map[TaskCategory.UNKNOWN])
        
        # Adjust for complexity
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            # For complex tasks, prefer ensemble or program synthesis
            if strategy_info['primary'] != 'ensemble':
                strategy_info['alternatives'].insert(0, strategy_info['primary'])
                strategy_info['primary'] = 'ensemble'
                strategy_info['reasoning'] += ' (high complexity detected)'
        
        # Calculate confidence
        confidence = self._calculate_confidence(category, features, complexity)
        
        # Determine timeout
        timeout_map = {
            ComplexityLevel.TRIVIAL: 5.0,
            ComplexityLevel.SIMPLE: 10.0,
            ComplexityLevel.MODERATE: 15.0,
            ComplexityLevel.COMPLEX: 30.0,
            ComplexityLevel.VERY_COMPLEX: 60.0
        }
        
        suggested_timeout = timeout_map[complexity]
        
        return StrategyRecommendation(
            primary_strategy=strategy_info['primary'],
            alternative_strategies=strategy_info['alternatives'],
            confidence=confidence,
            reasoning=strategy_info['reasoning'],
            expected_difficulty=complexity,
            suggested_timeout=suggested_timeout
        )
    
    def _calculate_confidence(self,
                            category: TaskCategory,
                            features: TaskFeatures,
                            complexity: ComplexityLevel) -> float:
        """Calculate confidence in strategy selection"""
        
        base_confidence = 0.5
        
        # Adjust based on category clarity
        if category != TaskCategory.UNKNOWN:
            base_confidence += 0.2
        
        # Adjust based on feature consistency
        if features.transformation_consistency > 0.8:
            base_confidence += 0.15
        
        # Adjust based on complexity
        if complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE]:
            base_confidence += 0.1
        elif complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            base_confidence -= 0.1
        
        # Adjust based on number of examples
        if features.num_examples >= 3:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))