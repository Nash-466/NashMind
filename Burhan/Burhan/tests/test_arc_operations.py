"""
Comprehensive tests for ARC operations modules
"""

import unittest
import numpy as np
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arc.task_loader import TaskLoader, ARCTask, ARCExample
from src.arc.grid_operations import Grid
from src.arc.pattern_detector import PatternDetector
from src.arc.transformation_rules import (
    TransformationRule, TransformationType, RuleChain, 
    TransformationInference, RuleLibrary
)


class TestTaskLoader(unittest.TestCase):
    """Test the TaskLoader class"""
    
    def setUp(self):
        self.loader = TaskLoader()
        
        # Sample task data
        self.sample_data = {
            "test_task": {
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                    {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]}
                ],
                "test": [
                    {"input": [[9, 0], [1, 2]]}
                ]
            }
        }
    
    def test_load_from_json_string(self):
        """Test loading tasks from JSON string"""
        tasks = self.loader.load_from_json_string(json.dumps(self.sample_data))
        
        self.assertEqual(len(tasks), 1)
        self.assertIn("test_task", tasks)
        
        task = tasks["test_task"]
        self.assertEqual(task.num_train, 2)
        self.assertEqual(task.num_test, 1)
    
    def test_arc_example(self):
        """Test ARCExample dataclass"""
        example = ARCExample(
            input=[[1, 2], [3, 4]],
            output=[[4, 3], [2, 1]]
        )
        
        self.assertEqual(example.input_shape, (2, 2))
        self.assertEqual(example.output_shape, (2, 2))
        self.assertTrue(isinstance(example.input, np.ndarray))
    
    def test_task_validation(self):
        """Test task validation"""
        task = self.loader.get_task("test_task")
        if task:
            self.assertTrue(task.validate())
    
    def test_get_statistics(self):
        """Test statistics generation"""
        self.loader.load_from_json_string(json.dumps(self.sample_data))
        stats = self.loader.get_statistics()
        
        self.assertEqual(stats['total_tasks'], 1)
        self.assertEqual(stats['total_train_examples'], 2)
        self.assertEqual(stats['total_test_examples'], 1)


class TestGridOperations(unittest.TestCase):
    """Test the Grid class and operations"""
    
    def setUp(self):
        self.grid = Grid(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
    
    def test_grid_creation(self):
        """Test grid creation and properties"""
        self.assertEqual(self.grid.shape, (3, 3))
        self.assertEqual(self.grid.height, 3)
        self.assertEqual(self.grid.width, 3)
        self.assertEqual(self.grid.unique_colors, {1, 2, 3, 4, 5, 6, 7, 8, 9})
    
    def test_rotation(self):
        """Test grid rotation"""
        rotated_90 = self.grid.rotate(90)
        expected = np.array([
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ])
        np.testing.assert_array_equal(rotated_90.data, expected)
        
        rotated_180 = self.grid.rotate(180)
        expected_180 = np.array([
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ])
        np.testing.assert_array_equal(rotated_180.data, expected_180)
    
    def test_flip(self):
        """Test grid flipping"""
        flipped_h = self.grid.flip_horizontal()
        expected_h = np.array([
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ])
        np.testing.assert_array_equal(flipped_h.data, expected_h)
        
        flipped_v = self.grid.flip_vertical()
        expected_v = np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ])
        np.testing.assert_array_equal(flipped_v.data, expected_v)
    
    def test_transpose(self):
        """Test grid transpose"""
        transposed = self.grid.transpose()
        expected = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ])
        np.testing.assert_array_equal(transposed.data, expected)
    
    def test_scale(self):
        """Test grid scaling"""
        small_grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        scaled = small_grid.scale(2)
        expected = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ])
        np.testing.assert_array_equal(scaled.data, expected)
    
    def test_tile(self):
        """Test grid tiling"""
        small_grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        tiled = small_grid.tile(2, 2)
        expected = np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]
        ])
        np.testing.assert_array_equal(tiled.data, expected)
    
    def test_color_operations(self):
        """Test color operations"""
        # Replace color
        replaced = self.grid.replace_color(5, 0)
        self.assertEqual(replaced.data[1, 1], 0)
        
        # Map colors
        color_map = {1: 9, 9: 1}
        mapped = self.grid.map_colors(color_map)
        self.assertEqual(mapped.data[0, 0], 9)
        self.assertEqual(mapped.data[2, 2], 1)
        
        # Filter color
        filtered = self.grid.filter_color(5, background=0)
        self.assertEqual(filtered.data[1, 1], 5)
        self.assertEqual(filtered.data[0, 0], 0)
        
        # Count colors
        counts = self.grid.count_colors()
        self.assertEqual(len(counts), 9)
        self.assertEqual(counts[5], 1)
    
    def test_subgrid_operations(self):
        """Test subgrid extraction and replacement"""
        # Extract subgrid
        subgrid = self.grid.extract_subgrid(1, 1, 2, 2)
        expected = np.array([
            [5, 6],
            [8, 9]
        ])
        np.testing.assert_array_equal(subgrid.data, expected)
        
        # Replace subgrid
        new_subgrid = Grid(np.array([
            [0, 0],
            [0, 0]
        ]))
        replaced = self.grid.replace_subgrid(new_subgrid, 1, 1)
        self.assertEqual(replaced.data[1, 1], 0)
        self.assertEqual(replaced.data[2, 2], 0)
    
    def test_pattern_finding(self):
        """Test pattern finding"""
        grid_with_pattern = Grid(np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]
        ]))
        
        pattern = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        
        positions = grid_with_pattern.find_pattern(pattern)
        self.assertEqual(len(positions), 4)
        self.assertIn((0, 0), positions)
        self.assertIn((0, 2), positions)
        self.assertIn((2, 0), positions)
        self.assertIn((2, 2), positions)
    
    def test_bounding_box(self):
        """Test bounding box detection"""
        sparse_grid = Grid(np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ]))
        
        bbox = sparse_grid.get_bounding_box()
        self.assertEqual(bbox, (1, 1, 3, 3))
        
        cropped = sparse_grid.crop_to_content()
        self.assertEqual(cropped.shape, (2, 2))
    
    def test_padding(self):
        """Test grid padding"""
        small_grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        
        padded = small_grid.pad(1, 1, 1, 1, fill_value=0)
        expected = np.array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ])
        np.testing.assert_array_equal(padded.data, expected)


class TestPatternDetector(unittest.TestCase):
    """Test the PatternDetector class"""
    
    def test_symmetry_detection(self):
        """Test symmetry detection"""
        # Horizontal symmetry
        h_symmetric = Grid(np.array([
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 5]
        ]))
        detector = PatternDetector(h_symmetric)
        self.assertTrue(detector.has_horizontal_symmetry())
        
        # Vertical symmetry
        v_symmetric = Grid(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3]
        ]))
        detector = PatternDetector(v_symmetric)
        self.assertTrue(detector.has_vertical_symmetry())
        
        # Diagonal symmetry
        d_symmetric = Grid(np.array([
            [1, 2, 3],
            [2, 5, 6],
            [3, 6, 9]
        ]))
        detector = PatternDetector(d_symmetric)
        self.assertTrue(detector.has_diagonal_symmetry())
    
    def test_periodicity_detection(self):
        """Test periodicity detection"""
        periodic_grid = Grid(np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]
        ]))
        
        detector = PatternDetector(periodic_grid)
        periodicity = detector.detect_periodicity()
        
        self.assertEqual(periodicity['horizontal'], 2)
        self.assertEqual(periodicity['vertical'], 2)
    
    def test_shape_detection(self):
        """Test rectangle and line detection"""
        grid_with_shapes = Grid(np.array([
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0]
        ]))
        
        detector = PatternDetector(grid_with_shapes)
        
        # Find rectangles
        rectangles = detector.find_rectangles()
        self.assertTrue(len(rectangles) > 0)
        
        # Find lines
        lines = detector.find_lines(min_length=3)
        self.assertTrue(len(lines) > 0)
        
        # Check for horizontal line of 2s
        has_horizontal_line = any(
            line['type'] == 'horizontal' and line['color'] == 2 
            for line in lines
        )
        self.assertTrue(has_horizontal_line)
    
    def test_connected_components(self):
        """Test connected component detection"""
        grid_with_components = Grid(np.array([
            [1, 1, 0, 2, 2],
            [1, 0, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [4, 0, 3, 3, 0],
            [4, 4, 0, 0, 0]
        ]))
        
        detector = PatternDetector(grid_with_components)
        components = detector.find_connected_components(connectivity=4)
        
        # Should find 4 components (one for each color)
        self.assertEqual(len(components), 4)
        
        # Check component properties
        for component in components:
            self.assertIn('color', component)
            self.assertIn('area', component)
            self.assertIn('centroid', component)
            self.assertIn('bbox', component)


class TestTransformationRules(unittest.TestCase):
    """Test the transformation rules system"""
    
    def setUp(self):
        self.grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
    
    def test_transformation_rule_application(self):
        """Test applying transformation rules"""
        # Test rotation rule
        rotate_rule = TransformationRule(
            name="rotate_90",
            transformation_type=TransformationType.ROTATE,
            parameters={'degrees': 90},
            description="Rotate 90 degrees"
        )
        
        rotated = rotate_rule.apply(self.grid)
        expected = np.array([
            [3, 1],
            [4, 2]
        ])
        np.testing.assert_array_equal(rotated.data, expected)
        
        # Test scale rule
        scale_rule = TransformationRule(
            name="scale_2x",
            transformation_type=TransformationType.SCALE,
            parameters={'factor': 2},
            description="Scale by factor 2"
        )
        
        scaled = scale_rule.apply(self.grid)
        self.assertEqual(scaled.shape, (4, 4))
    
    def test_rule_chain(self):
        """Test chaining multiple rules"""
        chain = RuleChain([
            TransformationRule(
                name="flip_h",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'horizontal'},
                description="Flip horizontally"
            ),
            TransformationRule(
                name="flip_v",
                transformation_type=TransformationType.FLIP,
                parameters={'direction': 'vertical'},
                description="Flip vertically"
            )
        ])
        
        result = chain.apply(self.grid)
        expected = np.array([
            [4, 3],
            [2, 1]
        ])
        np.testing.assert_array_equal(result.data, expected)
    
    def test_transformation_inference(self):
        """Test inferring transformations from examples"""
        # Create example with rotation
        input_grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        
        output_grid = Grid(np.array([
            [3, 1],
            [4, 2]
        ]))
        
        rule = TransformationInference.infer_simple_transformation(input_grid, output_grid)
        
        self.assertIsNotNone(rule)
        self.assertEqual(rule.transformation_type, TransformationType.ROTATE)
        self.assertEqual(rule.parameters['degrees'], 90)
    
    def test_rule_library(self):
        """Test the rule library"""
        # Get rotation rules
        rotation_rules = RuleLibrary.get_rotation_rules()
        self.assertEqual(len(rotation_rules), 3)
        
        # Get flip rules
        flip_rules = RuleLibrary.get_flip_rules()
        self.assertEqual(len(flip_rules), 2)
        
        # Get scale rule
        scale_rule = RuleLibrary.get_scale_rule(3)
        self.assertEqual(scale_rule.parameters['factor'], 3)
        
        # Test color swap
        grid_with_colors = Grid(np.array([
            [1, 2, 1],
            [2, 1, 2],
            [1, 2, 1]
        ]))
        
        swap_chain = RuleLibrary.get_color_swap_rule(1, 2)
        swapped = swap_chain.apply(grid_with_colors)
        
        # Colors should be swapped
        self.assertEqual(swapped.data[0, 0], 2)
        self.assertEqual(swapped.data[0, 1], 1)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()