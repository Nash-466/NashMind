#!/usr/bin/env python3
"""
ARC Prize 2025 Solver - Comprehensive Task Parser and Grid Manipulator
Demonstrates all capabilities of the ARC system
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our ARC modules
from src.arc.task_loader import TaskLoader, ARCTask, ARCExample
from src.arc.grid_operations import Grid
from src.arc.pattern_detector import PatternDetector
from src.arc.transformation_rules import (
    TransformationRule, TransformationType, RuleChain,
    TransformationInference, RuleLibrary
)

# Import the new solvers
from src.solvers.program_synthesis import ProgramSynthesisEngine
from src.solvers.csp_solver import CSPSolver
from src.solvers.pattern_solver import PatternSolver
from src.solvers.ensemble_solver import EnsembleSolver
from src.strategies.strategy_selector import StrategySelector


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(text.center(60))
    print("=" * 60)


def visualize_grid(grid: Grid, title: str = "Grid"):
    """Visualize a grid with ASCII characters"""
    print(f"\n{title} ({grid.height}x{grid.width}):")
    
    # Simple ASCII representation
    for row in grid.data:
        print(' '.join(str(cell) if cell != 0 else '.' for cell in row))


def demonstrate_task_loading():
    """Demonstrate task loading capabilities"""
    print_header("TASK LOADING DEMONSTRATION")
    
    loader = TaskLoader()
    
    # Load from attached files if they exist
    attached_challenges = "attached_assets/arc-agi_training_challenges (2)_1757573784094.txt"
    attached_solutions = "attached_assets/arc-agi_training_solutions_1757573784094.txt"
    
    tasks_loaded = False
    
    if os.path.exists(attached_challenges):
        print(f"Loading challenges from: {attached_challenges}")
        try:
            with open(attached_challenges, 'r') as f:
                content = f.read()
                tasks = loader.load_from_json_string(content)
                print(f"✓ Loaded {len(tasks)} tasks")
                tasks_loaded = True
        except Exception as e:
            print(f"Error loading challenges: {e}")
    
    if os.path.exists(attached_solutions):
        print(f"Loading solutions from: {attached_solutions}")
        try:
            with open(attached_solutions, 'r') as f:
                content = f.read()
                solutions = loader.load_from_json_string(content, is_solution=True)
                print(f"✓ Loaded solutions for {len(solutions)} tasks")
        except Exception as e:
            print(f"Error loading solutions: {e}")
    
    # Display statistics
    stats = loader.get_statistics()
    print("\nTask Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Total training examples: {stats['total_train_examples']}")
    print(f"  Total test examples: {stats['total_test_examples']}")
    
    # Show sample task
    task_ids = loader.list_tasks()
    if task_ids and len(task_ids) > 0:
        sample_task_id = task_ids[0]
        task = loader.get_task(sample_task_id)
        
        if task and task.num_train > 0:
            print(f"\nSample Task: {sample_task_id}")
            input_grid, output_grid = task.get_train_pair(0)
            visualize_grid(Grid(input_grid), "First Training Input")
            if output_grid is not None:
                visualize_grid(Grid(output_grid), "First Training Output")
    
    return loader


def demonstrate_grid_operations():
    """Demonstrate grid manipulation operations"""
    print_header("GRID OPERATIONS DEMONSTRATION")
    
    # Create sample grid
    sample_data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    grid = Grid(sample_data)
    
    print("\n--- Basic Transformations ---")
    visualize_grid(grid, "Original")
    
    # Rotation
    rotated = grid.rotate(90)
    visualize_grid(rotated, "Rotated 90°")
    
    # Scale
    small_grid = Grid(np.array([[1, 2], [3, 4]]))
    scaled = small_grid.scale(2)
    visualize_grid(scaled, "Scaled 2x")
    
    # Tile
    tiled = small_grid.tile(2, 2)
    visualize_grid(tiled, "Tiled 2x2")
    
    print(f"\nGrid shape: {grid.shape}")
    print(f"Unique colors: {grid.unique_colors}")
    print(f"Color counts: {grid.count_colors()}")


def demonstrate_pattern_detection():
    """Demonstrate pattern detection capabilities"""
    print_header("PATTERN DETECTION DEMONSTRATION")
    
    # Create grid with patterns
    symmetric_grid = Grid(np.array([
        [1, 2, 3, 2, 1],
        [4, 5, 6, 5, 4],
        [7, 8, 9, 8, 7]
    ]))
    
    detector = PatternDetector(symmetric_grid)
    symmetries = detector.get_symmetries()
    
    print("\nSymmetry Detection:")
    for sym_type, has_sym in symmetries.items():
        status = "✓" if has_sym else "✗"
        print(f"  {status} {sym_type}")
    
    # Pattern statistics
    stats = detector.get_pattern_statistics()
    print(f"\nPattern Statistics:")
    print(f"  Grid shape: {stats['grid_shape']}")
    print(f"  Unique colors: {stats['unique_colors']}")
    print(f"  Sparsity: {stats['sparsity']:.2f}")


def demonstrate_transformation_rules():
    """Demonstrate transformation rules and inference"""
    print_header("TRANSFORMATION RULES DEMONSTRATION")
    
    grid = Grid(np.array([[1, 2], [3, 4]]))
    
    print("\n--- Applying Individual Rules ---")
    visualize_grid(grid, "Original")
    
    # Apply rotation
    rotate_rule = RuleLibrary.get_rotation_rules()[0]
    rotated = rotate_rule.apply(grid)
    visualize_grid(rotated, f"After {rotate_rule.description}")
    
    # Infer transformation
    print("\n--- Transformation Inference ---")
    input_grid = Grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    output_grid = input_grid.rotate(90)
    
    inferred_rule = TransformationInference.infer_simple_transformation(input_grid, output_grid)
    if inferred_rule:
        print(f"Inferred: {inferred_rule.description}")


def demonstrate_solvers():
    """Demonstrate the new solving capabilities"""
    print_header("SOLVER DEMONSTRATIONS")
    
    # Example 1: Scaling task
    print("\n--- Program Synthesis Solver ---")
    train_inputs = [
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        np.array([[5, 6], [7, 8]], dtype=np.int8)
    ]
    train_outputs = [
        np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int8),
        np.array([[5, 5, 6, 6], [5, 5, 6, 6], [7, 7, 8, 8], [7, 7, 8, 8]], dtype=np.int8)
    ]
    test_input = np.array([[2, 3], [4, 5]], dtype=np.int8)
    
    ps_solver = ProgramSynthesisEngine()
    program = ps_solver.get_program(train_inputs, train_outputs)
    if program:
        print(f"✓ Synthesized program: {program.to_string()}")
        result = ps_solver.solve(train_inputs, train_outputs, test_input)
        if result is not None:
            print(f"✓ Applied to test input - output shape: {result.shape}")
    
    # Example 2: Pattern Solver
    print("\n--- Pattern Solver ---")
    pattern_solver = PatternSolver()
    result = pattern_solver.solve(train_inputs, train_outputs, test_input)
    if result is not None:
        print(f"✓ Pattern solver found solution with shape: {result.shape}")
    
    # Example 3: Strategy Selection
    print("\n--- Strategy Selector ---")
    selector = StrategySelector()
    recommendation = selector.select_strategy(train_inputs, train_outputs)
    print(f"✓ Recommended strategy: {recommendation.primary_strategy}")
    print(f"  Confidence: {recommendation.confidence:.3f}")
    print(f"  Reasoning: {recommendation.reasoning}")
    
    # Example 4: Ensemble Solver
    print("\n--- Ensemble Solver ---")
    ensemble_config = {
        'use_program_synthesis': True,
        'use_csp': False,  # Disable for speed
        'use_pattern': True,
        'use_hybrid': False,
        'parallel': False,
        'timeout_per_solver': 2.0
    }
    ensemble = EnsembleSolver(ensemble_config)
    ensemble_result = ensemble.solve(train_inputs, train_outputs, test_input)
    
    print(f"✓ Ensemble completed in {ensemble_result.total_time:.2f}s")
    print(f"  Consensus score: {ensemble_result.consensus_score:.3f}")
    print(f"  Strategy used: {ensemble_result.strategy_used}")
    if ensemble_result.final_solution is not None:
        print(f"  Solution shape: {ensemble_result.final_solution.shape}")


def run_quick_tests():
    """Run a quick test of the system"""
    print_header("SYSTEM VALIDATION")
    
    try:
        # Test Grid operations
        grid = Grid(np.array([[1, 2], [3, 4]]))
        assert grid.shape == (2, 2), "Grid shape test failed"
        
        # Test rotation
        rotated = grid.rotate(90)
        assert rotated.data[0, 0] == 2, "Rotation test failed"
        
        # Test pattern detector
        detector = PatternDetector(grid)
        stats = detector.get_pattern_statistics()
        assert 'unique_colors' in stats, "Pattern detector test failed"
        
        # Test transformation rules
        rule = RuleLibrary.get_scale_rule(2)
        scaled = rule.apply(grid)
        assert scaled.shape == (4, 4), "Transformation rule test failed"
        
        # Test new solvers
        ps_solver = ProgramSynthesisEngine()
        assert ps_solver is not None, "Program synthesis initialization failed"
        
        csp_solver = CSPSolver()
        assert csp_solver is not None, "CSP solver initialization failed"
        
        pattern_solver = PatternSolver()
        assert pattern_solver is not None, "Pattern solver initialization failed"
        
        ensemble_solver = EnsembleSolver()
        assert ensemble_solver is not None, "Ensemble solver initialization failed"
        
        selector = StrategySelector()
        assert selector is not None, "Strategy selector initialization failed"
        
        print("✓ All system components validated successfully!")
        
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")
    except Exception as e:
        print(f"✗ Error during validation: {e}")


def main():
    """Main demonstration function"""
    print("=" * 60)
    print("ARC PRIZE 2025 SOLVER")
    print("Task Parser & Grid Manipulator System")
    print("=" * 60)
    
    print("\nEnvironment Setup Complete!")
    
    # Check library versions
    print("\nInstalled Libraries:")
    print("  ✓ NumPy:", np.__version__)
    
    try:
        import scipy
        print("  ✓ SciPy:", scipy.__version__)
    except ImportError:
        print("  ✗ SciPy: Not installed")
    
    print("\nSystem Modules:")
    print("  ✓ src/arc/task_loader.py - Task loading and parsing")
    print("  ✓ src/arc/grid_operations.py - Grid manipulation")
    print("  ✓ src/arc/pattern_detector.py - Pattern detection")
    print("  ✓ src/arc/transformation_rules.py - Transformation rules")
    print("  ✓ tests/test_arc_operations.py - Test suite")
    
    # Run demonstrations
    print("\n" + "=" * 60)
    print("RUNNING DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Task Loading
    loader = demonstrate_task_loading()
    
    # 2. Grid Operations
    demonstrate_grid_operations()
    
    # 3. Pattern Detection
    demonstrate_pattern_detection()
    
    # 4. Transformation Rules
    demonstrate_transformation_rules()
    
    # 5. Solver Demonstrations
    demonstrate_solvers()
    
    # 6. System Validation
    run_quick_tests()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nThe ARC Task Parser and Grid Manipulator system is ready!")
    print("\nCapabilities:")
    print("  ✓ Load and parse ARC tasks from JSON")
    print("  ✓ Comprehensive grid manipulation operations")
    print("  ✓ Pattern and symmetry detection")
    print("  ✓ Transformation rule inference")
    print("  ✓ Task solving framework")
    
    print("\nReady to solve ARC puzzles!")


if __name__ == "__main__":
    main()