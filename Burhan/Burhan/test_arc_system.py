#!/usr/bin/env python3
"""
Simple test script to verify ARC solver system functionality
Tests basic operations and checks for method name mismatches
"""

import json
import numpy as np
import sys
import traceback

# Import ARC modules
from src.arc.grid_operations import Grid
from src.arc.pattern_detector import PatternDetector
from src.arc.task_loader import TaskLoader
from src.solvers.pattern_solver import PatternSolver
from src.solvers.csp_solver import CSPSolver
from src.arc_agent import ARCAgent


def test_grid_operations():
    """Test basic grid operations"""
    print("\n=== Testing Grid Operations ===")
    
    # Create a simple grid
    data = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    grid = Grid(data)
    print(f"Created grid with shape: {grid.shape}")
    print(f"Unique colors: {grid.unique_colors}")
    
    # Test rotation
    rotated = grid.rotate(90)
    print(f"Rotated grid shape: {rotated.shape}")
    
    # Test get_connected_components (this was previously broken)
    components = grid.get_connected_components()
    print(f"Found {len(components)} connected components")
    
    return True


def test_pattern_detector():
    """Test pattern detector functionality"""
    print("\n=== Testing Pattern Detector ===")
    
    # Create a symmetric grid
    data = np.array([
        [1, 2, 1],
        [3, 4, 3],
        [1, 2, 1]
    ])
    
    grid = Grid(data)
    detector = PatternDetector(grid)
    
    # Test symmetry detection
    symmetries = detector.get_symmetries()
    print(f"Symmetries detected: {symmetries}")
    
    # Test get_connected_components (this was the renamed method)
    components = detector.get_connected_components()
    print(f"Found {len(components)} components in pattern detector")
    
    # Test with background inclusion
    components_with_bg = detector.get_connected_components(include_background=True)
    print(f"Found {len(components_with_bg)} components with background")
    
    return True


def test_simple_arc_task():
    """Test solving a simple ARC task"""
    print("\n=== Testing Simple ARC Task ===")
    
    # Create a simple tiling task
    train_inputs = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]])
    ]
    
    train_outputs = [
        np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        np.array([[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]])
    ]
    
    test_input = np.array([[2, 3], [4, 5]])
    
    # Test with pattern solver
    solver = PatternSolver()
    print("Testing PatternSolver...")
    
    try:
        result = solver.solve(train_inputs, train_outputs, test_input)
        if result is not None:
            print(f"Pattern solver produced output with shape: {result.shape}")
        else:
            print("Pattern solver could not find a solution")
    except Exception as e:
        print(f"Pattern solver error: {e}")
        traceback.print_exc()
    
    # Test with CSP solver
    print("\nTesting CSPSolver...")
    csp_solver = CSPSolver()
    
    try:
        result = csp_solver.solve(train_inputs, train_outputs, test_input)
        if result is not None:
            print(f"CSP solver produced output with shape: {result.shape}")
        else:
            print("CSP solver could not find a solution")
    except Exception as e:
        print(f"CSP solver error: {e}")
        traceback.print_exc()
    
    return True


def test_arc_agent():
    """Test the main ARC agent"""
    print("\n=== Testing ARC Agent ===")
    
    # Create a simple color mapping task
    from src.arc.task_loader import ARCTask, ARCExample
    
    train_examples = [
        ARCExample(
            input=np.array([[1, 0], [0, 1]]),
            output=np.array([[3, 0], [0, 3]])
        ),
        ARCExample(
            input=np.array([[2, 0], [0, 2]]),
            output=np.array([[4, 0], [0, 4]])
        )
    ]
    
    test_examples = [
        ARCExample(
            input=np.array([[5, 0], [0, 5]]),
            output=None  # We don't know the output yet
        )
    ]
    
    # Create ARCTask object
    task = ARCTask(
        task_id="test_simple",
        train_examples=train_examples,
        test_examples=test_examples
    )
    
    # Create agent with short timeout for testing
    agent_config = {
        'timeout': 5.0,
        'max_attempts': 3
    }
    
    agent = ARCAgent(config=agent_config)
    
    try:
        result = agent.solve_task(task)
        if result and result.test_solutions:
            print(f"Agent produced {len(result.test_solutions)} solutions")
            if result.test_solutions[0] is not None:
                print(f"First solution shape: {result.test_solutions[0].shape}")
                print(f"Output:\n{result.test_solutions[0]}")
                print(f"Confidence: {result.confidence}")
                print(f"Strategy used: {result.strategy_used}")
        else:
            print("Agent could not find a solution")
    except Exception as e:
        print(f"Agent error: {e}")
        traceback.print_exc()
    
    return True


def main():
    """Run all tests"""
    print("=" * 50)
    print("ARC Solver System Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        ("Grid Operations", test_grid_operations),
        ("Pattern Detector", test_pattern_detector),
        ("Simple ARC Task", test_simple_arc_task),
        ("ARC Agent", test_arc_agent)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! The ARC solver system is functional.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())