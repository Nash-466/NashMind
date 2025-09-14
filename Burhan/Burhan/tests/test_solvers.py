"""
Test Suite for ARC Solvers
Validates all solving strategies and components
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.arc.task_loader import TaskLoader
from src.arc.grid_operations import Grid
from src.solvers.program_synthesis import ProgramSynthesisEngine, Program, DSLOperation, DSLInstruction
from src.solvers.csp_solver import CSPSolver
from src.solvers.pattern_solver import PatternSolver
from src.solvers.ensemble_solver import EnsembleSolver
from src.strategies.strategy_selector import StrategySelector


def test_program_synthesis():
    """Test program synthesis solver"""
    print("\n" + "="*60)
    print("Testing Program Synthesis Solver")
    print("="*60)
    
    # Create simple test case - scaling
    train_inputs = [
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        np.array([[5, 6], [7, 8]], dtype=np.int8)
    ]
    
    train_outputs = [
        np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int8),
        np.array([[5, 5, 6, 6], [5, 5, 6, 6], [7, 7, 8, 8], [7, 7, 8, 8]], dtype=np.int8)
    ]
    
    test_input = np.array([[2, 3], [4, 5]], dtype=np.int8)
    
    # Initialize solver
    solver = ProgramSynthesisEngine()
    
    # Get synthesized program
    program = solver.get_program(train_inputs, train_outputs)
    
    if program:
        print(f"✓ Synthesized program with {len(program.instructions)} instructions")
        print(f"  Program: {program.to_string()}")
        print(f"  Score: {program.score:.3f}")
        
        # Apply to test input
        result = solver.solve(train_inputs, train_outputs, test_input)
        if result is not None:
            print(f"✓ Applied program to test input")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {result.shape}")
    else:
        print("✗ Failed to synthesize program")
    
    return program is not None


def test_csp_solver():
    """Test CSP solver"""
    print("\n" + "="*60)
    print("Testing CSP Solver")
    print("="*60)
    
    # Create simple test case - color mapping
    train_inputs = [
        np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.int8),
        np.array([[3, 4, 3], [4, 3, 4], [3, 4, 3]], dtype=np.int8)
    ]
    
    train_outputs = [
        np.array([[2, 1, 2], [1, 2, 1], [2, 1, 2]], dtype=np.int8),
        np.array([[4, 3, 4], [3, 4, 3], [4, 3, 4]], dtype=np.int8)
    ]
    
    test_input = np.array([[5, 6, 5], [6, 5, 6], [5, 6, 5]], dtype=np.int8)
    
    # Initialize solver
    solver = CSPSolver()
    
    # Solve
    result = solver.solve(train_inputs, train_outputs, test_input)
    
    if result is not None:
        print("✓ CSP solver found solution")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Unique colors in output: {np.unique(result)}")
    else:
        print("✗ CSP solver failed to find solution")
    
    return result is not None


def test_pattern_solver():
    """Test pattern solver"""
    print("\n" + "="*60)
    print("Testing Pattern Solver")
    print("="*60)
    
    # Create test case - tiling pattern
    train_inputs = [
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        np.array([[5]], dtype=np.int8)
    ]
    
    train_outputs = [
        np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]], dtype=np.int8),
        np.array([[5, 5], [5, 5]], dtype=np.int8)
    ]
    
    test_input = np.array([[7, 8], [9, 0]], dtype=np.int8)
    
    # Initialize solver
    solver = PatternSolver()
    
    # Solve
    result = solver.solve(train_inputs, train_outputs, test_input)
    
    if result is not None:
        print("✓ Pattern solver found solution")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {result.shape}")
        
        # Check if it's a tiling pattern
        if result.shape[0] == test_input.shape[0] * 2 and result.shape[1] == test_input.shape[1] * 2:
            print("  Pattern type: Tiling (2x2)")
    else:
        print("✗ Pattern solver failed to find solution")
    
    return result is not None


def test_ensemble_solver():
    """Test ensemble solver"""
    print("\n" + "="*60)
    print("Testing Ensemble Solver")
    print("="*60)
    
    # Create test case
    train_inputs = [
        np.array([[1, 0], [0, 1]], dtype=np.int8),
        np.array([[2, 0], [0, 2]], dtype=np.int8)
    ]
    
    train_outputs = [
        np.array([[0, 1], [1, 0]], dtype=np.int8),
        np.array([[0, 2], [2, 0]], dtype=np.int8)
    ]
    
    test_input = np.array([[3, 0], [0, 3]], dtype=np.int8)
    
    # Initialize solver with configuration
    config = {
        'use_program_synthesis': True,
        'use_csp': True,
        'use_pattern': True,
        'use_hybrid': False,  # Skip for speed
        'voting_type': 'weighted',
        'parallel': False,  # Sequential for testing
        'timeout_per_solver': 5.0
    }
    
    solver = EnsembleSolver(config)
    
    # Solve
    result = solver.solve(train_inputs, train_outputs, test_input)
    
    print(f"✓ Ensemble solver completed")
    print(f"  Total time: {result.total_time:.2f}s")
    print(f"  Consensus score: {result.consensus_score:.3f}")
    print(f"  Strategy used: {result.strategy_used}")
    print(f"  Individual results:")
    
    for res in result.individual_results:
        status = "✓" if res.solution is not None else "✗"
        print(f"    {status} {res.solver_type.value}: confidence={res.confidence:.3f}, time={res.execution_time:.2f}s")
    
    if result.final_solution is not None:
        print(f"✓ Final solution found")
        print(f"  Shape: {result.final_solution.shape}")
    else:
        print("✗ No consensus solution found")
    
    return result.final_solution is not None


def test_strategy_selector():
    """Test strategy selection"""
    print("\n" + "="*60)
    print("Testing Strategy Selector")
    print("="*60)
    
    # Create test cases with different characteristics
    test_cases = [
        {
            'name': 'Scaling Task',
            'inputs': [np.array([[1, 2], [3, 4]], dtype=np.int8)],
            'outputs': [np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int8)]
        },
        {
            'name': 'Color Swap Task',
            'inputs': [np.array([[1, 2, 1], [2, 1, 2]], dtype=np.int8)],
            'outputs': [np.array([[2, 1, 2], [1, 2, 1]], dtype=np.int8)]
        },
        {
            'name': 'Rotation Task',
            'inputs': [np.array([[1, 2], [3, 4]], dtype=np.int8)],
            'outputs': [np.array([[2, 4], [1, 3]], dtype=np.int8)]
        }
    ]
    
    selector = StrategySelector()
    
    for test_case in test_cases:
        print(f"\n  {test_case['name']}:")
        
        recommendation = selector.select_strategy(
            test_case['inputs'],
            test_case['outputs']
        )
        
        print(f"    Primary strategy: {recommendation.primary_strategy}")
        print(f"    Alternatives: {', '.join(recommendation.alternative_strategies)}")
        print(f"    Confidence: {recommendation.confidence:.3f}")
        print(f"    Difficulty: {recommendation.expected_difficulty.name}")
        print(f"    Reasoning: {recommendation.reasoning}")
    
    return True


def test_real_arc_task():
    """Test on a real ARC task from the dataset"""
    print("\n" + "="*60)
    print("Testing on Real ARC Task")
    print("="*60)
    
    # Load a real task
    loader = TaskLoader()
    
    # Try to load from attached assets
    try:
        loader.load_from_json_file("attached_assets/arc-agi_training_challenges (2)_1757573784094.txt")
        loader.load_from_json_file("attached_assets/arc-agi_training_solutions_1757573784094.txt")
    except:
        print("  Could not load real ARC tasks from files")
        return False
    
    # Get first task
    task_ids = loader.list_tasks()[:5]  # Test first 5 tasks
    
    if not task_ids:
        print("  No tasks loaded")
        return False
    
    # Initialize ensemble solver
    solver = EnsembleSolver({
        'use_program_synthesis': True,
        'use_csp': False,  # Disable for speed
        'use_pattern': True,
        'use_hybrid': False,
        'parallel': False,
        'timeout_per_solver': 3.0
    })
    
    success_count = 0
    
    for task_id in task_ids:
        task = loader.get_task(task_id)
        solution = loader.get_solution(task_id)
        
        if not task or not solution:
            continue
        
        # Get training data
        train_inputs = []
        train_outputs = []
        
        for example in task.train_examples:
            train_inputs.append(example.input)
            train_outputs.append(example.output)
        
        # Get test input
        test_input = task.test_examples[0].input
        expected_output = solution[0]
        
        # Solve
        result = solver.solve(train_inputs, train_outputs, test_input)
        
        if result.final_solution is not None:
            # Check if solution matches expected
            if np.array_equal(result.final_solution, expected_output):
                print(f"  ✓ Task {task_id}: CORRECT (strategy: {result.strategy_used})")
                success_count += 1
            else:
                print(f"  ✗ Task {task_id}: INCORRECT (strategy: {result.strategy_used})")
        else:
            print(f"  ✗ Task {task_id}: NO SOLUTION")
    
    print(f"\n  Success rate: {success_count}/{len(task_ids)} tasks solved correctly")
    
    return success_count > 0


def run_all_tests():
    """Run all solver tests"""
    print("\n" + "="*60)
    print("         ARC SOLVER TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test individual solvers
    results['Program Synthesis'] = test_program_synthesis()
    results['CSP Solver'] = test_csp_solver()
    results['Pattern Solver'] = test_pattern_solver()
    
    # Test ensemble and strategy
    results['Ensemble Solver'] = test_ensemble_solver()
    results['Strategy Selector'] = test_strategy_selector()
    
    # Test on real data
    results['Real ARC Task'] = test_real_arc_task()
    
    # Summary
    print("\n" + "="*60)
    print("                TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed successfully!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)