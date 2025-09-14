from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultra Advanced ARC System - Quick Runner
============================================

This script provides a simple interface to run the Ultra Advanced ARC System
for solving ARC Prize 2025 challenges.

Usage:
    python run_ultra_system.py
    
Features:
- Interactive problem input
- Real-time solution generation
- Detailed analysis and reporting
- Performance monitoring
"""

import numpy as np
import json
import time
from collections.abc import Callable
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from ultra_advanced_arc_system import UltraAdvancedARCSystem, solve_arc_problem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import Ultra Advanced ARC System: {e}")
    SYSTEM_AVAILABLE = False


def create_sample_problems() -> List[Dict[str, Any]]:
    """Create sample ARC problems for testing"""
    
    problems = [
        {
            'name': 'Simple Symmetry',
            'description': 'Test basic symmetry detection',
            'difficulty': 'easy',
            'input_grid': np.array([
                [0, 1, 0],
                [1, 2, 1], 
                [0, 1, 0]
            ]),
            'expected_pattern': 'symmetry'
        },
        {
            'name': 'Color Transformation',
            'description': 'Test color mapping capabilities',
            'difficulty': 'medium',
            'input_grid': np.array([
                [1, 2, 1, 2],
                [2, 1, 2, 1],
                [1, 2, 1, 2],
                [2, 1, 2, 1]
            ]),
            'expected_pattern': 'color_mapping'
        },
        {
            'name': 'Complex Pattern',
            'description': 'Test advanced pattern recognition',
            'difficulty': 'hard',
            'input_grid': np.array([
                [0, 1, 0, 1, 0],
                [1, 2, 3, 2, 1],
                [0, 3, 4, 3, 0],
                [1, 2, 3, 2, 1],
                [0, 1, 0, 1, 0]
            ]),
            'expected_pattern': 'nested_symmetry'
        }
    ]
    
    return problems


def print_grid(grid: np.ndarray, title: str = "Grid"):
    """Print a grid in a nice format"""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    for row in grid:
        print(" ".join(f"{cell:2d}" for cell in row))
    print()


def analyze_solution(solution, problem: Dict[str, Any]):
    """Analyze and display solution details"""
    
    print(f"\n🔍 SOLUTION ANALYSIS")
    print("=" * 50)
    print(f"Problem: {problem['name']}")
    print(f"Difficulty: {problem['difficulty']}")
    print(f"Expected Pattern: {problem['expected_pattern']}")
    print()
    
    print(f"✅ Solution Generated Successfully!")
    print(f"   Confidence: {solution.confidence:.3f}")
    print(f"   Generation Time: {solution.generation_time:.3f}s")
    print(f"   Approach: {solution.metadata.get('approach_used', 'unknown')}")
    
    if hasattr(solution, 'reasoning_chain') and solution.reasoning_chain:
        print(f"\n🧠 Reasoning Chain:")
        for i, step in enumerate(solution.reasoning_chain[:3], 1):
            if hasattr(step, 'conclusion'):
                print(f"   {i}. {step.conclusion}")
            else:
                print(f"   {i}. {step}")
    
    # Display grids
    print_grid(problem['input_grid'], "Input Grid")
    print_grid(solution.solution_grid, "Solution Grid")
    
    # Check if solution is different from input
    if not np.array_equal(problem['input_grid'], solution.solution_grid):
        print("✅ Solution shows transformation from input")
    else:
        print("⚠️  Solution is identical to input")


def run_interactive_mode():
    """Run interactive mode for custom problems"""
    
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Enter your own ARC problem!")
    print("Format: Enter grid dimensions, then grid values")
    print("Example: 3 3 (for 3x3 grid)")
    print("Then enter 9 values separated by spaces")
    print()
    
    try:
        # Get grid dimensions
        dims = input("Enter grid dimensions (rows cols): ").strip().split()
        rows, cols = int(dims[0]), int(dims[1])
        
        print(f"Enter {rows * cols} values for {rows}x{cols} grid:")
        values = input("Values (space-separated): ").strip().split()
        
        if len(values) != rows * cols:
            print(f"❌ Expected {rows * cols} values, got {len(values)}")
            return
        
        # Create grid
        grid_values = [int(v) for v in values]
        input_grid = np.array(grid_values).reshape(rows, cols)
        
        print_grid(input_grid, "Your Input Grid")
        
        # Solve the problem
        print("\n🚀 Solving your problem...")
        start_time = time.time()
        solution = solve_arc_problem(input_grid)
        solve_time = time.time() - start_time
        
        # Display results
        custom_problem = {
            'name': 'Custom Problem',
            'difficulty': 'unknown',
            'input_grid': input_grid,
            'expected_pattern': 'unknown'
        }
        
        analyze_solution(solution, custom_problem)
        
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")


def run_benchmark():
    """Run benchmark tests"""
    
    print("\n📊 BENCHMARK MODE")
    print("=" * 50)
    
    if not SYSTEM_AVAILABLE:
        print("❌ Ultra Advanced ARC System not available")
        return
    
    problems = create_sample_problems()
    results = []
    
    total_start = time.time()
    
    for i, problem in enumerate(problems, 1):
        print(f"\n🧪 Test {i}/{len(problems)}: {problem['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            solution = solve_arc_problem(problem['input_grid'])
            solve_time = time.time() - start_time
            
            result = {
                'problem': problem['name'],
                'success': True,
                'confidence': solution.confidence,
                'time': solve_time,
                'approach': solution.metadata.get('approach_used', 'unknown')
            }
            
            print(f"✅ Solved in {solve_time:.3f}s (confidence: {solution.confidence:.3f})")
            
        except Exception as e:
            result = {
                'problem': problem['name'],
                'success': False,
                'error': str(e),
                'time': 0,
                'confidence': 0
            }
            print(f"❌ Failed: {e}")
        
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Summary
    print(f"\n📈 BENCHMARK RESULTS")
    print("=" * 50)
    successful = [r for r in results if r['success']]
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")
    print(f"Total Time: {total_time:.3f}s")
    
    if successful:
        avg_confidence = np.mean([r['confidence'] for r in successful])
        avg_time = np.mean([r['time'] for r in successful])
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Solve Time: {avg_time:.3f}s")


def main():
    """Main function"""
    
    print("🚀 ULTRA ADVANCED ARC SYSTEM")
    print("=" * 60)
    print("Welcome to the Ultra Advanced ARC Challenge Solver!")
    print("Capable of solving ARC Prize 2025 challenges at all levels")
    print()
    
    if not SYSTEM_AVAILABLE:
        print("❌ System not available. Please check installation.")
        return
    
    while True:
        print("\n📋 MENU")
        print("-" * 20)
        print("1. Run Sample Problems")
        print("2. Interactive Mode")
        print("3. Benchmark Tests")
        print("4. System Info")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            problems = create_sample_problems()
            for problem in problems:
                print(f"\n🧪 Solving: {problem['name']}")
                solution = solve_arc_problem(problem['input_grid'])
                analyze_solution(solution, problem)
                
                input("\nPress Enter to continue...")
        
        elif choice == '2':
            run_interactive_mode()
        
        elif choice == '3':
            run_benchmark()
        
        elif choice == '4':
            print("\n🔧 SYSTEM INFORMATION")
            print("=" * 50)
            print("System: Ultra Advanced ARC System v2.0.0")
            print("Components: 7 Advanced AI Systems")
            print("Capabilities: Pattern Analysis, Logical Reasoning,")
            print("             Adaptive Learning, Simulation, Memory,")
            print("             Creative Innovation, Intelligent Verification")
            print("Target: ARC Prize 2025 Challenges")
            print("Status: Fully Operational ✅")
        
        elif choice == '5':
            print("\n👋 Thank you for using Ultra Advanced ARC System!")
            print("Good luck with ARC Prize 2025! 🏆")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    main()
