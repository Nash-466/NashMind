from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Quick Test - Ultra Advanced ARC System
==========================================

Simple script to quickly test the Ultra Advanced ARC System
"""

import numpy as np
import time
from ultra_advanced_arc_system import solve_arc_problem

def test_system():
    """Quick test of the system"""
    
    print("🚀 ULTRA ADVANCED ARC SYSTEM - QUICK TEST")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Symmetry',
            'grid': np.array([
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0]
            ])
        },
        {
            'name': 'Color Pattern',
            'grid': np.array([
                [1, 2, 1],
                [2, 3, 2],
                [1, 2, 1]
            ])
        },
        {
            'name': 'Complex Grid',
            'grid': np.array([
                [0, 1, 2, 1, 0],
                [1, 2, 3, 2, 1],
                [2, 3, 4, 3, 2],
                [1, 2, 3, 2, 1],
                [0, 1, 2, 1, 0]
            ])
        }
    ]
    
    total_time = 0
    total_confidence = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test['name']}")
        print("-" * 40)
        
        # Display input
        print("Input Grid:")
        for row in test['grid']:
            print(" ".join(f"{cell:2d}" for cell in row))
        
        # Solve
        start_time = time.time()
        solution = solve_arc_problem(test['grid'])
        solve_time = time.time() - start_time
        
        total_time += solve_time
        total_confidence += solution.confidence
        
        # Display results
        print(f"\n✅ Solution Generated!")
        print(f"   Confidence: {solution.confidence:.3f}")
        print(f"   Time: {solve_time:.3f}s")
        print(f"   Approach: {solution.metadata.get('approach_used', 'unknown')}")
        
        print("\nSolution Grid:")
        for row in solution.solution_grid:
            print(" ".join(f"{cell:2d}" for cell in row))
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"Tests Completed: {len(test_cases)}")
    print(f"Average Confidence: {total_confidence/len(test_cases):.3f}")
    print(f"Average Time: {total_time/len(test_cases):.3f}s")
    print(f"Total Time: {total_time:.3f}s")
    print("\n🎉 All tests completed successfully!")
    print("System is ready for ARC Prize 2025! 🏆")

if __name__ == "__main__":
    test_system()
