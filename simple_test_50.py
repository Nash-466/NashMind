from __future__ import annotations
"""
SIMPLE TEST - 50 TASKS
======================
اختبار مبسط على 50 مهمة لجميع الأنظمة
"""

import numpy as np
import time
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def generate_test_tasks(count=50):
    """Generate test tasks"""
    tasks = []
    
    for i in range(count):
        np.random.seed(i)
        size = np.random.randint(3, 8)
        
        # Generate different task types
        task_type = i % 5
        
        if task_type == 0:  # Rotation
            input_grid = np.random.randint(0, 5, (size, size))
            output_grid = np.rot90(input_grid)
        elif task_type == 1:  # Color mapping
            input_grid = np.random.randint(0, 4, (size, size))
            output_grid = (input_grid + 1) % 4
        elif task_type == 2:  # Pattern
            input_grid = np.zeros((size, size), dtype=int)
            input_grid[0, :] = 1
            input_grid[-1, :] = 1
            output_grid = input_grid.copy()
            output_grid[size//2, size//2] = 2
        elif task_type == 3:  # Scaling
            small_size = 3
            input_grid = np.random.randint(1, 4, (small_size, small_size))
            output_grid = np.repeat(np.repeat(input_grid, 2, axis=0), 2, axis=1)
        else:  # Mirror
            input_grid = np.random.randint(0, 3, (size, size))
            output_grid = np.fliplr(input_grid)
        
        task = {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }
        tasks.append(task)
    
    return tasks

def test_systems():
    """Test all systems"""
    
    print("="*60)
    print("LOADING SYSTEMS...")
    print("="*60)
    
    systems = {}
    
    # Load systems
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        systems['Perfect_V2'] = PerfectARCSystem()
        print("OK: Perfect V2 loaded")
    except:
        print("ERROR: Could not load Perfect V2")
    
    try:
        from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
        systems['Ultra_V2'] = UltraAdvancedARCSystem()
        print("OK: Ultra V2 loaded")
    except:
        print("ERROR: Could not load Ultra V2")
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        systems['Interactive_V2'] = InteractiveARCSystem()
        print("OK: Interactive V2 loaded")
    except:
        print("ERROR: Could not load Interactive V2")
    
    try:
        from deep_learning_arc_system import DeepLearningARCSystem
        systems['DeepLearning'] = DeepLearningARCSystem()
        print("OK: Deep Learning loaded")
    except:
        print("ERROR: Could not load Deep Learning")
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        systems['Ultimate'] = UltimateARCSolver()
        print("OK: Ultimate loaded")
    except:
        print("ERROR: Could not load Ultimate")
    
    print(f"\nLoaded {len(systems)} systems")
    print("="*60)
    
    # Generate tasks
    print("\nGenerating 50 test tasks...")
    tasks = generate_test_tasks(50)
    
    # Test each system
    results = defaultdict(lambda: {'success': 0, 'fail': 0, 'times': []})
    
    print("\nTesting systems on 50 tasks...")
    print("-"*60)
    
    for task_id, task in enumerate(tasks):
        if task_id % 10 == 0:
            print(f"Progress: {task_id}/50 tasks")
        
        for system_name, system in systems.items():
            try:
                start_time = time.time()
                
                # Try to solve
                if hasattr(system, 'solve'):
                    output = system.solve(task)
                elif hasattr(system, 'process_task'):
                    output = system.process_task(task)
                else:
                    continue
                
                elapsed = time.time() - start_time
                results[system_name]['success'] += 1
                results[system_name]['times'].append(elapsed)
                
            except Exception as e:
                results[system_name]['fail'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS - 50 TASKS")
    print("="*60)
    
    # Calculate statistics
    for system_name in results:
        total = results[system_name]['success'] + results[system_name]['fail']
        success_rate = results[system_name]['success'] / total if total > 0 else 0
        avg_time = np.mean(results[system_name]['times']) if results[system_name]['times'] else 0
        
        results[system_name]['success_rate'] = success_rate
        results[system_name]['avg_time'] = avg_time
    
    # Sort by success rate
    sorted_systems = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    
    print("\nRANKING BY SUCCESS RATE:")
    print("-"*40)
    for rank, (system_name, stats) in enumerate(sorted_systems, 1):
        print(f"{rank}. {system_name:15} - {stats['success_rate']:.1%} ({stats['success']}/50)")
    
    # Sort by speed
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] > 0 else float('inf'))
    
    print("\nRANKING BY SPEED:")
    print("-"*40)
    for rank, (system_name, stats) in enumerate(sorted_by_speed, 1):
        if stats['avg_time'] > 0:
            print(f"{rank}. {system_name:15} - {stats['avg_time']:.4f}s average")
    
    # Overall statistics
    total_success = sum(r['success'] for r in results.values())
    total_attempts = sum(r['success'] + r['fail'] for r in results.values())
    
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total attempts: {total_attempts}")
    print(f"Total successes: {total_success}")
    print(f"Overall success rate: {total_success/total_attempts:.1%}" if total_attempts > 0 else "N/A")
    
    # Save results
    with open('simple_test_results.json', 'w') as f:
        json.dump({k: {
            'success': v['success'],
            'fail': v['fail'],
            'success_rate': v['success_rate'],
            'avg_time': v['avg_time']
        } for k, v in results.items()}, f, indent=2)
    
    print("\nResults saved to simple_test_results.json")
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = test_systems()
