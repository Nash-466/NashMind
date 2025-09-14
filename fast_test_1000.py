from __future__ import annotations
"""
FAST TEST - 1000 TASKS WITH SOLUTIONS (OPTIMIZED)
==================================================
اختبار سريع ومحسّن لـ 1000 مهمة
"""

import numpy as np
import time
import json
import os
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# تعطيل التسجيل
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def generate_synthetic_tasks(count):
    """Generate synthetic tasks quickly"""
    tasks = []
    
    for i in range(count):
        np.random.seed(i)
        size = np.random.randint(3, 8)
        
        # توليد أنماط بسيطة
        task_type = i % 8
        
        if task_type == 0:  # Rotation
            input_grid = np.random.randint(0, 4, (size, size))
            output_grid = np.rot90(input_grid)
        elif task_type == 1:  # Flip
            input_grid = np.random.randint(0, 4, (size, size))
            output_grid = np.fliplr(input_grid)
        elif task_type == 2:  # Transpose
            input_grid = np.random.randint(0, 4, (size, size))
            output_grid = input_grid.T
        elif task_type == 3:  # Color change
            input_grid = np.random.randint(0, 3, (size, size))
            output_grid = (input_grid + 1) % 4
        elif task_type == 4:  # Fill
            input_grid = np.zeros((size, size), dtype=int)
            input_grid[0, :] = 1
            input_grid[-1, :] = 1
            output_grid = np.ones((size, size), dtype=int)
        elif task_type == 5:  # Diagonal
            input_grid = np.zeros((size, size), dtype=int)
            np.fill_diagonal(input_grid, 2)
            output_grid = input_grid.copy()
            output_grid = output_grid + input_grid.T
        elif task_type == 6:  # Border
            input_grid = np.zeros((size, size), dtype=int)
            output_grid = np.zeros((size, size), dtype=int)
            output_grid[0, :] = 1
            output_grid[-1, :] = 1
            output_grid[:, 0] = 1
            output_grid[:, -1] = 1
        else:  # Copy
            input_grid = np.random.randint(0, 3, (size, size))
            output_grid = input_grid.copy()
        
        task = {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ]
        }
        
        tasks.append({
            'data': task,
            'solution': output_grid.tolist(),
            'name': f'synthetic_{i}'
        })
    
    return tasks

def compare_solutions(predicted, actual):
    """Fast comparison"""
    try:
        if predicted is None or actual is None:
            return 0.0
        
        pred = np.array(predicted) if not isinstance(predicted, np.ndarray) else predicted
        act = np.array(actual) if not isinstance(actual, np.ndarray) else actual
        
        if pred.shape != act.shape:
            return 0.0
        
        if np.array_equal(pred, act):
            return 1.0
        
        # نسبة التطابق
        matches = np.sum(pred == act)
        return matches / act.size
    except:
        return 0.0

def test_fast_1000():
    """Fast test for 1000 tasks"""
    
    print("="*70)
    print("FAST TEST - 1000 TASKS WITH SOLUTION COMPARISON")
    print("="*70)
    
    # تحميل الأنظمة السريعة فقط
    print("\nLoading fast systems only...")
    systems = {}
    
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        systems['Perfect_V2'] = PerfectARCSystem()
        print("✓ Perfect V2 loaded")
    except Exception as e:
        print(f"✗ Perfect V2: {e}")
    
    try:
        from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
        systems['Ultra_V2'] = UltraAdvancedARCSystem()
        print("✓ Ultra V2 loaded")
    except Exception as e:
        print(f"✗ Ultra V2: {e}")
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        systems['Interactive_V2'] = InteractiveARCSystem()
        print("✓ Interactive V2 loaded")
    except Exception as e:
        print(f"✗ Interactive V2: {e}")
    
    # تخطي Deep Learning لأنه بطيء
    print("⏭️  Skipping Deep Learning (too slow)")
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        systems['Ultimate'] = UltimateARCSolver()
        print("✓ Ultimate loaded")
    except Exception as e:
        print(f"✗ Ultimate: {e}")
    
    if not systems:
        print("No systems loaded!")
        return
    
    print(f"\nLoaded {len(systems)} fast systems")
    
    # توليد 1000 مهمة بسرعة
    print("\nGenerating 1000 synthetic tasks...")
    all_tasks = generate_synthetic_tasks(1000)
    print(f"Generated {len(all_tasks)} tasks")
    
    # نتائج
    results = defaultdict(lambda: {
        'perfect': 0,
        'partial': 0,
        'failed': 0,
        'errors': 0,
        'times': [],
        'accuracies': []
    })
    
    # اختبار سريع
    print("\nRunning fast test...")
    print("-"*70)
    
    start_time = time.time()
    timeout_per_task = 0.5  # نصف ثانية كحد أقصى لكل مهمة
    
    for task_idx, task_info in enumerate(all_tasks):
        if task_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {task_idx}/1000 | Time: {elapsed:.1f}s")
        
        task = task_info['data']
        solution = task_info['solution']
        
        for system_name, system in systems.items():
            try:
                # تطبيق timeout
                task_start = time.time()
                
                # محاولة الحل
                if hasattr(system, 'solve'):
                    predicted = system.solve(task)
                elif hasattr(system, 'process_task'):
                    predicted = system.process_task(task)
                else:
                    predicted = None
                
                task_time = time.time() - task_start
                
                # تخطي إذا تجاوز الوقت
                if task_time > timeout_per_task:
                    results[system_name]['failed'] += 1
                    continue
                
                results[system_name]['times'].append(task_time)
                
                # مقارنة
                accuracy = compare_solutions(predicted, solution)
                results[system_name]['accuracies'].append(accuracy)
                
                if accuracy >= 1.0:
                    results[system_name]['perfect'] += 1
                elif accuracy > 0.5:
                    results[system_name]['partial'] += 1
                else:
                    results[system_name]['failed'] += 1
                    
            except Exception as e:
                results[system_name]['errors'] += 1
                results[system_name]['failed'] += 1
    
    total_time = time.time() - start_time
    
    # النتائج
    print("\n" + "="*70)
    print("FAST TEST RESULTS - 1000 TASKS")
    print("="*70)
    print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
    
    # ترتيب النتائج
    system_scores = []
    for system_name, stats in results.items():
        total = stats['perfect'] + stats['partial'] + stats['failed']
        perfect_rate = stats['perfect'] / total if total > 0 else 0
        avg_accuracy = np.mean(stats['accuracies']) if stats['accuracies'] else 0
        avg_time = np.mean(stats['times']) if stats['times'] else 0
        
        system_scores.append({
            'name': system_name,
            'perfect': stats['perfect'],
            'partial': stats['partial'],
            'failed': stats['failed'],
            'errors': stats['errors'],
            'perfect_rate': perfect_rate,
            'avg_accuracy': avg_accuracy,
            'avg_time': avg_time,
            'total': total
        })
    
    # ترتيب حسب معدل النجاح
    system_scores.sort(key=lambda x: x['perfect_rate'], reverse=True)
    
    print("\n🏆 SYSTEM RANKINGS:")
    print("-"*70)
    
    for rank, score in enumerate(system_scores, 1):
        print(f"\n{rank}. {score['name']}")
        print(f"   ✅ Perfect: {score['perfect']}/{score['total']} ({score['perfect_rate']:.1%})")
        print(f"   ⚠️  Partial: {score['partial']}/{score['total']}")
        print(f"   ❌ Failed: {score['failed']}/{score['total']}")
        print(f"   📊 Avg accuracy: {score['avg_accuracy']:.1%}")
        print(f"   ⏱️  Avg time: {score['avg_time']*1000:.0f}ms")
        if score['errors'] > 0:
            print(f"   ⚠️  Errors: {score['errors']}")
    
    # ملخص
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_perfect = sum(s['perfect'] for s in system_scores)
    total_tests = sum(s['total'] for s in system_scores)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total tests: {total_tests}")
    print(f"   Perfect matches: {total_perfect}")
    print(f"   Success rate: {total_perfect/total_tests:.1%}")
    print(f"   Test duration: {total_time:.1f}s")
    print(f"   Avg per task: {total_time/1000:.3f}s")
    
    if system_scores:
        best = system_scores[0]
        print(f"\n🏆 WINNER: {best['name']}")
        print(f"   Perfect match rate: {best['perfect_rate']:.1%}")
        print(f"   Average accuracy: {best['avg_accuracy']:.1%}")
    
    # حفظ النتائج
    with open('fast_test_1000_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': total_time,
            'task_count': 1000,
            'systems': len(systems),
            'results': system_scores
        }, f, indent=2)
    
    print("\n📁 Results saved to fast_test_1000_results.json")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = test_fast_1000()
