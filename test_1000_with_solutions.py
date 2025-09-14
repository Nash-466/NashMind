from __future__ import annotations
"""
TEST 1000 TASKS WITH SOLUTION COMPARISON
=========================================
اختبار 1000 مهمة مع مقارنة الحلول الحقيقية
"""

import numpy as np
import time
import json
import os
import glob
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# تعطيل رسائل التسجيل المزعجة
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def load_arc_tasks_from_files():
    """Load ARC tasks from JSON files in the directory"""
    tasks = []
    
    # البحث عن ملفات ARC
    patterns = [
        'arc*.json',
        'training/*.json',
        'evaluation/*.json',
        'test/*.json',
        'data/*.json'
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file_path in files:
            if any(skip in file_path for skip in ['result', 'report', 'test_', 'simple_']):
                continue
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # إذا كان الملف يحتوي على مهمة واحدة
                    if 'train' in data and 'test' in data:
                        tasks.append({'data': data, 'file': file_path})
                    
                    # إذا كان الملف يحتوي على عدة مهام
                    elif isinstance(data, dict):
                        for task_name, task_data in data.items():
                            if isinstance(task_data, dict) and 'train' in task_data:
                                tasks.append({
                                    'data': task_data, 
                                    'file': file_path,
                                    'name': task_name
                                })
            except:
                pass
    
    return tasks

def generate_synthetic_tasks(count):
    """Generate synthetic tasks for testing"""
    tasks = []
    
    for i in range(count):
        np.random.seed(i)
        size = np.random.randint(3, 10)
        
        # أنواع مختلفة من المهام
        task_type = i % 10
        
        if task_type == 0:  # Rotation
            input_grid = np.random.randint(0, 5, (size, size))
            output_grid = np.rot90(input_grid, k=1)
        elif task_type == 1:  # Flip horizontal
            input_grid = np.random.randint(0, 5, (size, size))
            output_grid = np.fliplr(input_grid)
        elif task_type == 2:  # Flip vertical
            input_grid = np.random.randint(0, 5, (size, size))
            output_grid = np.flipud(input_grid)
        elif task_type == 3:  # Color increment
            input_grid = np.random.randint(0, 4, (size, size))
            output_grid = (input_grid + 1) % 5
        elif task_type == 4:  # Transpose
            input_grid = np.random.randint(0, 5, (size, size))
            output_grid = input_grid.T
        elif task_type == 5:  # Invert colors
            input_grid = np.random.randint(1, 5, (size, size))
            output_grid = 5 - input_grid
        elif task_type == 6:  # Scale 2x
            small_size = size // 2
            input_grid = np.random.randint(0, 5, (small_size, small_size))
            output_grid = np.repeat(np.repeat(input_grid, 2, axis=0), 2, axis=1)
        elif task_type == 7:  # Fill pattern
            input_grid = np.zeros((size, size), dtype=int)
            input_grid[0, :] = 1
            input_grid[-1, :] = 1
            input_grid[:, 0] = 1
            input_grid[:, -1] = 1
            output_grid = input_grid.copy()
            output_grid[size//2, size//2] = 2
        elif task_type == 8:  # Diagonal
            input_grid = np.zeros((size, size), dtype=int)
            np.fill_diagonal(input_grid, np.random.randint(1, 5))
            output_grid = input_grid.copy()
            np.fill_diagonal(np.fliplr(output_grid), np.random.randint(1, 5))
        else:  # Random pattern
            input_grid = np.random.randint(0, 3, (size, size))
            output_grid = np.random.randint(0, 3, (size, size))
        
        task = {
            'data': {
                'train': [
                    {'input': input_grid.tolist(), 'output': output_grid.tolist()}
                ],
                'test': [
                    {'input': input_grid.tolist(), 'output': output_grid.tolist()}
                ]
            },
            'name': f'synthetic_{i}',
            'solution': output_grid
        }
        
        tasks.append(task)
    
    return tasks

def compare_solutions(predicted, actual):
    """Compare predicted solution with actual solution"""
    
    # تحويل إلى numpy arrays
    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)
    if not isinstance(actual, np.ndarray):
        actual = np.array(actual)
    
    # التحقق من التطابق الكامل
    if predicted.shape == actual.shape:
        if np.array_equal(predicted, actual):
            return 1.0  # نجاح كامل
        else:
            # حساب نسبة التطابق
            matches = np.sum(predicted == actual)
            total = actual.size
            return matches / total
    else:
        # أحجام مختلفة = فشل
        return 0.0

def test_all_systems_on_1000_tasks():
    """Test all systems on 1000 tasks and compare with solutions"""
    
    print("="*70)
    print("TESTING ALL SYSTEMS ON 1000 TASKS WITH SOLUTION COMPARISON")
    print("="*70)
    
    # تحميل الأنظمة
    print("\nLoading systems...")
    systems = {}
    
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        systems['Perfect_V2'] = PerfectARCSystem()
        print("✓ Perfect V2 loaded")
    except:
        print("✗ Could not load Perfect V2")
    
    try:
        from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
        systems['Ultra_V2'] = UltraAdvancedARCSystem()
        print("✓ Ultra V2 loaded")
    except:
        print("✗ Could not load Ultra V2")
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        systems['Interactive_V2'] = InteractiveARCSystem()
        print("✓ Interactive V2 loaded")
    except:
        print("✗ Could not load Interactive V2")
    
    try:
        from deep_learning_arc_system import DeepLearningARCSystem
        systems['DeepLearning'] = DeepLearningARCSystem()
        print("✓ Deep Learning loaded")
    except:
        print("✗ Could not load Deep Learning")
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        systems['Ultimate'] = UltimateARCSolver()
        print("✓ Ultimate loaded")
    except:
        print("✗ Could not load Ultimate")
    
    print(f"\nLoaded {len(systems)} systems")
    
    # تحميل المهام
    print("\nLoading tasks...")
    
    # محاولة تحميل مهام حقيقية
    real_tasks = load_arc_tasks_from_files()
    print(f"Found {len(real_tasks)} real ARC tasks")
    
    # إنشاء مهام اصطناعية للوصول إلى 1000
    if len(real_tasks) < 1000:
        synthetic_count = 1000 - len(real_tasks)
        print(f"Generating {synthetic_count} synthetic tasks...")
        synthetic_tasks = generate_synthetic_tasks(synthetic_count)
        all_tasks = real_tasks + synthetic_tasks
    else:
        all_tasks = real_tasks[:1000]
    
    print(f"Total tasks to test: {len(all_tasks)}")
    
    # إحصائيات لكل نظام
    results = defaultdict(lambda: {
        'total': 0,
        'perfect_match': 0,
        'partial_match': 0,
        'failed': 0,
        'errors': 0,
        'accuracy_scores': [],
        'times': []
    })
    
    # اختبار كل مهمة
    print("\nTesting in progress...")
    print("-"*70)
    
    for task_idx, task_info in enumerate(all_tasks):
        if task_idx % 100 == 0:
            print(f"Progress: {task_idx}/{len(all_tasks)} tasks completed")
        
        task = task_info['data']
        
        # الحصول على الحل الصحيح
        if 'solution' in task_info:
            correct_solution = task_info['solution']
        elif 'test' in task and len(task['test']) > 0 and 'output' in task['test'][0]:
            correct_solution = np.array(task['test'][0]['output'])
        else:
            # لا يوجد حل للمقارنة
            correct_solution = None
        
        # اختبار كل نظام
        for system_name, system in systems.items():
            results[system_name]['total'] += 1
            
            try:
                start_time = time.time()
                
                # حل المهمة
                if hasattr(system, 'solve'):
                    predicted_solution = system.solve(task)
                elif hasattr(system, 'process_task'):
                    predicted_solution = system.process_task(task)
                else:
                    predicted_solution = None
                
                elapsed_time = time.time() - start_time
                results[system_name]['times'].append(elapsed_time)
                
                # مقارنة الحلول
                if predicted_solution is not None and correct_solution is not None:
                    accuracy = compare_solutions(predicted_solution, correct_solution)
                    results[system_name]['accuracy_scores'].append(accuracy)
                    
                    if accuracy == 1.0:
                        results[system_name]['perfect_match'] += 1
                    elif accuracy > 0.5:
                        results[system_name]['partial_match'] += 1
                    else:
                        results[system_name]['failed'] += 1
                else:
                    results[system_name]['failed'] += 1
                    
            except Exception as e:
                results[system_name]['errors'] += 1
                results[system_name]['failed'] += 1
    
    # طباعة النتائج النهائية
    print("\n" + "="*70)
    print("FINAL RESULTS - 1000 TASKS")
    print("="*70)
    
    # ترتيب الأنظمة حسب الأداء
    system_scores = []
    
    for system_name, stats in results.items():
        perfect_rate = stats['perfect_match'] / stats['total'] if stats['total'] > 0 else 0
        partial_rate = stats['partial_match'] / stats['total'] if stats['total'] > 0 else 0
        fail_rate = stats['failed'] / stats['total'] if stats['total'] > 0 else 0
        avg_accuracy = np.mean(stats['accuracy_scores']) if stats['accuracy_scores'] else 0
        avg_time = np.mean(stats['times']) if stats['times'] else 0
        
        system_scores.append({
            'name': system_name,
            'perfect': stats['perfect_match'],
            'partial': stats['partial_match'],
            'failed': stats['failed'],
            'errors': stats['errors'],
            'perfect_rate': perfect_rate,
            'avg_accuracy': avg_accuracy,
            'avg_time': avg_time
        })
    
    # ترتيب حسب معدل النجاح الكامل
    system_scores.sort(key=lambda x: x['perfect_rate'], reverse=True)
    
    print("\n🏆 SYSTEM RANKINGS BY PERFECT MATCH RATE:")
    print("-"*70)
    
    for rank, score in enumerate(system_scores, 1):
        print(f"\n{rank}. {score['name']}")
        print(f"   ✅ Perfect matches: {score['perfect']} / 1000 ({score['perfect_rate']:.1%})")
        print(f"   ⚠️  Partial matches: {score['partial']} / 1000")
        print(f"   ❌ Failed: {score['failed']} / 1000")
        print(f"   📊 Average accuracy: {score['avg_accuracy']:.2%}")
        print(f"   ⏱️  Average time: {score['avg_time']:.3f}s")
        
        if score['errors'] > 0:
            print(f"   ⚠️  Errors encountered: {score['errors']}")
    
    # ملخص عام
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    total_perfect = sum(s['perfect'] for s in system_scores)
    total_partial = sum(s['partial'] for s in system_scores)
    total_failed = sum(s['failed'] for s in system_scores)
    total_tests = len(all_tasks) * len(systems)
    
    print(f"\n📊 Aggregate Statistics:")
    print(f"   Total tests run: {total_tests}")
    print(f"   Total perfect matches: {total_perfect}")
    print(f"   Total partial matches: {total_partial}")
    print(f"   Total failures: {total_failed}")
    print(f"   Overall success rate: {total_perfect/total_tests:.1%}")
    
    # أفضل نظام
    if system_scores:
        best_system = system_scores[0]
        print(f"\n🏆 BEST PERFORMING SYSTEM: {best_system['name']}")
        print(f"   With {best_system['perfect_rate']:.1%} perfect match rate")
        print(f"   And {best_system['avg_accuracy']:.1%} average accuracy")
    
    # حفظ النتائج
    with open('test_1000_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tasks': len(all_tasks),
            'systems_tested': list(systems.keys()),
            'results': {
                system_name: {
                    'perfect_matches': stats['perfect'],
                    'partial_matches': stats['partial'],
                    'failed': stats['failed'],
                    'errors': stats['errors'],
                    'perfect_rate': stats['perfect_rate'],
                    'avg_accuracy': stats['avg_accuracy'],
                    'avg_time': stats['avg_time']
                }
                for system_name, stats in zip(results.keys(), system_scores)
            }
        }, f, indent=2)
    
    print("\n📁 Results saved to test_1000_results.json")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = test_all_systems_on_1000_tasks()
