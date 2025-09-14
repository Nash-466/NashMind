from __future__ import annotations
"""
COMPLETE SYSTEM TEST - ALL SYSTEMS ON ALL AVAILABLE TASKS
==========================================================
اختبار شامل لجميع الأنظمة على جميع المهام الموجودة
"""

import numpy as np
import json
import time
import os
import glob
from collections import defaultdict
import traceback
import warnings
warnings.filterwarnings('ignore')

# تعطيل رسائل التسجيل
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

def find_all_arc_tasks():
    """Find all ARC tasks in the project directory"""
    all_tasks = []
    task_files = []
    
    # البحث في المجلد الحالي
    os.chdir(r"C:\Users\Lenovo\OneDrive\Desktop\مشروع برهان")
    
    # أنماط البحث
    patterns = [
        "*.json",
        "training/*.json", 
        "evaluation/*.json",
        "test/*.json",
        "data/*.json",
        "arc_tasks/*.json",
        "tasks/*.json"
    ]
    
    print("Searching for ARC task files...")
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            # تخطي ملفات النتائج والتقارير
            skip_keywords = ['result', 'report', 'output', 'solution', 'fast_test', 
                           'simple_test', 'test_50', 'test_100', 'test_1000']
            if any(keyword in file_path.lower() for keyword in skip_keywords):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # تحقق من أن الملف يحتوي على مهام ARC
                    if isinstance(data, dict):
                        # ملف مهمة واحدة
                        if 'train' in data and isinstance(data['train'], list):
                            all_tasks.append({
                                'file': file_path,
                                'name': os.path.basename(file_path).replace('.json', ''),
                                'data': data
                            })
                            task_files.append(file_path)
                            
                        # ملف يحتوي على عدة مهام
                        else:
                            for task_id, task_data in data.items():
                                if isinstance(task_data, dict) and 'train' in task_data:
                                    all_tasks.append({
                                        'file': file_path,
                                        'name': task_id,
                                        'data': task_data
                                    })
                            if all_tasks and all_tasks[-1]['file'] == file_path:
                                task_files.append(file_path)
                                
                    # قائمة من المهام
                    elif isinstance(data, list):
                        for idx, task in enumerate(data):
                            if isinstance(task, dict) and 'train' in task:
                                all_tasks.append({
                                    'file': file_path,
                                    'name': f"{os.path.basename(file_path)}_{idx}",
                                    'data': task
                                })
                        if all_tasks and all_tasks[-1]['file'] == file_path:
                            task_files.append(file_path)
                            
            except Exception as e:
                continue
    
    print(f"Found {len(task_files)} task files")
    print(f"Total tasks found: {len(all_tasks)}")
    
    return all_tasks

def load_all_systems():
    """Load all available systems"""
    systems = {}
    
    print("\nLoading all systems...")
    
    # قائمة الأنظمة المتاحة
    system_modules = [
        ('perfect_arc_system_v2', 'PerfectARCSystem', 'Perfect_V2'),
        ('ultra_advanced_arc_system_v2', 'UltraAdvancedARCSystem', 'Ultra_V2'),
        ('interactive_arc_system_v2', 'InteractiveARCSystem', 'Interactive_V2'),
        ('deep_learning_arc_system', 'DeepLearningARCSystem', 'DeepLearning'),
        ('ultimate_arc_solver', 'UltimateARCSolver', 'Ultimate'),
        ('perfect_arc_system', 'PerfectARCSystem', 'Perfect_V1'),
        ('ultra_advanced_arc_system', 'UltraAdvancedARCSystem', 'Ultra_V1'),
        ('interactive_arc_system', 'InteractiveARCSystem', 'Interactive_V1'),
    ]
    
    for module_name, class_name, system_name in system_modules:
        try:
            module = __import__(module_name)
            system_class = getattr(module, class_name)
            systems[system_name] = system_class()
            print(f"  ✓ {system_name} loaded successfully")
        except Exception as e:
            print(f"  ✗ {system_name} failed: {str(e)[:50]}")
    
    print(f"\nTotal systems loaded: {len(systems)}")
    return systems

def solve_task_with_system(system, task, timeout=5.0):
    """Solve a task with a system (with timeout)"""
    try:
        start_time = time.time()
        
        # محاولة حل المهمة
        result = None
        if hasattr(system, 'solve'):
            result = system.solve(task)
        elif hasattr(system, 'process_task'):
            result = system.process_task(task)
        elif hasattr(system, 'predict'):
            result = system.predict(task)
        
        elapsed = time.time() - start_time
        
        # تحقق من timeout
        if elapsed > timeout:
            return None, elapsed, "Timeout"
            
        return result, elapsed, None
        
    except Exception as e:
        return None, 0, str(e)[:100]

def compare_outputs(output1, output2):
    """Compare two outputs"""
    try:
        if output1 is None or output2 is None:
            return 0.0
            
        arr1 = np.array(output1)
        arr2 = np.array(output2)
        
        if arr1.shape != arr2.shape:
            return 0.0
            
        if np.array_equal(arr1, arr2):
            return 1.0
            
        # حساب نسبة التطابق
        matches = np.sum(arr1 == arr2)
        total = arr1.size
        return matches / total
        
    except:
        return 0.0

def run_complete_test():
    """Run complete test on all systems and tasks"""
    
    print("="*80)
    print("COMPLETE SYSTEM TEST - ALL SYSTEMS ON ALL AVAILABLE TASKS")
    print("="*80)
    
    # تحميل جميع المهام
    tasks = find_all_arc_tasks()
    
    if not tasks:
        print("No tasks found! Please check the directory.")
        return
    
    # تحميل جميع الأنظمة
    systems = load_all_systems()
    
    if not systems:
        print("No systems loaded! Please check the implementations.")
        return
    
    # إحصائيات
    results = defaultdict(lambda: {
        'total': 0,
        'solved': 0,
        'failed': 0,
        'errors': 0,
        'timeouts': 0,
        'times': [],
        'accuracies': [],
        'task_results': {}
    })
    
    # مقارنة بين الأنظمة
    system_agreements = defaultdict(int)
    
    print("\n" + "="*80)
    print("TESTING IN PROGRESS")
    print("="*80)
    
    total_tests = len(tasks) * len(systems)
    current_test = 0
    
    for task_idx, task_info in enumerate(tasks):
        task_data = task_info['data']
        task_name = task_info['name']
        
        # حفظ نتائج كل نظام لهذه المهمة
        task_solutions = {}
        
        # طباعة التقدم
        if task_idx % 10 == 0:
            print(f"\nProgress: {task_idx}/{len(tasks)} tasks")
            print(f"Current task: {task_name}")
        
        for system_name, system in systems.items():
            current_test += 1
            results[system_name]['total'] += 1
            
            # حل المهمة
            solution, elapsed, error = solve_task_with_system(system, task_data)
            
            if error:
                if "Timeout" in error:
                    results[system_name]['timeouts'] += 1
                else:
                    results[system_name]['errors'] += 1
                results[system_name]['failed'] += 1
                results[system_name]['task_results'][task_name] = 'Failed'
            elif solution is not None:
                results[system_name]['solved'] += 1
                results[system_name]['times'].append(elapsed)
                results[system_name]['task_results'][task_name] = 'Solved'
                task_solutions[system_name] = solution
            else:
                results[system_name]['failed'] += 1
                results[system_name]['task_results'][task_name] = 'Failed'
        
        # مقارنة الحلول بين الأنظمة
        if len(task_solutions) > 1:
            system_names = list(task_solutions.keys())
            for i in range(len(system_names)):
                for j in range(i+1, len(system_names)):
                    similarity = compare_outputs(
                        task_solutions[system_names[i]], 
                        task_solutions[system_names[j]]
                    )
                    if similarity > 0.9:  # تطابق عالي
                        key = f"{system_names[i]}-{system_names[j]}"
                        system_agreements[key] += 1
    
    # طباعة النتائج
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # ترتيب الأنظمة حسب الأداء
    system_scores = []
    for system_name, stats in results.items():
        if stats['total'] > 0:
            success_rate = stats['solved'] / stats['total']
            avg_time = np.mean(stats['times']) if stats['times'] else 0
            
            system_scores.append({
                'name': system_name,
                'solved': stats['solved'],
                'failed': stats['failed'],
                'errors': stats['errors'],
                'timeouts': stats['timeouts'],
                'total': stats['total'],
                'success_rate': success_rate,
                'avg_time': avg_time
            })
    
    # ترتيب حسب معدل النجاح
    system_scores.sort(key=lambda x: x['success_rate'], reverse=True)
    
    print("\n🏆 SYSTEM PERFORMANCE RANKING:")
    print("-"*80)
    
    for rank, score in enumerate(system_scores, 1):
        print(f"\n{rank}. {score['name']}")
        print(f"   ✅ Solved: {score['solved']}/{score['total']} ({score['success_rate']:.1%})")
        print(f"   ❌ Failed: {score['failed']}/{score['total']}")
        if score['errors'] > 0:
            print(f"   ⚠️  Errors: {score['errors']}")
        if score['timeouts'] > 0:
            print(f"   ⏱️  Timeouts: {score['timeouts']}")
        print(f"   ⏰ Avg time: {score['avg_time']:.3f}s")
    
    # ملخص الاتفاق بين الأنظمة
    print("\n" + "="*80)
    print("SYSTEM AGREEMENT ANALYSIS")
    print("="*80)
    
    if system_agreements:
        print("\nSystems with highest agreement:")
        sorted_agreements = sorted(system_agreements.items(), key=lambda x: x[1], reverse=True)
        for pair, count in sorted_agreements[:5]:
            print(f"  {pair}: {count} tasks with similar solutions")
    
    # إحصائيات عامة
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    total_solved = sum(s['solved'] for s in system_scores)
    print(f"\n📊 Summary:")
    print(f"   Total tasks tested: {len(tasks)}")
    print(f"   Total systems tested: {len(systems)}")
    print(f"   Total tests performed: {total_tests}")
    print(f"   Total successful solutions: {total_solved}")
    print(f"   Overall success rate: {total_solved/total_tests:.1%}")
    
    # أفضل نظام
    if system_scores:
        best = system_scores[0]
        print(f"\n🏆 BEST PERFORMING SYSTEM: {best['name']}")
        print(f"   Success rate: {best['success_rate']:.1%}")
        print(f"   Tasks solved: {best['solved']}/{best['total']}")
    
    # حفظ النتائج التفصيلية
    detailed_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'task_count': len(tasks),
        'system_count': len(systems),
        'total_tests': total_tests,
        'systems': system_scores,
        'agreements': dict(system_agreements),
        'task_files': list(set(t['file'] for t in tasks))
    }
    
    with open('complete_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print("\n📁 Detailed results saved to complete_test_results.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = run_complete_test()
