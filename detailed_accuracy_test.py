from __future__ import annotations
"""
DETAILED ACCURACY TEST - Check partial solutions
=================================================
"""

import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.ERROR)
for logger in logging.root.manager.loggerDict:
    logging.getLogger(logger).setLevel(logging.ERROR)

def calculate_similarity(pred, actual):
    """حساب نسبة التشابه بين حلين"""
    try:
        if pred is None or actual is None:
            return 0.0
        
        pred_arr = np.array(pred)
        actual_arr = np.array(actual)
        
        # إذا الأحجام مختلفة، نسبة التشابه 0
        if pred_arr.shape != actual_arr.shape:
            return 0.0
        
        # حساب نسبة العناصر المتطابقة
        matches = np.sum(pred_arr == actual_arr)
        total = actual_arr.size
        
        return (matches / total) * 100
    except:
        return 0.0

def test_detailed_accuracy():
    print("="*80)
    print("🔬 DETAILED ACCURACY ANALYSIS")
    print("="*80)
    
    # تحميل المهام والحلول
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        print(f"✓ Loaded {len(challenges)} challenges and solutions")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # تحميل الأنظمة
    systems = {}
    
    print("\nTrying to load systems with error handling...")
    
    # محاولة تحميل كل نظام مع معالجة أفضل للأخطاء
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        system = PerfectARCSystem()
        systems['Perfect_V2'] = system
        print("✓ Perfect V2 loaded")
    except Exception as e:
        print(f"✗ Perfect V2 error: {str(e)[:50]}")
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        system = UltimateARCSolver()
        systems['Ultimate'] = system
        print("✓ Ultimate loaded")
    except Exception as e:
        print(f"✗ Ultimate error: {str(e)[:50]}")
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        system = InteractiveARCSystem()
        systems['Interactive_V2'] = system
        print("✓ Interactive V2 loaded")
    except Exception as e:
        print(f"✗ Interactive V2 error: {str(e)[:50]}")
    
    if not systems:
        print("❌ No systems could be loaded!")
        return
    
    print(f"\n{len(systems)} systems loaded successfully")
    
    # اختبار 10 مهام فقط بتفصيل أكثر
    print("\n" + "="*80)
    print("TESTING 10 TASKS IN DETAIL")
    print("="*80)
    
    task_list = list(challenges.items())[:10]
    
    all_results = {}
    
    for task_idx, (task_id, task) in enumerate(task_list):
        print(f"\n📋 Task {task_idx+1}/10: {task_id}")
        print("-"*40)
        
        # الحل الصحيح
        correct_solution = solutions[task_id][0]  # أول حل اختباري (الحل مخزن مباشرة كقائمة)
        solution_shape = np.array(correct_solution).shape
        print(f"   Solution shape: {solution_shape}")
        
        task_results = {}
        
        for system_name, system in systems.items():
            try:
                # محاولة الحل
                print(f"   Testing {system_name}...", end=" ")
                
                if hasattr(system, 'solve'):
                    result = system.solve(task)
                elif hasattr(system, 'process_task'):
                    result = system.process_task(task)
                else:
                    result = None
                
                # استخراج المخرج
                if result is not None:
                    if isinstance(result, dict) and 'output' in result:
                        predicted = result['output']
                    else:
                        predicted = result
                    
                    # حساب التشابه
                    similarity = calculate_similarity(predicted, correct_solution)
                    
                    pred_shape = np.array(predicted).shape if predicted is not None else "None"
                    
                    if similarity == 100:
                        print(f"✅ CORRECT! (100% match)")
                    elif similarity > 0:
                        print(f"⚠️  Partial: {similarity:.1f}% match")
                    else:
                        print(f"❌ Wrong (shape: {pred_shape} vs {solution_shape})")
                    
                    task_results[system_name] = {
                        'similarity': similarity,
                        'shape': str(pred_shape)
                    }
                else:
                    print("❌ No output")
                    task_results[system_name] = {
                        'similarity': 0,
                        'shape': 'None'
                    }
                    
            except Exception as e:
                print(f"💥 Error: {str(e)[:30]}")
                task_results[system_name] = {
                    'similarity': 0,
                    'shape': 'Error'
                }
        
        all_results[task_id] = task_results
    
    # ملخص النتائج
    print("\n" + "="*80)
    print("📊 SUMMARY OF PARTIAL SOLUTIONS")
    print("="*80)
    
    system_stats = {name: {
        'perfect': 0,
        'partial': 0,
        'wrong': 0,
        'errors': 0,
        'total_similarity': 0
    } for name in systems.keys()}
    
    for task_id, task_results in all_results.items():
        for system_name, result in task_results.items():
            sim = result['similarity']
            if result['shape'] == 'Error':
                system_stats[system_name]['errors'] += 1
            elif sim == 100:
                system_stats[system_name]['perfect'] += 1
            elif sim > 0:
                system_stats[system_name]['partial'] += 1
            else:
                system_stats[system_name]['wrong'] += 1
            system_stats[system_name]['total_similarity'] += sim
    
    print("\nSystem Performance:")
    print("-"*40)
    
    for system_name, stats in system_stats.items():
        avg_sim = stats['total_similarity'] / 10
        print(f"\n{system_name}:")
        print(f"   ✅ Perfect solutions: {stats['perfect']}/10")
        print(f"   ⚠️  Partial solutions: {stats['partial']}/10")
        print(f"   ❌ Wrong solutions: {stats['wrong']}/10")
        print(f"   💥 Errors: {stats['errors']}/10")
        print(f"   📊 Average similarity: {avg_sim:.1f}%")
    
    # أفضل نظام
    best_system = max(system_stats.items(), key=lambda x: x[1]['total_similarity'])
    
    print("\n" + "="*80)
    print("💡 FINDINGS")
    print("="*80)
    
    if best_system[1]['total_similarity'] > 0:
        print(f"\n🏆 Best performing: {best_system[0]}")
        print(f"   Average similarity: {best_system[1]['total_similarity']/10:.1f}%")
        if best_system[1]['partial'] > 0:
            print(f"   ✨ Has {best_system[1]['partial']} partial solutions!")
            print("   This shows the system understands some patterns")
    else:
        print("\n⚠️ No system achieved even partial success")
        print("   All systems are producing completely wrong outputs")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_detailed_accuracy()
