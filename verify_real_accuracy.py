from __future__ import annotations
"""
VERIFY REAL ACCURACY - Compare system solutions with correct answers
=====================================================================
التحقق من دقة الحلول الفعلية
"""

import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# تعطيل رسائل التسجيل
import logging
logging.basicConfig(level=logging.ERROR)

def compare_solutions(pred, actual):
    """مقارنة حلين"""
    try:
        if pred is None or actual is None:
            return False
        pred_arr = np.array(pred)
        actual_arr = np.array(actual)
        return np.array_equal(pred_arr, actual_arr)
    except:
        return False

def test_real_accuracy():
    """اختبار الدقة الحقيقية للأنظمة"""
    
    print("="*80)
    print("🔍 VERIFYING REAL ACCURACY AGAINST CORRECT SOLUTIONS")
    print("="*80)
    
    # تحميل المهام والحلول الصحيحة
    print("\nLoading tasks and solutions...")
    
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        print(f"✓ Loaded {len(challenges)} challenges")
    except:
        print("❌ Could not load challenges")
        return
    
    try:
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        print(f"✓ Loaded {len(solutions)} solutions")
    except:
        print("❌ Could not load solutions")
        return
    
    # تحميل الأنظمة
    print("\nLoading systems for testing...")
    systems = {}
    
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        systems['Perfect_V2'] = PerfectARCSystem()
        print("✓ Perfect V2 loaded")
    except:
        pass
    
    try:
        from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
        systems['Ultra_V2'] = UltraAdvancedARCSystem()
        print("✓ Ultra V2 loaded")
    except:
        pass
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        systems['Interactive_V2'] = InteractiveARCSystem()
        print("✓ Interactive V2 loaded")
    except:
        pass
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        systems['Ultimate'] = UltimateARCSolver()
        print("✓ Ultimate loaded")
    except:
        pass
    
    if not systems:
        print("❌ No systems loaded!")
        return
    
    print(f"\n{len(systems)} systems loaded for testing")
    
    # اختبار عينة من المهام
    print("\n" + "="*80)
    print("TESTING SAMPLE TASKS WITH REAL SOLUTIONS")
    print("="*80)
    
    # نتائج كل نظام
    results = {name: {'correct': 0, 'wrong': 0, 'errors': 0} 
              for name in systems.keys()}
    
    # اختبار أول 50 مهمة كعينة
    sample_size = min(50, len(challenges))
    print(f"\nTesting {sample_size} sample tasks...")
    
    for idx, (task_id, task) in enumerate(list(challenges.items())[:sample_size]):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{sample_size}")
        
        # الحل الصحيح
        if task_id in solutions:
            correct_solution = solutions[task_id]
            
            # اختبار كل نظام
            for system_name, system in systems.items():
                try:
                    # حل المهمة
                    if hasattr(system, 'solve'):
                        pred = system.solve(task)
                    elif hasattr(system, 'process_task'):
                        pred = system.process_task(task)
                    else:
                        pred = None
                    
                    # المقارنة مع الحل الصحيح
                    if pred is not None:
                        # إذا كان الناتج dict، استخرج output
                        if isinstance(pred, dict) and 'output' in pred:
                            pred = pred['output']
                        
                        # قارن مع كل حل في test
                        is_correct = False
                        for test_case in correct_solution:
                            if compare_solutions(pred, test_case['output']):
                                is_correct = True
                                break
                        
                        if is_correct:
                            results[system_name]['correct'] += 1
                        else:
                            results[system_name]['wrong'] += 1
                    else:
                        results[system_name]['wrong'] += 1
                        
                except Exception as e:
                    results[system_name]['errors'] += 1
    
    # عرض النتائج
    print("\n" + "="*80)
    print("📊 REAL ACCURACY RESULTS")
    print("="*80)
    
    system_accuracies = []
    for system_name, stats in results.items():
        total = stats['correct'] + stats['wrong'] + stats['errors']
        if total > 0:
            accuracy = (stats['correct'] / total) * 100
            system_accuracies.append((system_name, accuracy, stats))
    
    # ترتيب حسب الدقة
    system_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 SYSTEM RANKING BY REAL ACCURACY:")
    print("-"*80)
    
    for rank, (name, accuracy, stats) in enumerate(system_accuracies, 1):
        total = stats['correct'] + stats['wrong'] + stats['errors']
        print(f"\n{rank}. {name}")
        print(f"   ✅ Correct: {stats['correct']}/{total}")
        print(f"   ❌ Wrong: {stats['wrong']}/{total}")
        if stats['errors'] > 0:
            print(f"   ⚠️  Errors: {stats['errors']}/{total}")
        print(f"   📊 REAL ACCURACY: {accuracy:.1f}%")
    
    # الملخص
    print("\n" + "="*80)
    print("💡 IMPORTANT FINDINGS")
    print("="*80)
    
    if system_accuracies:
        best = system_accuracies[0]
        print(f"\n🥇 Most Accurate System: {best[0]}")
        print(f"   Real Accuracy: {best[1]:.1f}%")
        print(f"   Correctly Solved: {best[2]['correct']} tasks")
        
        if best[1] < 50:
            print("\n⚠️ WARNING: Real accuracy is low!")
            print("The systems are producing outputs but not necessarily correct solutions.")
        elif best[1] < 80:
            print("\n📈 Systems show moderate accuracy.")
            print("There's room for improvement in solving strategies.")
        else:
            print("\n✨ Excellent! Systems show high real accuracy.")
    
    print("\n" + "="*80)
    
    return results

if __name__ == "__main__":
    results = test_real_accuracy()
