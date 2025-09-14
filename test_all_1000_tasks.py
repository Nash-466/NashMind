from __future__ import annotations
"""
COMPLETE TEST - ALL 1000 TASKS WITH REAL ACCURACY
==================================================
اختبار شامل لجميع الـ 1000 مهمة
"""

import json
import numpy as np
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.ERROR)
for logger in logging.root.manager.loggerDict:
    logging.getLogger(logger).setLevel(logging.ERROR)

def calculate_similarity(pred, actual):
    """حساب نسبة التطابق"""
    try:
        if pred is None or actual is None:
            return 0.0
        
        pred_arr = np.array(pred)
        actual_arr = np.array(actual)
        
        if pred_arr.shape != actual_arr.shape:
            return 0.0
        
        matches = np.sum(pred_arr == actual_arr)
        total = actual_arr.size
        
        return (matches / total) * 100
    except:
        return 0.0

def test_all_1000_tasks():
    print("="*80)
    print("🎯 TESTING ALL 1000 ARC TASKS - REAL ACCURACY")
    print("="*80)
    
    # تحميل المهام والحلول
    print("\nLoading tasks and solutions...")
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        print(f"✓ Loaded {len(challenges)} tasks with solutions")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # تحميل الأنظمة
    print("\nLoading systems...")
    systems = {}
    
    try:
        from perfect_arc_system_v2 import PerfectARCSystem
        systems['Perfect_V2'] = PerfectARCSystem()
        print("✓ Perfect V2 loaded")
    except:
        pass
    
    try:
        from ultimate_arc_solver import UltimateARCSolver
        systems['Ultimate'] = UltimateARCSolver()
        print("✓ Ultimate loaded")
    except:
        pass
    
    try:
        from interactive_arc_system_v2 import InteractiveARCSystem
        systems['Interactive_V2'] = InteractiveARCSystem()
        print("✓ Interactive V2 loaded")
    except:
        pass
    
    # Ultra V2 يحتاج إصلاح، نتخطاه الآن
    # try:
    #     from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
    #     systems['Ultra_V2'] = UltraAdvancedARCSystem()
    #     print("✓ Ultra V2 loaded")
    # except:
    #     pass
    
    print(f"\nTotal systems loaded: {len(systems)}")
    
    if not systems:
        print("❌ No systems loaded!")
        return
    
    # إحصائيات لكل نظام
    results = {name: {
        'perfect': 0,
        'high_partial': 0,  # >80%
        'medium_partial': 0,  # 50-80%
        'low_partial': 0,  # <50%
        'wrong': 0,
        'errors': 0,
        'total_similarity': 0,
        'task_times': [],
        'similarities': []
    } for name in systems.keys()}
    
    print("\n" + "="*80)
    print("TESTING IN PROGRESS...")
    print("="*80)
    
    total_tasks = len(challenges)
    start_time = time.time()
    
    for idx, (task_id, task) in enumerate(challenges.items()):
        # تقدم كل 100 مهمة
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"\nProgress: {idx}/{total_tasks} tasks ({idx/total_tasks*100:.1f}%)")
            if idx > 0:
                eta = (elapsed / idx) * (total_tasks - idx)
                print(f"Time elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        # الحل الصحيح
        correct_solution = solutions[task_id][0]  # أول حل
        
        # اختبار كل نظام
        for system_name, system in systems.items():
            try:
                task_start = time.time()
                
                # حل المهمة
                if hasattr(system, 'solve'):
                    result = system.solve(task)
                elif hasattr(system, 'process_task'):
                    result = system.process_task(task)
                else:
                    result = None
                
                task_time = time.time() - task_start
                results[system_name]['task_times'].append(task_time)
                
                # استخراج المخرج
                if result is not None:
                    if isinstance(result, dict) and 'output' in result:
                        predicted = result['output']
                    else:
                        predicted = result
                    
                    # حساب التشابه
                    similarity = calculate_similarity(predicted, correct_solution)
                    results[system_name]['similarities'].append(similarity)
                    results[system_name]['total_similarity'] += similarity
                    
                    # تصنيف النتيجة
                    if similarity == 100:
                        results[system_name]['perfect'] += 1
                    elif similarity >= 80:
                        results[system_name]['high_partial'] += 1
                    elif similarity >= 50:
                        results[system_name]['medium_partial'] += 1
                    elif similarity > 0:
                        results[system_name]['low_partial'] += 1
                    else:
                        results[system_name]['wrong'] += 1
                else:
                    results[system_name]['wrong'] += 1
                    results[system_name]['similarities'].append(0)
                    
            except Exception as e:
                results[system_name]['errors'] += 1
                results[system_name]['similarities'].append(0)
    
    total_time = time.time() - start_time
    
    # حساب الإحصائيات النهائية
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - 1000 TASKS")
    print("="*80)
    print(f"\n⏱️ Total test time: {total_time:.1f} seconds")
    
    # ترتيب الأنظمة
    system_scores = []
    for system_name, stats in results.items():
        avg_similarity = stats['total_similarity'] / total_tasks
        avg_time = np.mean(stats['task_times']) if stats['task_times'] else 0
        
        # نسب النجاح
        perfect_rate = (stats['perfect'] / total_tasks) * 100
        partial_rate = ((stats['high_partial'] + stats['medium_partial'] + stats['low_partial']) / total_tasks) * 100
        failure_rate = ((stats['wrong'] + stats['errors']) / total_tasks) * 100
        
        system_scores.append({
            'name': system_name,
            'avg_similarity': avg_similarity,
            'perfect': stats['perfect'],
            'perfect_rate': perfect_rate,
            'high_partial': stats['high_partial'],
            'medium_partial': stats['medium_partial'],
            'low_partial': stats['low_partial'],
            'partial_rate': partial_rate,
            'wrong': stats['wrong'],
            'errors': stats['errors'],
            'failure_rate': failure_rate,
            'avg_time': avg_time
        })
    
    # ترتيب حسب متوسط التشابه
    system_scores.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    print("\n🏆 SYSTEM RANKING BY REAL ACCURACY:")
    print("="*80)
    
    for rank, score in enumerate(system_scores, 1):
        print(f"\n{rank}. {score['name']}")
        print(f"   📊 Average Similarity: {score['avg_similarity']:.1f}%")
        print(f"   ✅ Perfect Solutions: {score['perfect']}/1000 ({score['perfect_rate']:.1f}%)")
        print(f"   ⚠️  Partial Solutions:")
        print(f"      • High (>80%): {score['high_partial']} tasks")
        print(f"      • Medium (50-80%): {score['medium_partial']} tasks")
        print(f"      • Low (<50%): {score['low_partial']} tasks")
        print(f"      • Total Partial: {score['partial_rate']:.1f}%")
        print(f"   ❌ Failed: {score['wrong']} tasks")
        if score['errors'] > 0:
            print(f"   💥 Errors: {score['errors']} tasks")
        print(f"   ⏱️  Avg Time: {score['avg_time']*1000:.1f}ms per task")
    
    # ملخص عام
    print("\n" + "="*80)
    print("💡 OVERALL SUMMARY")
    print("="*80)
    
    if system_scores:
        best = system_scores[0]
        print(f"\n🥇 BEST SYSTEM: {best['name']}")
        print(f"   • Average Accuracy: {best['avg_similarity']:.1f}%")
        print(f"   • Perfect Solutions: {best['perfect']}/1000")
        print(f"   • Tasks with >50% accuracy: {best['perfect'] + best['high_partial'] + best['medium_partial']}/1000")
        
        # تقييم الأداء
        if best['avg_similarity'] >= 70:
            print("\n✨ EXCELLENT! Systems show strong understanding of patterns")
        elif best['avg_similarity'] >= 50:
            print("\n📈 GOOD! Systems understand many patterns but need refinement")
        elif best['avg_similarity'] >= 30:
            print("\n⚠️ MODERATE! Systems capture some patterns but need significant improvement")
        else:
            print("\n❌ POOR! Systems need major redesign")
    
    # حفظ النتائج
    with open('final_1000_tasks_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tasks': total_tasks,
            'test_duration': total_time,
            'systems': system_scores
        }, f, indent=2)
    
    print("\n📁 Results saved to final_1000_tasks_results.json")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = test_all_1000_tasks()
