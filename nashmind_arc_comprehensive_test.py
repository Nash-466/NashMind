from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار شامل لنظام NashMind على جميع مهام ARC
مع مقارنة النتائج بملف الحلول الصحيحة
"""

import sys
import os
import json
import time
from collections.abc import Callable
from typing import Dict, List, Any, Tuple

# إضافة مسار NashMind
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NashMind'))

def load_arc_data():
    """تحميل بيانات ARC"""
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
            
        return challenges, solutions
    except Exception as e:
        print(f"❌ خطأ في تحميل بيانات ARC: {e}")
        return None, None

def calculate_accuracy(predicted_grid: List[List[int]], actual_grid: List[List[int]]) -> float:
    """حساب دقة الحل"""
    if not predicted_grid or not actual_grid:
        return 0.0
    
    if len(predicted_grid) != len(actual_grid):
        return 0.0
    
    total_cells = 0
    correct_cells = 0
    
    for i in range(len(predicted_grid)):
        if len(predicted_grid[i]) != len(actual_grid[i]):
            return 0.0
        
        for j in range(len(predicted_grid[i])):
            total_cells += 1
            if predicted_grid[i][j] == actual_grid[i][j]:
                correct_cells += 1
    
    return correct_cells / total_cells if total_cells > 0 else 0.0

def test_nashmind_on_arc():
    """اختبار NashMind على جميع مهام ARC"""
    
    print("🚀 بدء الاختبار الشامل لنظام NashMind على مهام ARC")
    print("="*80)
    
    # تحميل البيانات
    challenges, solutions = load_arc_data()
    if not challenges or not solutions:
        return
    
    # تهيئة NashMind
    try:
        from aces_system import ACES
        nashmind = ACES()
        print("✅ تم تهيئة نظام NashMind بنجاح")
    except Exception as e:
        print(f"❌ فشل في تهيئة NashMind: {e}")
        return
    
    # إحصائيات الاختبار
    total_tasks = 0
    successful_tasks = 0
    failed_tasks = 0
    total_accuracy = 0.0
    results = {}
    
    print(f"\n📊 بدء اختبار {len(challenges)} مهمة...")
    print("-" * 80)
    
    # اختبار كل مهمة
    for task_id, task_data in challenges.items():
        total_tasks += 1
        
        print(f"\n🧩 اختبار المهمة {task_id} ({total_tasks}/{len(challenges)})")
        
        try:
            # الحصول على الحل الصحيح
            if task_id not in solutions:
                print(f"⚠️  لا يوجد حل للمهمة {task_id}")
                failed_tasks += 1
                continue
            
            correct_solution = solutions[task_id][0]  # أول حل
            test_input = task_data['test'][0]['input']
            
            # تشغيل NashMind
            start_time = time.time()
            
            # تدريب النظام على الأمثلة
            for example in task_data['train']:
                nashmind.real_learning_from_experience(
                    f"ARC_pattern_{task_id}",
                    f"input: {example['input']}, output: {example['output']}"
                )
            
            # حل المهمة الاختبارية
            predicted_solution = nashmind.solve_arc_problem(test_input)
            
            solve_time = time.time() - start_time
            
            # حساب الدقة
            if predicted_solution and 'solution' in predicted_solution:
                predicted_grid = predicted_solution['solution']
                accuracy = calculate_accuracy(predicted_grid, correct_solution)
                
                if accuracy > 0.8:  # نعتبر الحل ناجحاً إذا كانت الدقة أكبر من 80%
                    successful_tasks += 1
                    status = "✅ نجح"
                else:
                    failed_tasks += 1
                    status = "❌ فشل"
                
                total_accuracy += accuracy
                
                # حفظ النتائج
                results[task_id] = {
                    'accuracy': accuracy,
                    'success': accuracy > 0.8,
                    'solve_time': solve_time,
                    'confidence': predicted_solution.get('confidence', 0.0),
                    'predicted_grid': predicted_grid,
                    'correct_grid': correct_solution
                }
                
                print(f"   {status} - دقة: {accuracy:.1%} - وقت: {solve_time:.2f}ث - ثقة: {predicted_solution.get('confidence', 0.0):.2f}")
                
            else:
                failed_tasks += 1
                print(f"   ❌ فشل - لم يتم إنتاج حل")
                results[task_id] = {
                    'accuracy': 0.0,
                    'success': False,
                    'solve_time': solve_time,
                    'confidence': 0.0,
                    'error': 'No solution generated'
                }
                
        except Exception as e:
            failed_tasks += 1
            print(f"   ❌ خطأ: {e}")
            results[task_id] = {
                'accuracy': 0.0,
                'success': False,
                'solve_time': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    # حساب الإحصائيات النهائية
    average_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    
    # طباعة النتائج النهائية
    print("\n" + "="*80)
    print("📊 النتائج النهائية لاختبار NashMind على مهام ARC")
    print("="*80)
    
    print(f"📈 إجمالي المهام: {total_tasks}")
    print(f"✅ المهام الناجحة: {successful_tasks}")
    print(f"❌ المهام الفاشلة: {failed_tasks}")
    print(f"🎯 معدل النجاح: {success_rate:.1%}")
    print(f"📊 متوسط الدقة: {average_accuracy:.1%}")
    
    # تفاصيل المهام الناجحة
    if successful_tasks > 0:
        print(f"\n🏆 المهام الناجحة ({successful_tasks}):")
        successful_results = {k: v for k, v in results.items() if v['success']}
        for task_id, result in successful_results.items():
            print(f"   • {task_id}: {result['accuracy']:.1%} دقة")
    
    # تفاصيل المهام الفاشلة
    if failed_tasks > 0:
        print(f"\n💔 المهام الفاشلة ({failed_tasks}):")
        failed_results = {k: v for k, v in results.items() if not v['success']}
        for task_id, result in failed_results.items():
            accuracy = result['accuracy']
            if accuracy > 0:
                print(f"   • {task_id}: {accuracy:.1%} دقة (قريب من النجاح)")
            else:
                print(f"   • {task_id}: فشل كامل")
    
    # حفظ النتائج التفصيلية
    with open('nashmind_arc_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 تم حفظ النتائج التفصيلية في: nashmind_arc_test_results.json")
    
    # إحصائيات النظام
    try:
        system_stats = nashmind.get_enhanced_system_stats()
        print(f"\n🧠 إحصائيات النظام بعد الاختبار:")
        print(f"   • التجارب المحفوظة: {system_stats['true_learning']['experiences_count']}")
        print(f"   • الأنماط المكتشفة: {system_stats['true_learning']['patterns_discovered']}")
        print(f"   • مستوى التعلم: {system_stats['true_learning']['overall_learning_level']:.3f}")
        print(f"   • مسائل ARC المحلولة: {system_stats['arc_solving']['problems_solved']}")
    except:
        pass
    
    print("\n🎉 انتهى الاختبار الشامل!")
    
    return {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': success_rate,
        'average_accuracy': average_accuracy,
        'results': results
    }

if __name__ == "__main__":
    test_results = test_nashmind_on_arc()
