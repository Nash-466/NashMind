from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار حقيقي على مهام التقييم مع مقارنة الحلول
"""

import os
import json
import numpy as np
import time
from collections.abc import Callable
from typing import Dict, List, Any

def load_evaluation_data():
    """تحميل بيانات التقييم والحلول"""
    
    print("📂 تحميل بيانات التقييم...")
    
    # مسارات الملفات
    eval_path = 'ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json'
    solutions_path = 'ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json'
    
    # تحميل مهام التقييم
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_tasks = json.load(f)
        print(f"✅ تم تحميل {len(eval_tasks)} مهمة تقييم")
    else:
        print("❌ ملف مهام التقييم غير موجود")
        return None, None
    
    # تحميل الحلول الرسمية
    if os.path.exists(solutions_path):
        with open(solutions_path, 'r') as f:
            official_solutions = json.load(f)
        print(f"✅ تم تحميل {len(official_solutions)} حل رسمي")
    else:
        print("❌ ملف الحلول الرسمية غير موجود")
        return eval_tasks, None
    
    return eval_tasks, official_solutions

def test_system_on_evaluation(system_name, system_class, eval_tasks, official_solutions, max_tasks=10):
    """اختبار نظام على مهام التقييم الحقيقية"""
    
    print(f"\n🧪 اختبار {system_name} على مهام التقييم الحقيقية")
    print("=" * 60)
    
    try:
        system = system_class()
        print(f"✅ تم إنشاء {system_name}")
    except Exception as e:
        print(f"❌ فشل إنشاء {system_name}: {e}")
        return {'system': system_name, 'error': str(e), 'correct': 0, 'total': 0}
    
    results = {
        'system': system_name,
        'correct': 0,
        'total': 0,
        'details': [],
        'execution_time': 0
    }
    
    task_ids = list(eval_tasks.keys())[:max_tasks]
    start_time = time.time()
    
    for i, task_id in enumerate(task_ids):
        print(f"\n📋 مهمة {i+1}/{len(task_ids)}: {task_id}")
        
        task = eval_tasks[task_id]
        official_solution = official_solutions.get(task_id, []) if official_solutions else []
        
        try:
            # حل المهمة
            task_start = time.time()
            solutions = system.solve_task(task)
            task_time = time.time() - task_start
            
            # مقارنة مع الحل الرسمي
            is_correct = False
            if solutions and official_solution:
                is_correct = compare_solutions(solutions, official_solution)
            
            if is_correct:
                results['correct'] += 1
                print(f"   ✅ صحيح ({task_time:.2f}s)")
            else:
                print(f"   ❌ خطأ ({task_time:.2f}s)")
            
            results['details'].append({
                'task_id': task_id,
                'correct': is_correct,
                'time': task_time,
                'has_solution': len(solutions) > 0 if solutions else False
            })
            
        except Exception as e:
            print(f"   💥 خطأ: {str(e)[:50]}...")
            results['details'].append({
                'task_id': task_id,
                'correct': False,
                'error': str(e),
                'time': 0
            })
        
        results['total'] += 1
        
        # عرض التقدم
        accuracy = (results['correct'] / results['total']) * 100
        print(f"   📊 التقدم: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
    
    results['execution_time'] = time.time() - start_time
    accuracy = (results['correct'] / results['total']) * 100 if results['total'] > 0 else 0
    
    print(f"\n📈 النتيجة النهائية لـ {system_name}:")
    print(f"   ✅ صحيح: {results['correct']}")
    print(f"   ❌ خطأ: {results['total'] - results['correct']}")
    print(f"   🎯 الدقة: {accuracy:.1f}%")
    print(f"   ⏱️ الوقت الإجمالي: {results['execution_time']:.2f}s")
    
    return results

def compare_solutions(system_solutions, official_solutions):
    """مقارنة حلول النظام مع الحلول الرسمية"""
    
    if not system_solutions or not official_solutions:
        return False
    
    if len(system_solutions) != len(official_solutions):
        return False
    
    for sys_sol, off_sol in zip(system_solutions, official_solutions):
        if sys_sol is None or not isinstance(sys_sol, np.ndarray):
            return False
        
        official_array = np.array(off_sol)
        
        if not np.array_equal(sys_sol, official_array):
            return False
    
    return True

def run_comprehensive_evaluation():
    """تشغيل التقييم الشامل"""
    
    print("🚀 بدء التقييم الشامل على مهام ARC الحقيقية")
    print("=" * 70)
    
    # تحميل البيانات
    eval_tasks, official_solutions = load_evaluation_data()
    
    if not eval_tasks:
        print("❌ لا يمكن المتابعة بدون بيانات التقييم")
        return
    
    # الأنظمة للاختبار
    systems_to_test = [
        ('final_arc_system', 'FinalARCSystem'),
        ('arc_clean_integrated_system', 'ARCCleanIntegratedSystem'),
        ('arc_ultimate_perfect_system', 'ARCUltimatePerfectSolver'),
    ]
    
    all_results = []
    
    for system_file, class_name in systems_to_test:
        print(f"\n{'='*70}")
        print(f"🎯 اختبار النظام: {system_file}")
        print(f"{'='*70}")
        
        try:
            # استيراد النظام
            module = __import__(system_file)
            system_class = getattr(module, class_name)
            
            # اختبار النظام
            result = test_system_on_evaluation(
                system_file, system_class, eval_tasks, official_solutions, max_tasks=5
            )
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ فشل في اختبار {system_file}: {e}")
            all_results.append({
                'system': system_file,
                'error': str(e),
                'correct': 0,
                'total': 0
            })
    
    # النتائج المقارنة
    print(f"\n{'='*70}")
    print("📊 مقارنة النتائج النهائية")
    print(f"{'='*70}")
    
    print(f"{'النظام':<35} {'صحيح':<8} {'إجمالي':<8} {'دقة%':<8} {'وقت(s)':<10}")
    print("-" * 70)
    
    best_system = None
    best_accuracy = 0
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['system']:<35} {'خطأ':<8} {'خطأ':<8} {'0.0':<8} {'N/A':<10}")
        else:
            accuracy = (result['correct'] / result['total']) * 100 if result['total'] > 0 else 0
            exec_time = result.get('execution_time', 0)
            
            print(f"{result['system']:<35} {result['correct']:<8} {result['total']:<8} {accuracy:<8.1f} {exec_time:<10.2f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_system = result['system']
    
    print(f"\n🏆 أفضل نظام: {best_system} بدقة {best_accuracy:.1f}%")
    
    # حفظ النتائج
    timestamp = int(time.time())
    filename = f'real_evaluation_results_{timestamp}.json'
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 تم حفظ النتائج في: {filename}")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
