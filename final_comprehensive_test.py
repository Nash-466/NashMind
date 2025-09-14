from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
الاختبار الشامل النهائي لنظام NashMind المحسن
"""

import json
import numpy as np
import time
from datetime import datetime

def run_final_test():
    """تشغيل الاختبار الشامل النهائي"""
    
    print("🚀 الاختبار الشامل النهائي لنظام NashMind")
    print("="*60)
    print(f"📅 التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # تحميل البيانات
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    # اختيار 10 مهام متنوعة للاختبار الشامل
    test_tasks = [
        "007bbfb7",  # مهمة التوسيع التي حققنا فيها 100%
        "00d62c1b",  # مهمة أخرى
        "025d127b",  # مهمة ثالثة
        "045e512c",  # مهمة رابعة
        "0520fde7",  # مهمة خامسة
    ]
    
    # التأكد من وجود المهام
    available_tasks = [task for task in test_tasks if task in challenges]
    if len(available_tasks) < len(test_tasks):
        # إضافة مهام إضافية من القائمة
        all_tasks = list(challenges.keys())
        for task in all_tasks:
            if task not in available_tasks and len(available_tasks) < 10:
                available_tasks.append(task)
    
    test_tasks = available_tasks[:10]
    
    print(f"📊 اختبار {len(test_tasks)} مهام متنوعة")
    
    # تهيئة النظام
    print("\n🧠 تهيئة NashMind...")
    try:
        from NashMind.aces_system import ACESSystem
        nashmind = ACESSystem()
        print("✅ تم تهيئة NashMind بنجاح")
    except Exception as e:
        print(f"❌ فشل في تهيئة NashMind: {e}")
        return
    
    results = []
    start_time = time.time()
    
    for i, task_id in enumerate(test_tasks):
        print(f"\n🎯 اختبار المهمة {i+1}/{len(test_tasks)}: {task_id}")
        print("-" * 40)
        
        task_data = challenges[task_id]
        correct_solutions = solutions[task_id]
        
        # تعلم من أمثلة التدريب
        print(f"📚 تعلم من {len(task_data['train'])} أمثلة تدريب...")
        for j, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            
            # تعلم من المثال
            nashmind.true_learning_engine.learn_from_experience(
                f"ARC_final_{task_id}_{j}",
                {
                    'input': input_grid,
                    'output': output_grid,
                    'task_id': task_id,
                    'example_type': 'training'
                }
            )
        
        # حل مثال الاختبار
        test_input = task_data['test'][0]['input']
        correct_output = correct_solutions[0]
        
        input_shape = np.array(test_input).shape
        output_shape = np.array(correct_output).shape
        
        print(f"  📋 الدخل: {input_shape}")
        print(f"  📋 الخرج المطلوب: {output_shape}")
        
        # حل المسألة
        task_start = time.time()
        result = nashmind.solve_arc_problem(test_input)
        task_time = time.time() - task_start
        
        if result and 'output' in result:
            predicted_output = result['output']
            strategy = result.get('strategy', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # حساب الدقة
            accuracy = calculate_accuracy(predicted_output, correct_output)
            
            print(f"  🎯 الاستراتيجية: {strategy}")
            print(f"  🎯 الثقة: {confidence:.3f}")
            print(f"  📊 الدقة: {accuracy:.1f}%")
            print(f"  ⏱️ الوقت: {task_time:.2f} ثانية")
            
            success = accuracy > 90
            if success:
                print(f"  ✅ نجح!")
            else:
                print(f"  ❌ فشل")
            
            results.append({
                'task_id': task_id,
                'strategy': strategy,
                'confidence': confidence,
                'accuracy': accuracy,
                'success': success,
                'time': task_time,
                'input_shape': input_shape,
                'output_shape': output_shape
            })
        else:
            print(f"  ❌ فشل في إنتاج حل")
            results.append({
                'task_id': task_id,
                'strategy': 'none',
                'confidence': 0.0,
                'accuracy': 0.0,
                'success': False,
                'time': task_time,
                'input_shape': input_shape,
                'output_shape': output_shape
            })
    
    total_time = time.time() - start_time
    
    # تحليل النتائج النهائي
    print("\n" + "="*60)
    print("📊 التحليل النهائي للنتائج")
    print("="*60)
    
    successful_tasks = [r for r in results if r['success']]
    total_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"✅ المهام الناجحة: {len(successful_tasks)}/{len(results)}")
    print(f"📈 متوسط الدقة: {total_accuracy:.1f}%")
    print(f"🎯 معدل النجاح: {len(successful_tasks)/len(results)*100:.1f}%")
    print(f"⏱️ متوسط الوقت: {avg_time:.2f} ثانية/مهمة")
    print(f"⏱️ الوقت الإجمالي: {total_time:.2f} ثانية")
    
    # تحليل الاستراتيجيات
    strategy_stats = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {
                'count': 0, 'success': 0, 'total_accuracy': 0, 'total_time': 0
            }
        
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['total_accuracy'] += result['accuracy']
        strategy_stats[strategy]['total_time'] += result['time']
        if result['success']:
            strategy_stats[strategy]['success'] += 1
    
    print(f"\n🔍 إحصائيات الاستراتيجيات:")
    for strategy, stats in strategy_stats.items():
        avg_accuracy = stats['total_accuracy'] / stats['count']
        avg_time = stats['total_time'] / stats['count']
        success_rate = stats['success'] / stats['count'] * 100
        print(f"  {strategy}:")
        print(f"    الاستخدام: {stats['count']} مرة")
        print(f"    النجاح: {success_rate:.0f}%")
        print(f"    الدقة: {avg_accuracy:.1f}%")
        print(f"    الوقت: {avg_time:.2f}s")
    
    # حفظ النتائج النهائية
    final_report = {
        'test_info': {
            'date': datetime.now().isoformat(),
            'total_tasks': len(results),
            'total_time': total_time
        },
        'results': results,
        'summary': {
            'successful_tasks': len(successful_tasks),
            'success_rate': len(successful_tasks)/len(results)*100,
            'average_accuracy': total_accuracy,
            'average_time_per_task': avg_time
        },
        'strategy_stats': strategy_stats
    }
    
    with open('final_comprehensive_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 تم حفظ التقرير النهائي في: final_comprehensive_test_results.json")
    
    # تقييم الأداء العام
    print(f"\n🏆 التقييم النهائي:")
    if len(successful_tasks)/len(results) >= 0.5:
        print("🌟 أداء ممتاز! النظام يحل أكثر من 50% من المهام")
    elif len(successful_tasks)/len(results) >= 0.3:
        print("👍 أداء جيد! النظام يحل أكثر من 30% من المهام")
    elif len(successful_tasks)/len(results) >= 0.1:
        print("📈 أداء مقبول! النظام يحل أكثر من 10% من المهام")
    else:
        print("🔧 يحتاج تحسين! النظام يحل أقل من 10% من المهام")
    
    return results

def calculate_accuracy(predicted, correct):
    """حساب دقة الحل"""
    try:
        pred_array = np.array(predicted)
        correct_array = np.array(correct)
        
        if pred_array.shape != correct_array.shape:
            return 0.0
        
        matches = np.sum(pred_array == correct_array)
        total = pred_array.size
        
        return (matches / total) * 100
    except:
        return 0.0

if __name__ == "__main__":
    run_final_test()
