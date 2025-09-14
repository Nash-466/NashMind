from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار شامل لنظام NashMind على عينة كبيرة من مهام ARC
"""

import json
import numpy as np
import time
from datetime import datetime
import sys
import os

def run_comprehensive_test():
    """تشغيل اختبار شامل على 20 مهمة ARC"""
    
    print("🚀 الاختبار الشامل لنظام NashMind على 20 مهمة")
    print("="*60)
    print(f"📅 التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # تحميل البيانات
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        print(f"✅ تم تحميل {len(challenges)} مهمة من قاعدة البيانات")
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        return
    
    # اختيار 20 مهمة متنوعة
    all_tasks = list(challenges.keys())
    test_tasks = all_tasks[:20]  # أول 20 مهمة
    
    print(f"📊 سيتم اختبار {len(test_tasks)} مهمة")
    
    # تهيئة النظام
    print("\n🧠 تهيئة نظام NashMind...")
    try:
        # إضافة مسار NashMind
        sys.path.append(os.path.join(os.getcwd(), 'NashMind'))
        from aces_system import ACES
        nashmind = ACES()
        print("✅ تم تهيئة NashMind بنجاح")
    except Exception as e:
        print(f"❌ فشل في تهيئة NashMind: {e}")
        import traceback
        traceback.print_exc()
        return
    
    results = []
    start_time = time.time()
    
    for i, task_id in enumerate(test_tasks):
        print(f"\n🎯 اختبار المهمة {i+1}/{len(test_tasks)}: {task_id}")
        print("-" * 50)
        
        try:
            task_data = challenges[task_id]
            correct_solutions = solutions[task_id]
            
            # معلومات المهمة
            num_train = len(task_data['train'])
            num_test = len(task_data['test'])
            
            print(f"  📚 أمثلة التدريب: {num_train}")
            print(f"  🧪 أمثلة الاختبار: {num_test}")
            
            # تعلم من أمثلة التدريب
            for j, example in enumerate(task_data['train']):
                input_grid = example['input']
                output_grid = example['output']
                
                # تعلم من المثال
                nashmind.true_learning_engine.learn_from_experience(
                    f"ARC_comprehensive_{task_id}_{j}",
                    {
                        'input': input_grid,
                        'output': output_grid,
                        'task_id': task_id,
                        'example_type': 'training'
                    }
                )
            
            # حل مثال الاختبار الأول
            test_input = task_data['test'][0]['input']
            correct_output = correct_solutions[0]
            
            input_shape = np.array(test_input).shape
            output_shape = np.array(correct_output).shape
            
            print(f"  📋 شكل الدخل: {input_shape}")
            print(f"  📋 شكل الخرج المطلوب: {output_shape}")
            
            # حل المسألة مع قياس الوقت
            task_start = time.time()
            result = nashmind.solve_arc_problem(task_data['train'], test_input)
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
                print(f"  ⏱️ الوقت: {task_time:.2f}s")
                
                success = accuracy >= 95.0  # نعتبر 95%+ نجاح
                status = "✅ نجح" if success else f"❌ فشل ({accuracy:.1f}%)"
                print(f"  {status}")
                
                results.append({
                    'task_id': task_id,
                    'strategy': strategy,
                    'confidence': confidence,
                    'accuracy': accuracy,
                    'success': success,
                    'time': task_time,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'num_train_examples': num_train
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
                    'output_shape': output_shape,
                    'num_train_examples': num_train
                })
                
        except Exception as e:
            print(f"  ❌ خطأ في معالجة المهمة: {e}")
            results.append({
                'task_id': task_id,
                'strategy': 'error',
                'confidence': 0.0,
                'accuracy': 0.0,
                'success': False,
                'time': 0.0,
                'input_shape': (0, 0),
                'output_shape': (0, 0),
                'num_train_examples': 0,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # تحليل النتائج الشامل
    print("\n" + "="*60)
    print("📊 النتائج النهائية الشاملة")
    print("="*60)
    
    successful_tasks = [r for r in results if r['success']]
    failed_tasks = [r for r in results if not r['success']]
    
    total_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"✅ المهام الناجحة: {len(successful_tasks)}/{len(results)}")
    print(f"❌ المهام الفاشلة: {len(failed_tasks)}/{len(results)}")
    print(f"🎯 معدل النجاح: {len(successful_tasks)/len(results)*100:.1f}%")
    print(f"📈 متوسط الدقة: {total_accuracy:.1f}%")
    print(f"⏱️ متوسط الوقت: {avg_time:.2f}s/مهمة")
    print(f"⏱️ الوقت الإجمالي: {total_time:.1f}s")
    
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
    for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['success'], reverse=True):
        avg_accuracy = stats['total_accuracy'] / stats['count']
        avg_time = stats['total_time'] / stats['count']
        success_rate = stats['success'] / stats['count'] * 100
        print(f"  📊 {strategy}:")
        print(f"     الاستخدام: {stats['count']} مرة")
        print(f"     النجاح: {success_rate:.0f}% ({stats['success']}/{stats['count']})")
        print(f"     متوسط الدقة: {avg_accuracy:.1f}%")
        print(f"     متوسط الوقت: {avg_time:.2f}s")
    
    # تحليل أنواع المهام
    print(f"\n📋 تحليل أنواع المهام:")
    shape_analysis = {}
    for result in results:
        input_shape = result['input_shape']
        output_shape = result['output_shape']
        shape_key = f"{input_shape} → {output_shape}"
        
        if shape_key not in shape_analysis:
            shape_analysis[shape_key] = {'count': 0, 'success': 0}
        
        shape_analysis[shape_key]['count'] += 1
        if result['success']:
            shape_analysis[shape_key]['success'] += 1
    
    for shape_key, stats in sorted(shape_analysis.items(), key=lambda x: x[1]['success'], reverse=True):
        success_rate = stats['success'] / stats['count'] * 100
        print(f"  📐 {shape_key}: {success_rate:.0f}% ({stats['success']}/{stats['count']})")
    
    # حفظ النتائج التفصيلية
    comprehensive_report = {
        'test_info': {
            'date': datetime.now().isoformat(),
            'total_tasks': len(results),
            'total_time': total_time,
            'test_type': 'comprehensive_20_tasks'
        },
        'results': results,
        'summary': {
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(successful_tasks)/len(results)*100,
            'average_accuracy': total_accuracy,
            'average_time_per_task': avg_time
        },
        'strategy_stats': strategy_stats,
        'shape_analysis': shape_analysis
    }
    
    with open('comprehensive_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 تم حفظ التقرير الشامل في: comprehensive_test_results.json")
    
    # التقييم النهائي
    success_rate = len(successful_tasks)/len(results)*100
    print(f"\n🏆 التقييم النهائي:")
    if success_rate >= 50:
        print("🌟 أداء ممتاز! النظام يحل أكثر من 50% من المهام")
    elif success_rate >= 30:
        print("👍 أداء جيد! النظام يحل أكثر من 30% من المهام")
    elif success_rate >= 15:
        print("📈 أداء مقبول! النظام يحل أكثر من 15% من المهام")
    elif success_rate >= 5:
        print("🔧 يحتاج تحسين! النظام يحل أكثر من 5% من المهام")
    else:
        print("⚠️ أداء ضعيف! النظام يحل أقل من 5% من المهام")
    
    print(f"\n📊 الخلاصة: {len(successful_tasks)} مهمة محلولة من أصل {len(results)} مهمة")
    
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
    run_comprehensive_test()
