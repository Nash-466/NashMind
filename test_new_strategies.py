from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار سريع للاستراتيجيات الجديدة في نظام NashMind
"""

import json
import numpy as np
from NashMind.aces_system import ACESSystem

def test_new_strategies():
    """اختبار الاستراتيجيات الجديدة على 5 مهام"""
    
    print("🧪 اختبار الاستراتيجيات الجديدة")
    print("="*50)
    
    # تحميل البيانات
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    # اختيار 5 مهام للاختبار
    test_tasks = list(challenges.keys())[:5]
    
    print(f"📊 اختبار {len(test_tasks)} مهام")
    
    # تهيئة النظام
    print("\n🧠 تهيئة NashMind...")
    nashmind = ACESSystem()
    print("✅ تم تهيئة NashMind بنجاح")
    
    results = []
    
    for i, task_id in enumerate(test_tasks):
        print(f"\n🎯 اختبار المهمة {i+1}: {task_id}")
        
        task_data = challenges[task_id]
        correct_solutions = solutions[task_id]
        
        # تعلم من أمثلة التدريب
        for j, example in enumerate(task_data['train']):
            input_grid = example['input']
            output_grid = example['output']
            
            # تعلم من المثال
            nashmind.true_learning_engine.learn_from_experience(
                f"ARC_test_{task_id}_{j}",
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
        
        print(f"  📋 الدخل: {np.array(test_input).shape}")
        print(f"  📋 الخرج المطلوب: {np.array(correct_output).shape}")
        
        # حل المسألة
        result = nashmind.solve_arc_problem(test_input)
        
        if result and 'output' in result:
            predicted_output = result['output']
            strategy = result.get('strategy', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # حساب الدقة
            accuracy = calculate_accuracy(predicted_output, correct_output)
            
            print(f"  🎯 الاستراتيجية: {strategy}")
            print(f"  🎯 الثقة: {confidence:.3f}")
            print(f"  📊 الدقة: {accuracy:.1f}%")
            
            results.append({
                'task_id': task_id,
                'strategy': strategy,
                'confidence': confidence,
                'accuracy': accuracy,
                'success': accuracy > 90
            })
        else:
            print(f"  ❌ فشل في إنتاج حل")
            results.append({
                'task_id': task_id,
                'strategy': 'none',
                'confidence': 0.0,
                'accuracy': 0.0,
                'success': False
            })
    
    # تحليل النتائج
    print("\n📊 تحليل النتائج:")
    print("="*50)
    
    successful_tasks = [r for r in results if r['success']]
    total_accuracy = sum(r['accuracy'] for r in results) / len(results)
    
    print(f"✅ المهام الناجحة: {len(successful_tasks)}/{len(results)}")
    print(f"📈 متوسط الدقة: {total_accuracy:.1f}%")
    print(f"🎯 معدل النجاح: {len(successful_tasks)/len(results)*100:.1f}%")
    
    # تحليل الاستراتيجيات
    strategy_stats = {}
    for result in results:
        strategy = result['strategy']
        if strategy not in strategy_stats:
            strategy_stats[strategy] = {'count': 0, 'success': 0, 'total_accuracy': 0}
        
        strategy_stats[strategy]['count'] += 1
        strategy_stats[strategy]['total_accuracy'] += result['accuracy']
        if result['success']:
            strategy_stats[strategy]['success'] += 1
    
    print(f"\n🔍 إحصائيات الاستراتيجيات:")
    for strategy, stats in strategy_stats.items():
        avg_accuracy = stats['total_accuracy'] / stats['count']
        success_rate = stats['success'] / stats['count'] * 100
        print(f"  {strategy}: {stats['count']} مرة، نجاح {success_rate:.0f}%، دقة {avg_accuracy:.1f}%")
    
    # حفظ النتائج
    with open('new_strategies_test_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'summary': {
                'total_tasks': len(results),
                'successful_tasks': len(successful_tasks),
                'success_rate': len(successful_tasks)/len(results)*100,
                'average_accuracy': total_accuracy
            },
            'strategy_stats': strategy_stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 تم حفظ النتائج في: new_strategies_test_results.json")
    
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
    test_new_strategies()
