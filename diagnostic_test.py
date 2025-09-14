from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار تشخيصي لفهم مشكلة حل مسائل ARC
"""

import json
import sys
import numpy as np

def load_single_task():
    """تحميل مهمة واحدة للاختبار"""
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        # اختيار مهمة بسيطة
        task_id = "007bbfb7"  # مهمة 3x3 بسيطة
        
        if task_id in challenges and task_id in solutions:
            return task_id, challenges[task_id], solutions[task_id]
        
        return None, None, None
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        return None, None, None

def test_single_task():
    """اختبار مهمة واحدة بالتفصيل"""
    
    print("🔍 اختبار تشخيصي - مهمة واحدة")
    print("="*50)
    
    # تحميل المهمة
    task_id, task_data, solution_data = load_single_task()
    if not task_id:
        print("❌ فشل في تحميل المهمة")
        return
    
    print(f"📋 المهمة: {task_id}")
    print(f"📊 عدد أمثلة التدريب: {len(task_data['train'])}")
    print(f"📊 عدد أمثلة الاختبار: {len(task_data['test'])}")
    
    # عرض أمثلة التدريب
    print("\n🎓 أمثلة التدريب:")
    for i, example in enumerate(task_data['train']):
        input_grid = example['input']
        output_grid = example['output']
        print(f"  مثال {i+1}:")
        print(f"    الدخل: {input_grid}")
        print(f"    الخرج: {output_grid}")
    
    # عرض مثال الاختبار
    test_example = task_data['test'][0]
    test_input = test_example['input']
    correct_output = solution_data[0]
    
    print(f"\n🧪 مثال الاختبار:")
    print(f"  الدخل: {test_input}")
    print(f"  الخرج الصحيح: {correct_output}")
    
    # تهيئة NashMind
    print(f"\n🧠 تهيئة NashMind...")
    try:
        sys.path.append('NashMind')
        from aces_system import ACES
        nashmind = ACES()
        print("✅ تم تهيئة NashMind بنجاح")
    except Exception as e:
        print(f"❌ فشل في تهيئة NashMind: {e}")
        return
    
    # تعلم من أمثلة التدريب
    print(f"\n📚 تعلم من أمثلة التدريب...")
    for i, example in enumerate(task_data['train']):
        try:
            result = nashmind.real_learning_from_experience(
                f"ARC_diagnostic_{task_id}_{i}",
                {
                    "input": example['input'],
                    "output": example['output'],
                    "type": "ARC_training_example"
                }
            )
            print(f"  ✅ تعلم من المثال {i+1}")
        except Exception as e:
            print(f"  ❌ خطأ في التعلم من المثال {i+1}: {e}")
    
    # حل مثال الاختبار
    print(f"\n🎯 حل مثال الاختبار...")
    try:
        result = nashmind.solve_arc_problem(task_data['train'], test_input)
        
        print(f"📊 نتيجة الحل:")
        print(f"  الاستراتيجية: {result.get('strategy_used', 'غير محدد')}")
        print(f"  الثقة: {result.get('confidence', 0.0):.3f}")
        print(f"  الأمثلة المشابهة: {result.get('similar_examples_found', 0)}")
        
        predicted_output = result.get('predicted_output', test_input)
        print(f"  الخرج المتوقع: {predicted_output}")
        
        # مقارنة مع الحل الصحيح
        print(f"\n📊 مقارنة النتائج:")
        print(f"  الخرج الصحيح: {correct_output}")
        print(f"  الخرج المتوقع: {predicted_output}")
        
        # حساب الدقة
        accuracy = calculate_accuracy(predicted_output, correct_output)
        print(f"  الدقة: {accuracy:.1%}")
        
        if accuracy == 1.0:
            print("🎉 حل صحيح 100%!")
        elif accuracy > 0.5:
            print("🟡 حل جزئي")
        else:
            print("❌ حل خاطئ")
            
        # تحليل الاختلافات
        if accuracy < 1.0:
            analyze_differences(predicted_output, correct_output)
        
    except Exception as e:
        print(f"❌ خطأ في حل المهمة: {e}")
        import traceback
        traceback.print_exc()

def calculate_accuracy(predicted, actual):
    """حساب دقة الحل"""
    try:
        predicted_array = np.array(predicted)
        actual_array = np.array(actual)
        
        if predicted_array.shape != actual_array.shape:
            return 0.0
        
        correct = np.sum(predicted_array == actual_array)
        total = predicted_array.size
        
        return correct / total if total > 0 else 0.0
    except:
        return 0.0

def analyze_differences(predicted, actual):
    """تحليل الاختلافات بين الحل المتوقع والصحيح"""
    try:
        print(f"\n🔍 تحليل الاختلافات:")
        
        predicted_array = np.array(predicted)
        actual_array = np.array(actual)
        
        if predicted_array.shape != actual_array.shape:
            print(f"  ❌ اختلاف في الحجم:")
            print(f"    المتوقع: {predicted_array.shape}")
            print(f"    الصحيح: {actual_array.shape}")
            return
        
        differences = np.where(predicted_array != actual_array)
        num_differences = len(differences[0])
        
        print(f"  📊 عدد الخلايا المختلفة: {num_differences}")
        
        if num_differences > 0 and num_differences <= 10:
            print(f"  📍 مواقع الاختلافات:")
            for i in range(min(num_differences, 5)):  # عرض أول 5 اختلافات
                row, col = differences[0][i], differences[1][i]
                predicted_val = predicted_array[row, col]
                actual_val = actual_array[row, col]
                print(f"    ({row}, {col}): متوقع={predicted_val}, صحيح={actual_val}")
        
    except Exception as e:
        print(f"  ❌ خطأ في تحليل الاختلافات: {e}")

if __name__ == "__main__":
    test_single_task()
