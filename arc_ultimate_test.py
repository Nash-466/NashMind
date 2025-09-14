from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار النظام النهائي على 10 مهام ARC حقيقية
مقارنة الحلول مع الحلول الصحيحة
"""

import json
import numpy as np
from ultimate_ai_system import UltimateAISystem
from arc_learning_solver import ARCLearningSolver
import time

def load_arc_data():
    """تحميل بيانات ARC"""
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    return challenges, solutions

def calculate_accuracy(predicted, actual):
    """حساب دقة الحل"""
    if not predicted or not actual:
        return 0.0
    
    try:
        pred_array = np.array(predicted)
        actual_array = np.array(actual[0])  # الحل الأول من الحلول المتعددة
        
        if pred_array.shape != actual_array.shape:
            return 0.0
        
        correct_cells = np.sum(pred_array == actual_array)
        total_cells = pred_array.size
        
        return correct_cells / total_cells
    except:
        return 0.0

def compare_solutions(predicted, actual):
    """مقارنة مفصلة بين الحل المتوقع والحل الصحيح"""
    accuracy = calculate_accuracy(predicted, actual)
    
    comparison = {
        "accuracy": accuracy,
        "perfect_match": accuracy == 1.0,
        "predicted_shape": np.array(predicted).shape if predicted else None,
        "actual_shape": np.array(actual[0]).shape if actual else None,
        "shape_match": False
    }
    
    if predicted and actual:
        pred_shape = np.array(predicted).shape
        actual_shape = np.array(actual[0]).shape
        comparison["shape_match"] = pred_shape == actual_shape
    
    return comparison

def test_ultimate_ai_on_arc():
    """اختبار النظام النهائي على مهام ARC"""
    
    print("🚀 بدء اختبار النظام النهائي على مهام ARC")
    print("="*80)
    
    # تحميل البيانات
    challenges, solutions = load_arc_data()
    
    # اختيار 10 مهام للاختبار
    task_ids = list(challenges.keys())[:10]
    
    # تهيئة الأنظمة
    ultimate_ai = UltimateAISystem()
    arc_solver = ARCLearningSolver()
    
    results = {
        "ultimate_ai": [],
        "arc_solver": [],
        "task_details": []
    }
    
    print(f"📋 اختبار {len(task_ids)} مهام من ARC")
    print("-"*80)
    
    for i, task_id in enumerate(task_ids, 1):
        print(f"\n🧩 مهمة {i}: {task_id}")
        
        task = challenges[task_id]
        correct_solution = solutions[task_id]
        
        # استخراج بيانات المهمة
        train_examples = task["train"]
        test_input = task["test"][0]["input"]
        
        print(f"📊 أمثلة التدريب: {len(train_examples)}")
        print(f"📐 حجم الاختبار: {np.array(test_input).shape}")
        
        # اختبار النظام النهائي
        print("🧠 اختبار النظام النهائي...")
        start_time = time.time()
        
        try:
            # تدريب النظام على الأمثلة
            for example in train_examples:
                ultimate_ai.real_learning_from_experience(
                    f"ARC Training: {example['input']} -> {example['output']}", 
                    "arc_training"
                )
            
            # حل المهمة
            ultimate_solution = ultimate_ai.ultimate_problem_solving(
                f"ARC Test Problem: {test_input}"
            )
            
            # محاولة استخراج حل من النص
            ultimate_prediction = test_input  # حل افتراضي
            
        except Exception as e:
            print(f"❌ خطأ في النظام النهائي: {e}")
            ultimate_prediction = test_input
            ultimate_solution = {"confidence": 0.0}
        
        ultimate_time = time.time() - start_time
        
        # اختبار نظام ARC المتخصص
        print("🎯 اختبار نظام ARC المتخصص...")
        start_time = time.time()
        
        try:
            # تدريب النظام على الأمثلة
            for example in train_examples:
                arc_solver.learn_from_arc_example(example["input"], example["output"])
            
            # حل المهمة
            arc_result = arc_solver.solve_arc_problem(test_input)
            arc_prediction = arc_result["predicted_output"]
            
        except Exception as e:
            print(f"❌ خطأ في نظام ARC: {e}")
            arc_prediction = test_input
            arc_result = {"confidence": 0.0}
        
        arc_time = time.time() - start_time
        
        # مقارنة النتائج
        ultimate_comparison = compare_solutions(ultimate_prediction, correct_solution)
        arc_comparison = compare_solutions(arc_prediction, correct_solution)
        
        # حفظ النتائج
        task_result = {
            "task_id": task_id,
            "train_examples": len(train_examples),
            "test_shape": np.array(test_input).shape,
            "correct_shape": np.array(correct_solution[0]).shape,
            "ultimate_ai": {
                "prediction": ultimate_prediction,
                "accuracy": ultimate_comparison["accuracy"],
                "perfect_match": ultimate_comparison["perfect_match"],
                "shape_match": ultimate_comparison["shape_match"],
                "confidence": ultimate_solution.get("confidence", 0.0),
                "processing_time": ultimate_time
            },
            "arc_solver": {
                "prediction": arc_prediction,
                "accuracy": arc_comparison["accuracy"],
                "perfect_match": arc_comparison["perfect_match"],
                "shape_match": arc_comparison["shape_match"],
                "confidence": arc_result.get("confidence", 0.0),
                "processing_time": arc_time
            }
        }
        
        results["task_details"].append(task_result)
        results["ultimate_ai"].append(ultimate_comparison["accuracy"])
        results["arc_solver"].append(arc_comparison["accuracy"])
        
        # عرض النتائج
        print(f"📈 النتائج:")
        print(f"   🧠 النظام النهائي: دقة {ultimate_comparison['accuracy']:.2%}, ثقة {ultimate_solution.get('confidence', 0):.2f}")
        print(f"   🎯 نظام ARC: دقة {arc_comparison['accuracy']:.2%}, ثقة {arc_result.get('confidence', 0):.2f}")
        
        if ultimate_comparison["perfect_match"]:
            print("   ✅ النظام النهائي: حل مثالي!")
        if arc_comparison["perfect_match"]:
            print("   ✅ نظام ARC: حل مثالي!")
        
        print(f"   ⏱️ الأوقات: النهائي {ultimate_time:.2f}s، ARC {arc_time:.2f}s")
    
    # تحليل النتائج الإجمالية
    print("\n" + "="*80)
    print("📊 تحليل النتائج الإجمالية")
    print("="*80)
    
    ultimate_avg = np.mean(results["ultimate_ai"])
    arc_avg = np.mean(results["arc_solver"])
    
    ultimate_perfect = sum(1 for task in results["task_details"] if task["ultimate_ai"]["perfect_match"])
    arc_perfect = sum(1 for task in results["task_details"] if task["arc_solver"]["perfect_match"])
    
    ultimate_shape_match = sum(1 for task in results["task_details"] if task["ultimate_ai"]["shape_match"])
    arc_shape_match = sum(1 for task in results["task_details"] if task["arc_solver"]["shape_match"])
    
    print(f"🧠 النظام النهائي:")
    print(f"   📈 متوسط الدقة: {ultimate_avg:.2%}")
    print(f"   ✅ حلول مثالية: {ultimate_perfect}/{len(task_ids)} ({ultimate_perfect/len(task_ids):.1%})")
    print(f"   📐 تطابق الشكل: {ultimate_shape_match}/{len(task_ids)} ({ultimate_shape_match/len(task_ids):.1%})")
    
    print(f"\n🎯 نظام ARC المتخصص:")
    print(f"   📈 متوسط الدقة: {arc_avg:.2%}")
    print(f"   ✅ حلول مثالية: {arc_perfect}/{len(task_ids)} ({arc_perfect/len(task_ids):.1%})")
    print(f"   📐 تطابق الشكل: {arc_shape_match}/{len(task_ids)} ({arc_shape_match/len(task_ids):.1%})")
    
    # مقارنة الأداء
    print(f"\n🏆 المقارنة:")
    if ultimate_avg > arc_avg:
        print(f"   🥇 النظام النهائي أفضل بـ {(ultimate_avg - arc_avg)*100:.1f} نقطة مئوية")
    elif arc_avg > ultimate_avg:
        print(f"   🥇 نظام ARC أفضل بـ {(arc_avg - ultimate_avg)*100:.1f} نقطة مئوية")
    else:
        print(f"   🤝 الأنظمة متساوية في الأداء")
    
    # تفاصيل المهام الناجحة
    print(f"\n📋 تفاصيل المهام:")
    for i, task in enumerate(results["task_details"], 1):
        status_ultimate = "✅" if task["ultimate_ai"]["perfect_match"] else "❌"
        status_arc = "✅" if task["arc_solver"]["perfect_match"] else "❌"
        
        print(f"   {i}. {task['task_id']}: النهائي {status_ultimate} ({task['ultimate_ai']['accuracy']:.1%}), "
              f"ARC {status_arc} ({task['arc_solver']['accuracy']:.1%})")
    
    # حفظ النتائج
    with open('arc_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 تم حفظ النتائج في arc_test_results.json")
    
    return results

if __name__ == "__main__":
    results = test_ultimate_ai_on_arc()
    
    print("\n🎊 انتهى الاختبار!")
    print("="*80)
