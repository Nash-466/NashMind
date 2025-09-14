from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تحليل مفصل لنتائج اختبار ARC
"""

import json
import numpy as np
from collections import defaultdict

def load_test_results():
    """تحميل نتائج الاختبار"""
    try:
        with open('arc_test_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ لم يتم العثور على ملف النتائج. يرجى تشغيل arc_ultimate_test.py أولاً")
        return None

def analyze_performance_patterns(results):
    """تحليل أنماط الأداء"""
    
    print("🔍 تحليل أنماط الأداء")
    print("-"*50)
    
    # تحليل حسب عدد أمثلة التدريب
    training_performance = defaultdict(lambda: {"ultimate": [], "arc": []})
    
    for task in results["task_details"]:
        train_count = task["train_examples"]
        training_performance[train_count]["ultimate"].append(task["ultimate_ai"]["accuracy"])
        training_performance[train_count]["arc"].append(task["arc_solver"]["accuracy"])
    
    print("📊 الأداء حسب عدد أمثلة التدريب:")
    for train_count in sorted(training_performance.keys()):
        ultimate_avg = np.mean(training_performance[train_count]["ultimate"])
        arc_avg = np.mean(training_performance[train_count]["arc"])
        
        print(f"   {train_count} أمثلة: النهائي {ultimate_avg:.2%}, ARC {arc_avg:.2%}")
    
    # تحليل حسب حجم المهمة
    size_performance = defaultdict(lambda: {"ultimate": [], "arc": []})
    
    for task in results["task_details"]:
        size = task["test_shape"][0] * task["test_shape"][1]  # إجمالي الخلايا
        size_category = "صغير" if size <= 25 else "متوسط" if size <= 100 else "كبير"
        
        size_performance[size_category]["ultimate"].append(task["ultimate_ai"]["accuracy"])
        size_performance[size_category]["arc"].append(task["arc_solver"]["accuracy"])
    
    print("\n📐 الأداء حسب حجم المهمة:")
    for size_cat in ["صغير", "متوسط", "كبير"]:
        if size_cat in size_performance:
            ultimate_avg = np.mean(size_performance[size_cat]["ultimate"])
            arc_avg = np.mean(size_performance[size_cat]["arc"])
            
            print(f"   {size_cat}: النهائي {ultimate_avg:.2%}, ARC {arc_avg:.2%}")

def analyze_failure_modes(results):
    """تحليل أنماط الفشل"""
    
    print("\n🚨 تحليل أنماط الفشل")
    print("-"*50)
    
    ultimate_failures = []
    arc_failures = []
    
    for task in results["task_details"]:
        if not task["ultimate_ai"]["perfect_match"]:
            ultimate_failures.append({
                "task_id": task["task_id"],
                "accuracy": task["ultimate_ai"]["accuracy"],
                "shape_match": task["ultimate_ai"]["shape_match"],
                "confidence": task["ultimate_ai"]["confidence"]
            })
        
        if not task["arc_solver"]["perfect_match"]:
            arc_failures.append({
                "task_id": task["task_id"],
                "accuracy": task["arc_solver"]["accuracy"],
                "shape_match": task["arc_solver"]["shape_match"],
                "confidence": task["arc_solver"]["confidence"]
            })
    
    print(f"❌ فشل النظام النهائي في {len(ultimate_failures)} مهام:")
    for failure in ultimate_failures:
        shape_status = "✅ شكل صحيح" if failure["shape_match"] else "❌ شكل خاطئ"
        print(f"   {failure['task_id']}: دقة {failure['accuracy']:.2%}, {shape_status}")
    
    print(f"\n❌ فشل نظام ARC في {len(arc_failures)} مهام:")
    for failure in arc_failures:
        shape_status = "✅ شكل صحيح" if failure["shape_match"] else "❌ شكل خاطئ"
        print(f"   {failure['task_id']}: دقة {failure['accuracy']:.2%}, {shape_status}")
    
    # تحليل الثقة مقابل الدقة
    print(f"\n🎯 تحليل الثقة مقابل الدقة:")
    
    ultimate_high_conf_low_acc = [
        task for task in results["task_details"]
        if task["ultimate_ai"]["confidence"] > 0.7 and task["ultimate_ai"]["accuracy"] < 0.5
    ]
    
    arc_high_conf_low_acc = [
        task for task in results["task_details"]
        if task["arc_solver"]["confidence"] > 0.7 and task["arc_solver"]["accuracy"] < 0.5
    ]
    
    print(f"   النظام النهائي - ثقة عالية ودقة منخفضة: {len(ultimate_high_conf_low_acc)} مهام")
    print(f"   نظام ARC - ثقة عالية ودقة منخفضة: {len(arc_high_conf_low_acc)} مهام")

def generate_performance_report(results):
    """إنتاج تقرير أداء شامل"""
    
    print("\n📋 تقرير الأداء الشامل")
    print("="*80)
    
    total_tasks = len(results["task_details"])
    
    # إحصائيات النظام النهائي
    ultimate_perfect = sum(1 for task in results["task_details"] if task["ultimate_ai"]["perfect_match"])
    ultimate_shape_correct = sum(1 for task in results["task_details"] if task["ultimate_ai"]["shape_match"])
    ultimate_avg_accuracy = np.mean(results["ultimate_ai"])
    ultimate_avg_confidence = np.mean([task["ultimate_ai"]["confidence"] for task in results["task_details"]])
    ultimate_avg_time = np.mean([task["ultimate_ai"]["processing_time"] for task in results["task_details"]])
    
    # إحصائيات نظام ARC
    arc_perfect = sum(1 for task in results["task_details"] if task["arc_solver"]["perfect_match"])
    arc_shape_correct = sum(1 for task in results["task_details"] if task["arc_solver"]["shape_match"])
    arc_avg_accuracy = np.mean(results["arc_solver"])
    arc_avg_confidence = np.mean([task["arc_solver"]["confidence"] for task in results["task_details"]])
    arc_avg_time = np.mean([task["arc_solver"]["processing_time"] for task in results["task_details"]])
    
    print(f"🧠 النظام النهائي:")
    print(f"   ✅ حلول مثالية: {ultimate_perfect}/{total_tasks} ({ultimate_perfect/total_tasks:.1%})")
    print(f"   📐 أشكال صحيحة: {ultimate_shape_correct}/{total_tasks} ({ultimate_shape_correct/total_tasks:.1%})")
    print(f"   📈 متوسط الدقة: {ultimate_avg_accuracy:.2%}")
    print(f"   🎯 متوسط الثقة: {ultimate_avg_confidence:.2f}")
    print(f"   ⏱️ متوسط الوقت: {ultimate_avg_time:.2f} ثانية")
    
    print(f"\n🎯 نظام ARC المتخصص:")
    print(f"   ✅ حلول مثالية: {arc_perfect}/{total_tasks} ({arc_perfect/total_tasks:.1%})")
    print(f"   📐 أشكال صحيحة: {arc_shape_correct}/{total_tasks} ({arc_shape_correct/total_tasks:.1%})")
    print(f"   📈 متوسط الدقة: {arc_avg_accuracy:.2%}")
    print(f"   🎯 متوسط الثقة: {arc_avg_confidence:.2f}")
    print(f"   ⏱️ متوسط الوقت: {arc_avg_time:.2f} ثانية")
    
    # مقارنة مفصلة
    print(f"\n🏆 المقارنة المفصلة:")
    
    accuracy_diff = ultimate_avg_accuracy - arc_avg_accuracy
    perfect_diff = ultimate_perfect - arc_perfect
    time_diff = ultimate_avg_time - arc_avg_time
    
    print(f"   📈 فرق الدقة: {accuracy_diff:+.2%} لصالح {'النظام النهائي' if accuracy_diff > 0 else 'نظام ARC'}")
    print(f"   ✅ فرق الحلول المثالية: {perfect_diff:+d} لصالح {'النظام النهائي' if perfect_diff > 0 else 'نظام ARC'}")
    print(f"   ⏱️ فرق الوقت: {time_diff:+.2f} ثانية ({'أبطأ' if time_diff > 0 else 'أسرع'} للنظام النهائي)")
    
    # توصيات للتحسين
    print(f"\n💡 توصيات للتحسين:")
    
    if ultimate_avg_accuracy < arc_avg_accuracy:
        print("   🧠 النظام النهائي يحتاج تحسين في:")
        print("      - خوارزميات التعرف على الأنماط")
        print("      - معالجة البيانات المكانية")
        print("      - التعلم من الأمثلة القليلة")
    
    if arc_avg_accuracy < ultimate_avg_accuracy:
        print("   🎯 نظام ARC يحتاج تحسين في:")
        print("      - التعميم على مهام جديدة")
        print("      - التكيف مع أنماط معقدة")
        print("      - الاستفادة من المعرفة السابقة")
    
    if ultimate_avg_time > arc_avg_time * 2:
        print("   ⚡ النظام النهائي يحتاج تحسين في السرعة")
    
    if arc_avg_time > ultimate_avg_time * 2:
        print("   ⚡ نظام ARC يحتاج تحسين في السرعة")

def create_visualization(results):
    """إنشاء رسوم بيانية للنتائج"""
    
    try:
        import matplotlib.pyplot as plt
        
        # رسم مقارنة الدقة
        task_names = [f"Task {i+1}" for i in range(len(results["task_details"]))]
        ultimate_accuracies = [task["ultimate_ai"]["accuracy"] for task in results["task_details"]]
        arc_accuracies = [task["arc_solver"]["accuracy"] for task in results["task_details"]]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(task_names))
        width = 0.35
        
        plt.bar(x - width/2, ultimate_accuracies, width, label='النظام النهائي', alpha=0.8)
        plt.bar(x + width/2, arc_accuracies, width, label='نظام ARC', alpha=0.8)
        
        plt.xlabel('المهام')
        plt.ylabel('الدقة')
        plt.title('مقارنة دقة الأنظمة على مهام ARC')
        plt.xticks(x, task_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('arc_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 تم حفظ الرسم البياني في arc_performance_comparison.png")
        
    except ImportError:
        print("⚠️ matplotlib غير متوفر، لا يمكن إنشاء الرسوم البيانية")

def main():
    """الدالة الرئيسية"""
    
    results = load_test_results()
    if not results:
        return
    
    print("🔬 تحليل نتائج اختبار ARC")
    print("="*80)
    
    # تحليل الأداء
    analyze_performance_patterns(results)
    
    # تحليل الفشل
    analyze_failure_modes(results)
    
    # تقرير شامل
    generate_performance_report(results)
    
    # رسوم بيانية
    create_visualization(results)
    
    print("\n✅ انتهى التحليل!")

if __name__ == "__main__":
    main()
