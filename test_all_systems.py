from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار جميع الأنظمة مع عرض النتائج المباشر
"""

import os
import sys
import traceback

def print_header(title):
    """طباعة عنوان مع تنسيق"""
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print(f"{'='*50}")

def test_file_exists(filename):
    """اختبار وجود ملف"""
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"✅ {filename} - الحجم: {size:,} بايت")
        return True
    else:
        print(f"❌ {filename} - غير موجود")
        return False

def test_import(module_name):
    """اختبار استيراد وحدة"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - استيراد ناجح")
        return True
    except Exception as e:
        print(f"❌ {module_name} - فشل الاستيراد: {str(e)[:100]}")
        return False

def main():
    """اختبار شامل لجميع الأنظمة"""
    
    print_header("اختبار وجود ملفات الأنظمة")
    
    # قائمة جميع الأنظمة
    system_files = [
        "main.py",
        "arc_complete_agent_part1.py",
        "arc_complete_agent_part2.py", 
        "arc_complete_agent_part3.py",
        "arc_complete_agent_part4.py",
        "arc_complete_agent_part5.py",
        "arc_complete_agent_part6.py",
        "arc_ultimate_mind_part7.py",
        "burhan_meta_brain.py",
        "arc_ultimate_perfect_system.py",
        "arc_core_redesign.py",
        "advanced_pattern_engine.py"
    ]
    
    existing_files = 0
    for file in system_files:
        if test_file_exists(file):
            existing_files += 1
    
    print(f"\n📊 النتيجة: {existing_files}/{len(system_files)} ملف موجود")
    
    print_header("اختبار ملفات البيانات")
    
    data_files = [
        "ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json",
        "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json", 
        "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json"
    ]
    
    data_count = 0
    for file in data_files:
        if test_file_exists(file):
            data_count += 1
    
    print(f"\n📊 النتيجة: {data_count}/{len(data_files)} ملف بيانات موجود")
    
    print_header("اختبار المكتبات الأساسية")
    
    libraries = ["numpy", "json", "time", "typing", "collections", "dataclasses"]
    lib_count = 0
    for lib in libraries:
        if test_import(lib):
            lib_count += 1
    
    print(f"\n📊 النتيجة: {lib_count}/{len(libraries)} مكتبة متاحة")
    
    print_header("اختبار استيراد الأنظمة")
    
    # اختبار استيراد الأنظمة الرئيسية
    modules_to_test = [
        "arc_ultimate_perfect_system",
        "arc_core_redesign", 
        "advanced_pattern_engine"
    ]
    
    working_modules = 0
    for module in modules_to_test:
        if test_import(module):
            working_modules += 1
    
    print(f"\n📊 النتيجة: {working_modules}/{len(modules_to_test)} نظام يعمل")
    
    print_header("اختبار تحميل مهام ARC")
    
    try:
        import json
        with open('ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
            training_tasks = json.load(f)
        print(f"✅ مهام التدريب: {len(training_tasks)} مهمة")
        
        with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
            eval_tasks = json.load(f)
        print(f"✅ مهام التقييم: {len(eval_tasks)} مهمة")
        
        # اختبار مهمة واحدة
        first_task = list(training_tasks.values())[0]
        print(f"✅ مثال مهمة: {len(first_task['train'])} أمثلة تدريب، {len(first_task['test'])} اختبار")
        
    except Exception as e:
        print(f"❌ فشل تحميل مهام ARC: {e}")
    
    print_header("اختبار النظام المتكامل")
    
    try:
        from arc_ultimate_perfect_system import ARCUltimatePerfectSolver
        solver = ARCUltimatePerfectSolver()
        print("✅ تم إنشاء حلال ARC بنجاح")
        
        # اختبار بسيط
        test_task = {
            'train': [{'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}],
            'test': [{'input': [[1, 0], [0, 1]]}]
        }
        
        result = solver.solve_task(test_task)
        print(f"✅ اختبار الحل: {type(result)} - {len(result) if result else 0}")
        
    except Exception as e:
        print(f"❌ فشل اختبار النظام المتكامل: {e}")
    
    print_header("ملخص التقييم النهائي")
    
    total_score = existing_files + data_count + lib_count + working_modules
    max_score = len(system_files) + len(data_files) + len(libraries) + len(modules_to_test)
    
    percentage = (total_score / max_score) * 100
    
    print(f"📊 النقاط الإجمالية: {total_score}/{max_score}")
    print(f"📈 نسبة النجاح: {percentage:.1f}%")
    
    if percentage >= 80:
        print("🎉 الأنظمة تعمل بشكل جيد!")
    elif percentage >= 60:
        print("⚠️ الأنظمة تحتاج بعض الإصلاحات")
    else:
        print("🚨 الأنظمة تحتاج إصلاحات جذرية")
    
    print(f"\n🎯 التوصيات:")
    if existing_files < len(system_files):
        print(f"  • إنشاء الملفات المفقودة ({len(system_files) - existing_files} ملف)")
    if working_modules < len(modules_to_test):
        print(f"  • إصلاح مشاكل الاستيراد ({len(modules_to_test) - working_modules} نظام)")
    if data_count < len(data_files):
        print(f"  • التأكد من ملفات البيانات ({len(data_files) - data_count} ملف)")
    
    print("\n✅ انتهى الاختبار الشامل")

if __name__ == "__main__":
    main()
