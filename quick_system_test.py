from __future__ import annotations
#!/usr/bin/env python3
"""
QUICK SYSTEM TEST - اختبار سريع للأنظمة
=====================================
اختبار مبسط لمعرفة حالة الأنظمة الحالية
"""

import os
import sys
import json

# تشغيل الاختبار مباشرة
print("🧪 اختبار شامل لجميع الأنظمة")
print("=" * 50)

# اختبار الملفات الأساسية
print("\n📁 فحص ملفات الأنظمة:")
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

existing_count = 0
for file in system_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✅ {file} ({size:,} بايت)")
        existing_count += 1
    else:
        print(f"  ❌ {file} - مفقود")

print(f"\n📊 الملفات الموجودة: {existing_count}/{len(system_files)}")

# اختبار ملفات البيانات
print("\n📊 فحص ملفات البيانات:")
data_files = [
    "ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json",
    "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json",
    "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json"
]

data_count = 0
for file in data_files:
    if os.path.exists(file):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            print(f"  ✅ {os.path.basename(file)} ({len(data)} مهمة)")
            data_count += 1
        except:
            print(f"  ⚠️ {os.path.basename(file)} - خطأ في القراءة")
    else:
        print(f"  ❌ {os.path.basename(file)} - مفقود")

print(f"\n📊 ملفات البيانات: {data_count}/{len(data_files)}")

# اختبار المكتبات
print("\n📦 فحص المكتبات:")
libraries = ["numpy", "json", "time", "typing", "collections", "dataclasses"]
lib_count = 0
for lib in libraries:
    try:
        __import__(lib)
        print(f"  ✅ {lib}")
        lib_count += 1
    except:
        print(f"  ❌ {lib}")

print(f"\n📊 المكتبات المتاحة: {lib_count}/{len(libraries)}")

# اختبار الأنظمة الرئيسية
print("\n🔧 فحص الأنظمة الرئيسية:")
main_systems = [
    "arc_ultimate_perfect_system",
    "arc_core_redesign", 
    "advanced_pattern_engine"
]

working_systems = 0
for system in main_systems:
    try:
        __import__(system)
        print(f"  ✅ {system}")
        working_systems += 1
    except Exception as e:
        print(f"  ❌ {system} - {str(e)[:50]}...")

print(f"\n📊 الأنظمة العاملة: {working_systems}/{len(main_systems)}")

# النتيجة النهائية
total_score = existing_count + data_count + lib_count + working_systems
max_score = len(system_files) + len(data_files) + len(libraries) + len(main_systems)
percentage = (total_score / max_score) * 100

print(f"\n🎯 التقييم النهائي:")
print(f"📈 النقاط: {total_score}/{max_score}")
print(f"📊 النسبة: {percentage:.1f}%")

if percentage >= 80:
    print("🎉 الأنظمة تعمل بشكل ممتاز!")
elif percentage >= 60:
    print("⚠️ الأنظمة تحتاج بعض الإصلاحات")
else:
    print("🚨 الأنظمة تحتاج إصلاحات جذرية")

print("\n" + "="*50)

def test_system_files():
    """اختبار ملفات النظام"""
    print("🔍 فحص ملفات النظام...")
    
    # قائمة الملفات المهمة
    important_files = [
        "main.py",
        "arc_complete_agent_part1.py",
        "arc_complete_agent_part2.py", 
        "arc_complete_agent_part3.py",
        "arc_complete_agent_part4.py",
        "arc_ultimate_mind_part7.py",
        "burhan_meta_brain.py",
        "arc_ultimate_perfect_system.py"
    ]
    
    results = {}
    
    for file in important_files:
        print(f"  📄 فحص {file}...", end=" ", flush=True)
        
        if os.path.exists(file):
            try:
                # قراءة الملف
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # فحص أساسي
                has_classes = "class " in content
                has_functions = "def " in content
                file_size = len(content)
                
                results[file] = {
                    "exists": True,
                    "size": file_size,
                    "has_classes": has_classes,
                    "has_functions": has_functions,
                    "status": "✅ موجود"
                }
                
                print(f"✅ ({file_size} حرف, كلاسات: {has_classes}, وظائف: {has_functions})")
                
            except Exception as e:
                results[file] = {
                    "exists": True,
                    "error": str(e),
                    "status": f"❌ خطأ: {e}"
                }
                print(f"❌ خطأ في القراءة - {e}")
        else:
            results[file] = {
                "exists": False,
                "status": "❌ غير موجود"
            }
            print("❌ غير موجود")
    
    return results

def test_imports():
    """اختبار الاستيرادات"""
    print("\n🔗 اختبار الاستيرادات...")
    
    import_tests = [
        ("numpy", "np"),
        ("json", "json"),
        ("time", "time"),
        ("typing", "typing")
    ]
    
    for module, alias in import_tests:
        print(f"  📦 {module}...", end=" ", flush=True)
        try:
            exec(f"import {module} as {alias}")
            print("✅ متاح")
        except ImportError:
            print("❌ غير متاح")

def test_data_files():
    """اختبار ملفات البيانات"""
    print("\n📊 اختبار ملفات البيانات...")
    
    data_files = [
        "ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json",
        "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json",
        "ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json"
    ]
    
    for file in data_files:
        print(f"  📁 {os.path.basename(file)}...", end=" ", flush=True)
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                print(f"✅ ({len(data)} مهمة)")
            except Exception as e:
                print(f"❌ خطأ - {e}")
        else:
            print("❌ غير موجود")

def test_simple_arc_task():
    """اختبار بسيط على مهمة ARC"""
    print("\n🎮 اختبار على مهمة ARC بسيطة...")
    
    try:
        # تحميل مهمة واحدة
        with open('ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
            tasks = json.load(f)
        
        # أخذ أول مهمة
        first_task_id = list(tasks.keys())[0]
        first_task = tasks[first_task_id]
        
        print(f"  🎯 اختبار المهمة: {first_task_id}")
        print(f"  📊 عدد الأمثلة: {len(first_task['train'])}")
        print(f"  🧪 عدد الاختبارات: {len(first_task['test'])}")
        
        # فحص أبعاد المدخلات والمخرجات
        train_example = first_task['train'][0]
        input_shape = (len(train_example['input']), len(train_example['input'][0]))
        output_shape = (len(train_example['output']), len(train_example['output'][0]))
        
        print(f"  📐 أبعاد المدخل: {input_shape}")
        print(f"  📐 أبعاد المخرج: {output_shape}")
        
        # محاولة استيراد نظام الحل
        print("  🔧 محاولة تحميل نظام الحل...", end=" ", flush=True)
        try:
            from arc_ultimate_perfect_system import ARCUltimatePerfectSolver
            solver = ARCUltimatePerfectSolver()
            print("✅ تم تحميل النظام")
            
            # محاولة حل المهمة
            print("  🚀 محاولة حل المهمة...", end=" ", flush=True)
            solution = solver.solve_task(first_task)
            print(f"✅ تم إنتاج حل بأبعاد: {len(solution)} x {len(solution[0]) if solution else 0}")
            
        except Exception as e:
            print(f"❌ فشل - {e}")
            
    except Exception as e:
        print(f"  ❌ فشل في تحميل المهام: {e}")

def main():
    """الوظيفة الرئيسية"""
    print("🧪 اختبار سريع للأنظمة")
    print("=" * 40)
    
    # اختبار الملفات
    file_results = test_system_files()
    
    # اختبار الاستيرادات
    test_imports()
    
    # اختبار ملفات البيانات
    test_data_files()
    
    # اختبار مهمة ARC
    test_simple_arc_task()
    
    # ملخص النتائج
    print("\n📋 ملخص النتائج:")
    print("-" * 30)
    
    total_files = len(file_results)
    existing_files = sum(1 for r in file_results.values() if r["exists"])
    working_files = sum(1 for r in file_results.values() 
                       if r["exists"] and "error" not in r)
    
    print(f"📁 إجمالي الملفات: {total_files}")
    print(f"✅ الملفات الموجودة: {existing_files}")
    print(f"🔧 الملفات العاملة: {working_files}")
    print(f"❌ الملفات المعطلة: {existing_files - working_files}")
    
    if working_files < total_files:
        print("\n⚠️ مشاكل مكتشفة:")
        for file, result in file_results.items():
            if not result["exists"]:
                print(f"  • {file}: غير موجود")
            elif "error" in result:
                print(f"  • {file}: {result['error']}")
    
    print("\n🎯 التوصيات:")
    if working_files == total_files:
        print("  ✅ جميع الأنظمة تعمل - يمكن البدء في التحسين")
    else:
        print("  🔧 إصلاح الملفات المعطلة أولاً")
        print("  🧹 تنظيف وتبسيط الكود")
        print("  🔗 تحسين التكامل بين الأنظمة")

if __name__ == "__main__":
    main()
