from __future__ import annotations
#!/usr/bin/env python3
"""
تشخيص مفصل للأنظمة - اكتشاف المشاكل الدقيقة
"""

import traceback
import sys
import os

def test_arc_ultimate_perfect_system():
    """اختبار النظام المثالي"""
    print("🔧 اختبار arc_ultimate_perfect_system...")
    
    try:
        print("  📦 محاولة الاستيراد...", end=" ")
        from arc_ultimate_perfect_system import ARCUltimatePerfectSolver
        print("✅")
        
        print("  🏗️ إنشاء كائن الحلال...", end=" ")
        solver = ARCUltimatePerfectSolver()
        print("✅")
        
        # اختبار بسيط
        test_task = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {'input': [[1, 0], [0, 1]]}
            ]
        }
        
        print("  🎯 اختبار حل مهمة بسيطة...", end=" ")
        result = solver.solve_task(test_task)
        print(f"✅ نتيجة: {type(result)}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        print("📋 تفاصيل الخطأ:")
        traceback.print_exc()
        return False

def test_individual_components():
    """اختبار المكونات الفردية"""
    print("\n🧩 اختبار المكونات الفردية...")
    
    components = [
        "arc_complete_agent_part1",
        "arc_complete_agent_part2", 
        "arc_complete_agent_part3",
        "arc_complete_agent_part4"
    ]
    
    results = {}
    
    for component in components:
        print(f"  🔍 اختبار {component}...", end=" ")
        try:
            module = __import__(component)
            print("✅ تم الاستيراد")
            results[component] = "success"
        except Exception as e:
            print(f"❌ فشل: {e}")
            results[component] = str(e)
    
    return results

def test_numpy_operations():
    """اختبار عمليات numpy المشكوك فيها"""
    print("\n🔢 اختبار عمليات numpy...")
    
    import numpy as np
    
    # اختبارات محتملة للمشكلة
    tests = [
        ("array comparison", lambda: np.array([1, 2]) == np.array([1, 2])),
        ("array truth value", lambda: bool(np.array([1]))),
        ("array any", lambda: np.array([1, 0]).any()),
        ("array all", lambda: np.array([1, 1]).all()),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"  🧪 {test_name}...", end=" ")
            result = test_func()
            print(f"✅ {result}")
        except Exception as e:
            print(f"❌ {e}")

def run_simple_arc_solver():
    """تشغيل حلال ARC بسيط"""
    print("\n🎮 تشغيل حلال ARC بسيط...")
    
    try:
        import json
        
        # تحميل مهمة واحدة
        with open('ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
            tasks = json.load(f)
        
        task_id = list(tasks.keys())[0]
        task = tasks[task_id]
        
        print(f"  📋 المهمة: {task_id}")
        print(f"  📊 أمثلة التدريب: {len(task['train'])}")
        
        # محاولة حل بسيط
        train_example = task['train'][0]
        input_grid = train_example['input']
        output_grid = train_example['output']
        
        print(f"  📐 المدخل: {len(input_grid)}x{len(input_grid[0])}")
        print(f"  📐 المخرج: {len(output_grid)}x{len(output_grid[0])}")
        
        # تحليل بسيط للنمط
        import numpy as np
        input_array = np.array(input_grid)
        output_array = np.array(output_grid)
        
        print(f"  🔍 قيم المدخل الفريدة: {np.unique(input_array)}")
        print(f"  🔍 قيم المخرج الفريدة: {np.unique(output_array)}")
        
        # فحص إذا كان هناك تحويل بسيط
        if input_array.shape == output_array.shape:
            print("  ✅ نفس الأبعاد - ربما تحويل ألوان")
        else:
            print("  🔄 أبعاد مختلفة - تحويل هندسي")
        
        return True
        
    except Exception as e:
        print(f"  ❌ فشل: {e}")
        traceback.print_exc()
        return False

def main():
    """الوظيفة الرئيسية للتشخيص"""
    print("🔬 تشخيص مفصل للأنظمة")
    print("=" * 50)
    
    # اختبار النظام الرئيسي
    system_works = test_arc_ultimate_perfect_system()
    
    # اختبار المكونات
    component_results = test_individual_components()
    
    # اختبار numpy
    test_numpy_operations()
    
    # اختبار حل ARC بسيط
    arc_works = run_simple_arc_solver()
    
    # تقرير التشخيص
    print("\n📊 تقرير التشخيص:")
    print("-" * 30)
    
    print(f"🎯 النظام الرئيسي: {'✅ يعمل' if system_works else '❌ معطل'}")
    print(f"🎮 حل ARC البسيط: {'✅ يعمل' if arc_works else '❌ معطل'}")
    
    print("\n🧩 حالة المكونات:")
    for component, status in component_results.items():
        icon = "✅" if status == "success" else "❌"
        print(f"  {icon} {component}: {status}")
    
    # توصيات الإصلاح
    print("\n🔧 توصيات الإصلاح:")
    if not system_works:
        print("  1. إصلاح مشاكل التكامل في النظام الرئيسي")
    
    failed_components = [k for k, v in component_results.items() if v != "success"]
    if failed_components:
        print(f"  2. إصلاح المكونات المعطلة: {', '.join(failed_components)}")
    
    if system_works and arc_works:
        print("  ✅ الأنظمة تعمل - يمكن البدء في التحسين")
    else:
        print("  🚨 مطلوب إصلاحات أساسية قبل المتابعة")

if __name__ == "__main__":
    main()
