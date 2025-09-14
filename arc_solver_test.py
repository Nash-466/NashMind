from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار حلال ARC الفعلي
"""

import json
import numpy as np
import traceback

def test_arc_solver():
    """اختبار حلال ARC على مهمة حقيقية"""
    print("🎯 اختبار حلال ARC الفعلي")
    print("=" * 40)
    
    # تحميل مهمة من البيانات
    try:
        with open('ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
            tasks = json.load(f)
        
        # أخذ أول مهمة
        task_id = list(tasks.keys())[0]
        task = tasks[task_id]
        
        print(f"📋 اختبار المهمة: {task_id}")
        print(f"📊 عدد أمثلة التدريب: {len(task['train'])}")
        print(f"🧪 عدد اختبارات: {len(task['test'])}")
        
        # عرض المثال الأول
        example = task['train'][0]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"\n📐 أبعاد المدخل: {input_grid.shape}")
        print(f"📐 أبعاد المخرج: {output_grid.shape}")
        print(f"🎨 ألوان المدخل: {np.unique(input_grid)}")
        print(f"🎨 ألوان المخرج: {np.unique(output_grid)}")
        
        # محاولة استيراد وتشغيل الحلال
        print(f"\n🔧 محاولة تحميل الحلال...")
        
        try:
            from arc_ultimate_perfect_system import ARCUltimatePerfectSolver
            solver = ARCUltimatePerfectSolver()
            print("✅ تم تحميل الحلال بنجاح")
            
            # محاولة حل المهمة
            print("🚀 محاولة حل المهمة...")
            solution = solver.solve_task(task)
            
            if solution is not None:
                print(f"✅ تم إنتاج حل: {type(solution)}")
                if hasattr(solution, 'shape'):
                    print(f"📐 أبعاد الحل: {solution.shape}")
                elif isinstance(solution, list):
                    print(f"📐 عدد الحلول: {len(solution)}")
            else:
                print("❌ لم يتم إنتاج حل")
                
        except ImportError as e:
            print(f"❌ فشل في استيراد الحلال: {e}")
            
        except Exception as e:
            print(f"❌ خطأ في تشغيل الحلال: {e}")
            print("📋 تفاصيل الخطأ:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ فشل في تحميل البيانات: {e}")

def test_simple_patterns():
    """اختبار أنماط بسيطة"""
    print(f"\n🔍 اختبار تحليل الأنماط البسيطة")
    print("-" * 30)
    
    # نمط بسيط: انعكاس أفقي
    test_input = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    expected_output = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]  # نفس الشكل (متماثل)
    
    print("📊 نمط الاختبار:")
    print("  المدخل:", test_input)
    print("  المخرج المتوقع:", expected_output)
    
    # تحليل بسيط
    input_array = np.array(test_input)
    output_array = np.array(expected_output)
    
    # فحص التماثل
    is_symmetric_h = np.array_equal(input_array, np.fliplr(input_array))
    is_symmetric_v = np.array_equal(input_array, np.flipud(input_array))
    
    print(f"🔄 تماثل أفقي: {is_symmetric_h}")
    print(f"🔄 تماثل عمودي: {is_symmetric_v}")
    
    # فحص الألوان
    unique_colors = np.unique(input_array)
    print(f"🎨 الألوان المستخدمة: {unique_colors}")

def analyze_project_weaknesses():
    """تحليل نقاط ضعف المشروع"""
    print(f"\n⚠️ تحليل نقاط الضعف")
    print("-" * 30)
    
    weaknesses = [
        "عدم وجود تكامل فعال بين الأنظمة المختلفة",
        "الاعتماد على pattern matching بسيط بدلاً من الفهم العميق",
        "عدم وجود آلية تعلم تكيفية من الأمثلة",
        "تعقيد غير مبرر في الكود (50,000+ سطر)",
        "عدم التركيز على الأنماط الأكثر شيوعاً (object_manipulation: 1234 مثال)",
        "نقص في معالجة المهام المعقدة (complex_unknown: 1003 مثال)",
        "عدم وجود نظام اختبار مستمر وتقييم الأداء"
    ]
    
    for i, weakness in enumerate(weaknesses, 1):
        print(f"  {i}. {weakness}")

def main():
    """الوظيفة الرئيسية"""
    test_arc_solver()
    test_simple_patterns()
    analyze_project_weaknesses()
    
    print(f"\n🎯 الخطوات التالية:")
    print("1. إصلاح مشاكل التكامل في الحلال الحالي")
    print("2. تطوير نظام تعلم تكيفي من الأنماط")
    print("3. التركيز على الأنماط الأكثر شيوعاً")
    print("4. بناء نظام اختبار مستمر")
    print("5. تبسيط وتنظيف الكود")

if __name__ == "__main__":
    main()
