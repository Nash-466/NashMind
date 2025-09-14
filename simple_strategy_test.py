from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار بسيط للاستراتيجيات الجديدة
"""

import numpy as np
import sys
import os

# إضافة مسار NashMind
sys.path.append(os.path.join(os.getcwd(), 'NashMind'))

from core.arc_problem_solver import ARCProblemSolver

def test_color_strategies():
    """اختبار استراتيجيات الألوان"""
    
    print("🎨 اختبار استراتيجيات الألوان")
    print("="*40)
    
    # إنشاء حلال المسائل
    solver = ARCProblemSolver()
    
    # شبكة اختبار
    test_grid = np.array([
        [1, 0, 2],
        [0, 1, 0],
        [2, 0, 1]
    ])
    
    print(f"الشبكة الأصلية:")
    print(test_grid)
    
    # اختبار عكس الألوان
    inverted = solver.invert_colors(test_grid)
    if inverted is not None:
        print(f"\nعكس الألوان:")
        print(inverted)
    
    # اختبار تبديل الألوان
    swapped = solver.swap_dominant_colors(test_grid)
    if swapped is not None:
        print(f"\nتبديل الألوان:")
        print(swapped)
    
    # اختبار التدرج
    gradient = solver.apply_color_gradient(test_grid)
    if gradient is not None:
        print(f"\nالتدرج اللوني:")
        print(gradient)
    
    print("\n✅ انتهى اختبار استراتيجيات الألوان")

def test_size_strategies():
    """اختبار استراتيجيات الحجم"""
    
    print("\n📏 اختبار استراتيجيات الحجم")
    print("="*40)
    
    # إنشاء حلال المسائل
    solver = ARCProblemSolver()
    
    # شبكة اختبار أكبر
    test_grid = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 1, 2, 3],
        [4, 5, 6, 7]
    ])
    
    print(f"الشبكة الأصلية ({test_grid.shape}):")
    print(test_grid)
    
    # اختبار التصغير
    shrunk = solver.shrink_by_half(test_grid)
    if shrunk is not None:
        print(f"\nتصغير بالنصف ({shrunk.shape}):")
        print(shrunk)
    
    # اختبار قص الحواف
    cropped = solver.crop_borders(test_grid)
    if cropped is not None:
        print(f"\nقص الحواف ({cropped.shape}):")
        print(cropped)
    
    print("\n✅ انتهى اختبار استراتيجيات الحجم")

def test_basic_strategies():
    """اختبار الاستراتيجيات الأساسية"""
    
    print("\n🔄 اختبار الاستراتيجيات الأساسية")
    print("="*40)
    
    # إنشاء حلال المسائل
    solver = ARCProblemSolver()
    
    # شبكة اختبار
    test_grid = np.array([
        [1, 0, 2],
        [0, 1, 0],
        [2, 0, 1]
    ])
    
    print(f"الشبكة الأصلية:")
    print(test_grid)
    
    # تطبيق الاستراتيجيات الأساسية
    candidates = solver.apply_basic_strategies(test_grid)
    
    print(f"\nعدد المرشحين: {len(candidates)}")
    
    for i, candidate in enumerate(candidates[:5]):  # أول 5 مرشحين
        strategy = candidate['strategy']
        confidence = candidate['confidence']
        output = np.array(candidate['output'])
        
        print(f"\nمرشح {i+1}: {strategy} (ثقة: {confidence})")
        print(output)
    
    print("\n✅ انتهى اختبار الاستراتيجيات الأساسية")

if __name__ == "__main__":
    try:
        test_color_strategies()
        test_size_strategies()
        test_basic_strategies()
        
        print("\n🎉 انتهى جميع الاختبارات بنجاح!")
        
    except Exception as e:
        print(f"\n❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
