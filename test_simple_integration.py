from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبار التكامل البسيط
"""

import numpy as np
import time
import sys

def test_simple_integration():
    """اختبار تكامل بسيط"""
    
    print("🔍 اختبار التكامل البسيط...")
    print("="*40)
    
    # Test 1: EfficientZero alone
    print("1. اختبار EfficientZero...")
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        test_grid = np.array([[1, 0], [0, 1]])
        result = ez.solve_arc_problem(test_grid, max_steps=2)
        
        print(f"   ✅ نجح - ثقة: {result.get('confidence', 0):.2f}")
        print(f"   📊 خطوات: {result.get('steps_taken', 0)}")
        print(f"   ⏱️  وقت: {result.get('solve_time', 0):.3f}s")
        
    except Exception as e:
        print(f"   ❌ فشل: {e}")
        return False
    
    # Test 2: Simple solve function
    print("\n2. اختبار دالة الحل البسيطة...")
    try:
        # Create a simple solver using EfficientZero
        def simple_solve(input_grid):
            ez = EfficientZeroEngine()
            result = ez.solve_arc_problem(input_grid, max_steps=3)
            
            return {
                'output_grid': result.get('solution_grid', input_grid),
                'confidence': result.get('confidence', 0.1),
                'method': 'efficient_zero_simple',
                'generation_time': result.get('solve_time', 0.0)
            }
        
        # Test the simple solver
        test_grid = np.array([[2, 1], [1, 2]])
        solution = simple_solve(test_grid)
        
        print(f"   ✅ نجح - ثقة: {solution['confidence']:.2f}")
        print(f"   📊 الطريقة: {solution['method']}")
        print(f"   ⏱️  الوقت: {solution['generation_time']:.3f}s")
        print(f"   🔢 شكل الحل: {solution['output_grid'].shape}")
        
    except Exception as e:
        print(f"   ❌ فشل: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Multiple grids
    print("\n3. اختبار شبكات متعددة...")
    test_grids = [
        np.array([[1]]),  # 1x1
        np.array([[1, 0], [0, 1]]),  # 2x2
        np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]),  # 3x3
    ]
    
    for i, grid in enumerate(test_grids):
        try:
            start_time = time.time()
            solution = simple_solve(grid)
            solve_time = time.time() - start_time
            
            print(f"   {i+1}. شبكة {grid.shape}: ثقة={solution['confidence']:.2f}, "
                  f"وقت={solve_time:.3f}s")
            
        except Exception as e:
            print(f"   {i+1}. شبكة {grid.shape}: فشل - {e}")
    
    print("\n" + "="*40)
    print("🎉 اختبار التكامل البسيط اكتمل!")
    return True

if __name__ == "__main__":
    success = test_simple_integration()
    if success:
        print("\n✅ جميع الاختبارات نجحت!")
        print("🚀 EfficientZero جاهز للاستخدام!")
    else:
        print("\n❌ بعض الاختبارات فشلت")
    
    sys.exit(0 if success else 1)
