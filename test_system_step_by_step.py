from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 اختبار النظام خطوة بخطوة
"""

import numpy as np
import time

def test_step_by_step():
    """اختبار تدريجي للنظام"""
    
    print("🔍 اختبار النظام خطوة بخطوة...")
    print("="*50)
    
    # Step 1: Test EfficientZero alone
    print("1️⃣ اختبار EfficientZero منفرداً...")
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        print("   ✅ EfficientZero تم تحميله بنجاح")
        
        test_grid = np.array([[1, 0], [0, 1]])
        result = ez.solve_arc_problem(test_grid, max_steps=3)
        print(f"   ✅ اختبار أساسي: ثقة={result.get('confidence', 0):.2f}")
        
    except Exception as e:
        print(f"   ❌ خطأ في EfficientZero: {e}")
        return False
    
    # Step 2: Test main system import
    print("\n2️⃣ اختبار استيراد النظام الرئيسي...")
    try:
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        print("   ✅ النظام الرئيسي تم استيراده")
    except Exception as e:
        print(f"   ❌ خطأ في استيراد النظام: {e}")
        return False
    
    # Step 3: Test system initialization
    print("\n3️⃣ اختبار تهيئة النظام...")
    try:
        system = UltraAdvancedARCSystem()
        print("   ✅ النظام تم تهيئته")
        print(f"   📊 EfficientZero متاح: {hasattr(system, 'efficient_zero_engine')}")
    except Exception as e:
        print(f"   ❌ خطأ في تهيئة النظام: {e}")
        return False
    
    # Step 4: Test basic solve
    print("\n4️⃣ اختبار الحل الأساسي...")
    try:
        test_grid = np.array([[1, 0], [0, 1]])
        start_time = time.time()
        solution = system.solve_arc_challenge(test_grid)
        solve_time = time.time() - start_time
        
        print(f"   ✅ الحل نجح:")
        print(f"      - الثقة: {solution.confidence:.2f}")
        print(f"      - الطريقة: {solution.method}")
        print(f"      - الوقت: {solve_time:.3f}s")
        print(f"      - شكل الحل: {solution.output_grid.shape}")
        
    except Exception as e:
        print(f"   ❌ خطأ في الحل: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test EfficientZero integration
    print("\n5️⃣ اختبار تكامل EfficientZero...")
    try:
        if hasattr(system, 'efficient_zero_engine'):
            ez_stats = system.efficient_zero_engine.get_performance_stats()
            print(f"   ✅ إحصائيات EfficientZero:")
            print(f"      - خطوات التدريب: {ez_stats['training_steps']}")
            print(f"      - المحاكاات: {ez_stats['num_simulations']}")
            print(f"      - درجة الحرارة: {ez_stats['temperature']:.2f}")
        else:
            print("   ⚠️  EfficientZero غير متاح في النظام")
    except Exception as e:
        print(f"   ❌ خطأ في تكامل EfficientZero: {e}")
    
    print("\n" + "="*50)
    print("🎉 جميع الاختبارات اكتملت بنجاح!")
    return True

if __name__ == "__main__":
    test_step_by_step()
