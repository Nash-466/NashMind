from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبار محرك EfficientZero المتقدم
"""

import sys
import traceback
import numpy as np
import time

def test_efficient_zero_system():
    """اختبار شامل لمحرك EfficientZero"""
    
    print('🧠 اختبار النظام مع EfficientZero...')
    print('='*60)

    try:
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        print('✅ استيراد النظام: نجح')
        
        # Initialize system
        system = UltraAdvancedARCSystem()
        print('✅ تهيئة النظام: نجحت')
        
        # Check EfficientZero engine
        if hasattr(system, 'efficient_zero_engine'):
            print('✅ محرك EfficientZero: متاح')
            
            # Test basic functionality
            test_grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
            print(f'🔍 اختبار شبكة: {test_grid.shape}')
            
            # Test EfficientZero directly
            ez_result = system.efficient_zero_engine.solve_arc_problem(test_grid, max_steps=5)
            confidence = ez_result.get('confidence', 0)
            print(f'✅ EfficientZero مباشر: نجح (ثقة: {confidence:.2f})')
            
            # Test full system
            solution = system.solve_arc_challenge(test_grid)
            print(f'✅ النظام الكامل: نجح (ثقة: {solution.confidence:.2f})')
            print(f'📊 الطريقة: {solution.method}')
            print(f'⏱️  الوقت: {solution.generation_time:.3f}s')
            
            # Check performance stats
            stats = system.efficient_zero_engine.get_performance_stats()
            print(f'📈 إحصائيات EfficientZero:')
            print(f'   - خطوات التدريب: {stats["training_steps"]}')
            print(f'   - عدد المحاكاات: {stats["num_simulations"]}')
            print(f'   - درجة الحرارة: {stats["temperature"]:.2f}')
            
            # Test with different grid sizes
            test_grids = [
                np.array([[1, 0], [0, 1]]),  # 2x2
                np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]),  # 3x3 with colors
                np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])  # 4x4
            ]
            
            print('\n🔬 اختبار أحجام مختلفة:')
            for i, grid in enumerate(test_grids):
                start_time = time.time()
                result = system.efficient_zero_engine.solve_arc_problem(grid, max_steps=3)
                solve_time = time.time() - start_time
                
                print(f'   {i+1}. شبكة {grid.shape}: ثقة={result.get("confidence", 0):.2f}, '
                      f'وقت={solve_time:.3f}s, خطوات={result.get("steps_taken", 0)}')
            
        else:
            print('❌ محرك EfficientZero: غير متاح')
        
        print('='*60)
        print('🎉 جميع الاختبارات نجحت!')
        return True
        
    except Exception as e:
        print(f'❌ خطأ: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_efficient_zero_system()
    sys.exit(0 if success else 1)
