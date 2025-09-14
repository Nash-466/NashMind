from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 فحص بسيط لجميع الملفات
"""

import sys
import importlib

def check_core_files():
    """فحص الملفات الأساسية"""
    
    print("🔍 فحص الملفات الأساسية...")
    print("="*50)
    
    # Core files to check
    files_to_check = [
        'ultra_advanced_arc_system',
        'efficient_zero_engine', 
        'arc_complete_agent_part2',
        'arc_hierarchical_reasoning',
        'semantic_memory_system',
        'creative_innovation_engine',
        'intelligent_verification_system',
        'advanced_simulation_engine',
        'arc_adaptive_self_improvement'
    ]
    
    results = {}
    success_count = 0
    
    for file_name in files_to_check:
        print(f"📁 {file_name}...")
        try:
            module = importlib.import_module(file_name)
            print(f"   ✅ استيراد نجح")
            
            # Check for main classes
            classes = [name for name in dir(module) 
                      if isinstance(getattr(module, name), type) 
                      and not name.startswith('_')]
            
            if classes:
                print(f"   📋 فئات: {', '.join(classes[:3])}")
                results[file_name] = {'status': 'success', 'classes': classes}
                success_count += 1
            else:
                print(f"   ⚠️  لا توجد فئات رئيسية")
                results[file_name] = {'status': 'no_classes'}
            
        except Exception as e:
            print(f"   ❌ فشل: {str(e)[:60]}...")
            results[file_name] = {'status': 'failed', 'error': str(e)}
    
    print(f"\n📊 النتائج: {success_count}/{len(files_to_check)} نجح")
    return results, success_count, len(files_to_check)

def test_efficient_zero():
    """اختبار EfficientZero"""
    
    print(f"\n🧠 اختبار EfficientZero...")
    print("-"*30)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        import numpy as np
        
        ez = EfficientZeroEngine()
        print("   ✅ تهيئة نجحت")
        
        # Test solve
        test_grid = np.array([[1, 0], [0, 1]])
        result = ez.solve_arc_problem(test_grid, max_steps=2)
        
        confidence = result.get('confidence', 0)
        print(f"   ✅ حل نجح - ثقة: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ فشل: {e}")
        return False

def test_main_system():
    """اختبار النظام الرئيسي"""
    
    print(f"\n🚀 اختبار النظام الرئيسي...")
    print("-"*30)
    
    try:
        # Try to import main system
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        print("   ✅ استيراد النظام نجح")
        
        # Try to initialize (with timeout protection)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("تجاوز الوقت المحدد")
        
        # Set timeout for 30 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            system = UltraAdvancedARCSystem()
            signal.alarm(0)  # Cancel timeout
            print("   ✅ تهيئة النظام نجحت")
            
            # Check subsystems
            subsystems = ['pattern_analyzer', 'reasoning_engine', 'efficient_zero_engine']
            active = sum(1 for s in subsystems if hasattr(system, s))
            print(f"   📊 أنظمة فرعية نشطة: {active}/{len(subsystems)}")
            
            return True
            
        except TimeoutError:
            signal.alarm(0)
            print("   ⚠️  تجاوز الوقت المحدد للتهيئة")
            return False
        
    except Exception as e:
        print(f"   ❌ فشل: {e}")
        return False

def main():
    """الدالة الرئيسية"""
    
    print("🔍 فحص بسيط لجميع ملفات النظام")
    print("="*50)
    
    # Check files
    results, success_count, total_count = check_core_files()
    
    # Test EfficientZero
    ez_success = test_efficient_zero()
    
    # Test main system (with caution)
    main_success = test_main_system()
    
    # Summary
    print("\n" + "="*50)
    print("📊 الملخص النهائي:")
    print(f"   📁 ملفات: {success_count}/{total_count} نجح")
    print(f"   🧠 EfficientZero: {'✅ يعمل' if ez_success else '❌ لا يعمل'}")
    print(f"   🚀 النظام الرئيسي: {'✅ يعمل' if main_success else '❌ لا يعمل'}")
    
    overall_success = success_count >= total_count * 0.7 and ez_success
    
    if overall_success:
        print("\n🎉 النظام في حالة جيدة!")
        print("✅ جاهز للاختبار على المهام")
    else:
        print("\n⚠️  النظام يحتاج مراجعة")
        print("🔧 بعض المكونات لا تعمل بشكل صحيح")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  تم إيقاف الفحص")
        sys.exit(1)
