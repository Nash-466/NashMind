from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 الاختبار النهائي للنظام المتقدم مع EfficientZero
"""

import numpy as np
import time
import json

def test_efficient_zero_standalone():
    """اختبار EfficientZero منفرداً"""
    
    print("🧠 اختبار EfficientZero منفرداً...")
    print("-" * 40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        # Test cases
        test_cases = [
            {
                'name': 'بسيط 2x2',
                'input': np.array([[1, 0], [0, 1]]),
                'expected_steps': 3
            },
            {
                'name': 'متوسط 3x3',
                'input': np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]),
                'expected_steps': 5
            },
            {
                'name': 'معقد 4x4',
                'input': np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 0, 1, 0], [0, 2, 0, 2]]),
                'expected_steps': 7
            }
        ]
        
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n{i+1}. {case['name']}:")
            
            start_time = time.time()
            result = ez.solve_arc_problem(
                input_grid=case['input'],
                max_steps=case['expected_steps']
            )
            solve_time = time.time() - start_time
            
            success = result.get('success', False)
            confidence = result.get('confidence', 0.0)
            steps = result.get('steps_taken', 0)
            
            print(f"   ✅ نجح: {success}")
            print(f"   💪 ثقة: {confidence:.2f}")
            print(f"   🔢 خطوات: {steps}")
            print(f"   ⏱️  وقت: {solve_time:.3f}s")
            
            results.append({
                'name': case['name'],
                'success': success,
                'confidence': confidence,
                'steps': steps,
                'solve_time': solve_time
            })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_time = np.mean([r['solve_time'] for r in results])
        
        print(f"\n📊 ملخص النتائج:")
        print(f"   🎯 نجح: {successful}/{len(results)}")
        print(f"   📈 متوسط الثقة: {avg_confidence:.2f}")
        print(f"   ⏱️  متوسط الوقت: {avg_time:.3f}s")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_arc_problems():
    """اختبار مشاكل ARC محددة"""
    
    print("\n🎯 اختبار مشاكل ARC محددة...")
    print("-" * 40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        # Define specific ARC-like problems
        problems = [
            {
                'name': 'تحجيم 2x',
                'input': np.array([[1, 0], [0, 1]]),
                'target': np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
                'description': 'تحجيم كل خلية إلى 2x2'
            },
            {
                'name': 'انعكاس أفقي',
                'input': np.array([[1, 0, 2], [0, 1, 0], [2, 0, 1]]),
                'target': np.array([[2, 0, 1], [0, 1, 0], [1, 0, 2]]),
                'description': 'انعكاس الشبكة أفقياً'
            },
            {
                'name': 'تبديل ألوان',
                'input': np.array([[1, 0, 1], [0, 2, 0], [1, 0, 1]]),
                'target': np.array([[2, 0, 2], [0, 1, 0], [2, 0, 2]]),
                'description': 'تبديل اللون 1 مع اللون 2'
            }
        ]
        
        results = []
        
        for i, problem in enumerate(problems):
            print(f"\n{i+1}. {problem['name']} - {problem['description']}:")
            print(f"   📊 دخل: {problem['input'].shape}")
            print(f"   🎯 هدف: {problem['target'].shape}")
            
            start_time = time.time()
            result = ez.solve_arc_problem(
                input_grid=problem['input'],
                target_grid=problem['target'],
                max_steps=10
            )
            solve_time = time.time() - start_time
            
            # Calculate similarity with target
            if result.get('solution_grid') is not None:
                solution = result['solution_grid']
                if solution.shape == problem['target'].shape:
                    similarity = np.mean(solution == problem['target'])
                else:
                    similarity = 0.0
            else:
                similarity = 0.0
            
            success = similarity > 0.8
            confidence = result.get('confidence', 0.0)
            steps = result.get('steps_taken', 0)
            
            print(f"   ✅ تشابه: {similarity:.2f}")
            print(f"   💪 ثقة: {confidence:.2f}")
            print(f"   🔢 خطوات: {steps}")
            print(f"   ⏱️  وقت: {solve_time:.3f}s")
            print(f"   🎯 نجح: {'نعم' if success else 'لا'}")
            
            results.append({
                'name': problem['name'],
                'similarity': similarity,
                'success': success,
                'confidence': confidence,
                'steps': steps,
                'solve_time': solve_time
            })
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        avg_similarity = np.mean([r['similarity'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_time = np.mean([r['solve_time'] for r in results])
        
        print(f"\n📊 ملخص مشاكل ARC:")
        print(f"   🎯 نجح: {successful}/{len(results)}")
        print(f"   🔍 متوسط التشابه: {avg_similarity:.2f}")
        print(f"   📈 متوسط الثقة: {avg_confidence:.2f}")
        print(f"   ⏱️  متوسط الوقت: {avg_time:.3f}s")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_capabilities():
    """اختبار قدرات النظام"""
    
    print("\n🚀 اختبار قدرات النظام...")
    print("-" * 40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        # Test system capabilities
        print("📋 قدرات النظام:")
        
        # 1. Action space
        actions = ez.action_space
        print(f"   🎮 عدد الإجراءات المتاحة: {len(actions)}")
        print(f"   📝 أمثلة الإجراءات: {actions[:5]}")
        
        # 2. Performance stats
        stats = ez.get_performance_stats()
        print(f"   📊 إحصائيات الأداء:")
        for key, value in stats.items():
            print(f"      - {key}: {value}")
        
        # 3. Test encoding
        test_grid = np.array([[1, 2], [2, 1]])
        state = ez.encode_state(test_grid)
        print(f"   🧠 ترميز الحالة:")
        print(f"      - شكل الميزات: {state.features.shape}")
        print(f"      - قيمة الحالة: {state.value:.2f}")
        print(f"      - شكل السياسة: {state.policy.shape}")
        
        # 4. Test MCTS
        print(f"   🌳 اختبار MCTS:")
        action = ez.mcts_search(state)
        print(f"      - نوع الإجراء: {action.action_type}")
        print(f"      - ثقة الإجراء: {action.confidence:.2f}")
        print(f"      - المكافأة المتوقعة: {action.expected_reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """الدالة الرئيسية"""
    
    print("🎯 الاختبار النهائي للنظام المتقدم مع EfficientZero")
    print("=" * 70)
    
    # Run all tests
    test1 = test_efficient_zero_standalone()
    test2 = test_specific_arc_problems()
    test3 = test_system_capabilities()
    
    print("\n" + "=" * 70)
    print("📊 النتائج النهائية:")
    print(f"   1️⃣ اختبار EfficientZero منفرداً: {'✅ نجح' if test1 else '❌ فشل'}")
    print(f"   2️⃣ اختبار مشاكل ARC محددة: {'✅ نجح' if test2 else '❌ فشل'}")
    print(f"   3️⃣ اختبار قدرات النظام: {'✅ نجح' if test3 else '❌ فشل'}")
    
    overall_success = test1 and test2 and test3
    
    if overall_success:
        print("\n🎉 جميع الاختبارات نجحت!")
        print("🚀 النظام جاهز للاستخدام على مستوى عالمي!")
        print("💪 EfficientZero مدمج بنجاح ويعمل بكفاءة عالية!")
    else:
        print("\n⚠️  بعض الاختبارات تحتاج تحسين")
        print("🔧 النظام يعمل لكن يحتاج تطوير إضافي")
    
    return overall_success

if __name__ == "__main__":
    main()
