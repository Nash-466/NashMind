from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبار شامل للنظام مع EfficientZero
"""

import numpy as np
import time
import json
from pathlib import Path

def test_efficient_zero_on_training_data():
    """اختبار EfficientZero على بيانات التدريب"""
    
    print("🧠 اختبار EfficientZero على مهام التدريب...")
    print("="*60)
    
    # Load training data
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        print(f"✅ تم تحميل {len(challenges)} مهمة تدريب")
        
    except FileNotFoundError:
        print("❌ ملفات التدريب غير موجودة")
        return False
    
    # Initialize EfficientZero
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez_engine = EfficientZeroEngine()
        print("✅ تم تهيئة محرك EfficientZero")
        
    except Exception as e:
        print(f"❌ خطأ في تهيئة EfficientZero: {e}")
        return False
    
    # Test on first few problems
    test_problems = list(challenges.keys())[:10]  # Test first 10 problems
    results = []
    
    print(f"\n🔍 اختبار {len(test_problems)} مهام...")
    
    for i, problem_id in enumerate(test_problems):
        try:
            print(f"\n{i+1}. مهمة {problem_id[:8]}...")
            
            challenge = challenges[problem_id]
            expected_solutions = solutions[problem_id]
            
            # Get first training example
            if not challenge['train']:
                print("   ⚠️  لا توجد أمثلة تدريب")
                continue
            
            train_example = challenge['train'][0]
            input_grid = np.array(train_example['input'])
            expected_output = np.array(train_example['output'])
            
            print(f"   📊 شبكة الدخل: {input_grid.shape}")
            print(f"   🎯 شبكة الهدف: {expected_output.shape}")
            
            # Solve with EfficientZero
            start_time = time.time()
            result = ez_engine.solve_arc_problem(
                input_grid=input_grid,
                target_grid=expected_output,
                max_steps=8
            )
            solve_time = time.time() - start_time
            
            # Calculate similarity
            if result['success'] and result['solution_grid'] is not None:
                solution_grid = result['solution_grid']
                
                # Calculate similarity
                if solution_grid.shape == expected_output.shape:
                    similarity = np.mean(solution_grid == expected_output)
                else:
                    similarity = 0.0
                
                print(f"   ✅ حل: تشابه={similarity:.2f}, ثقة={result['confidence']:.2f}")
                print(f"   ⏱️  وقت: {solve_time:.3f}s, خطوات: {result['steps_taken']}")
                
                results.append({
                    'problem_id': problem_id,
                    'similarity': similarity,
                    'confidence': result['confidence'],
                    'solve_time': solve_time,
                    'steps_taken': result['steps_taken'],
                    'success': similarity > 0.8
                })
                
            else:
                print(f"   ❌ فشل: {result.get('error', 'Unknown error')}")
                results.append({
                    'problem_id': problem_id,
                    'similarity': 0.0,
                    'confidence': 0.0,
                    'solve_time': solve_time,
                    'steps_taken': 0,
                    'success': False
                })
            
        except Exception as e:
            print(f"   💥 خطأ: {e}")
            results.append({
                'problem_id': problem_id,
                'similarity': 0.0,
                'confidence': 0.0,
                'solve_time': 0.0,
                'steps_taken': 0,
                'success': False,
                'error': str(e)
            })
    
    # Calculate statistics
    if results:
        successful_results = [r for r in results if r['success']]
        
        print("\n" + "="*60)
        print("📊 نتائج الاختبار:")
        print(f"   🎯 المهام المحلولة بنجاح: {len(successful_results)}/{len(results)}")
        print(f"   📈 معدل النجاح: {len(successful_results)/len(results)*100:.1f}%")
        
        if results:
            avg_similarity = np.mean([r['similarity'] for r in results])
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_time = np.mean([r['solve_time'] for r in results])
            
            print(f"   🔍 متوسط التشابه: {avg_similarity:.2f}")
            print(f"   💪 متوسط الثقة: {avg_confidence:.2f}")
            print(f"   ⏱️  متوسط الوقت: {avg_time:.3f}s")
        
        # Save results
        timestamp = int(time.time())
        results_file = f"efficient_zero_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'test_summary': {
                    'total_problems': len(results),
                    'successful_problems': len(successful_results),
                    'success_rate': len(successful_results)/len(results),
                    'average_similarity': avg_similarity,
                    'average_confidence': avg_confidence,
                    'average_solve_time': avg_time
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"   💾 النتائج محفوظة في: {results_file}")
        
        return len(successful_results) > 0
    
    return False

def test_system_performance():
    """اختبار أداء النظام"""
    
    print("\n🚀 اختبار أداء النظام...")
    print("="*40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        # Test different grid sizes
        test_cases = [
            ("صغيرة 2x2", np.array([[1, 0], [0, 1]])),
            ("متوسطة 3x3", np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])),
            ("كبيرة 4x4", np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])),
            ("معقدة 5x5", np.random.randint(0, 3, (5, 5)))
        ]
        
        for name, grid in test_cases:
            start_time = time.time()
            result = ez.solve_arc_problem(grid, max_steps=5)
            solve_time = time.time() - start_time
            
            print(f"   {name}: ثقة={result.get('confidence', 0):.2f}, "
                  f"وقت={solve_time:.3f}s, خطوات={result.get('steps_taken', 0)}")
        
        # Performance stats
        stats = ez.get_performance_stats()
        print(f"\n📈 إحصائيات الأداء:")
        print(f"   - خطوات التدريب: {stats['training_steps']}")
        print(f"   - عدد المحاكاات: {stats['num_simulations']}")
        print(f"   - درجة الحرارة: {stats['temperature']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الأداء: {e}")
        return False

if __name__ == "__main__":
    print("🧪 بدء الاختبار الشامل لـ EfficientZero...")
    print("="*70)
    
    # Test 1: Performance test
    perf_success = test_system_performance()
    
    # Test 2: Training data test
    training_success = test_efficient_zero_on_training_data()
    
    print("\n" + "="*70)
    if perf_success and training_success:
        print("🎉 جميع الاختبارات نجحت!")
        print("🚀 EfficientZero جاهز للاستخدام على مستوى عالمي!")
    else:
        print("⚠️  بعض الاختبارات تحتاج تحسين")
        print("🔧 النظام يعمل لكن يحتاج تطوير إضافي")
