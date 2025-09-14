from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبار النظام على جميع مهام التدريب
"""

import json
import numpy as np
import time
from pathlib import Path

def load_training_data():
    """تحميل بيانات التدريب"""
    
    print("📂 تحميل بيانات التدريب...")
    
    try:
        # Load challenges
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        # Load solutions  
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        
        print(f"   ✅ تم تحميل {len(challenges)} مهمة تدريب")
        return challenges, solutions
        
    except Exception as e:
        print(f"   ❌ فشل في تحميل البيانات: {e}")
        return None, None

def solve_with_efficient_zero(input_grid, task_id="unknown"):
    """حل مهمة باستخدام EfficientZero"""
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        
        ez = EfficientZeroEngine()
        
        # Convert to numpy array
        if isinstance(input_grid, list):
            input_grid = np.array(input_grid)
        
        # Solve
        start_time = time.time()
        result = ez.solve_arc_problem(input_grid, max_steps=5)
        solve_time = time.time() - start_time
        
        return {
            'success': True,
            'output_grid': result.get('solution_grid', input_grid).tolist(),
            'confidence': result.get('confidence', 0.0),
            'solve_time': solve_time,
            'method': 'efficient_zero'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'method': 'efficient_zero'
        }

def solve_with_fallback_methods(input_grid, task_id="unknown"):
    """حل مهمة باستخدام طرق احتياطية"""
    
    # Simple pattern-based solutions
    try:
        input_array = np.array(input_grid)
        
        # Method 1: Identity (no change)
        if np.random.random() > 0.7:
            return {
                'success': True,
                'output_grid': input_grid,
                'confidence': 0.3,
                'solve_time': 0.001,
                'method': 'identity'
            }
        
        # Method 2: Simple transformations
        transformations = [
            ('scale_2x', lambda x: np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)),
            ('rotate_90', lambda x: np.rot90(x)),
            ('flip_h', lambda x: np.fliplr(x)),
            ('flip_v', lambda x: np.flipud(x))
        ]
        
        # Try a random transformation
        transform_name, transform_func = transformations[np.random.randint(len(transformations))]
        
        try:
            output = transform_func(input_array)
            return {
                'success': True,
                'output_grid': output.tolist(),
                'confidence': 0.2,
                'solve_time': 0.002,
                'method': f'simple_{transform_name}'
            }
        except:
            pass
        
        # Method 3: Return input as fallback
        return {
            'success': True,
            'output_grid': input_grid,
            'confidence': 0.1,
            'solve_time': 0.001,
            'method': 'fallback'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'method': 'fallback'
        }

def calculate_similarity(grid1, grid2):
    """حساب التشابه بين شبكتين"""
    
    try:
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)
        
        # Check if same shape
        if arr1.shape != arr2.shape:
            return 0.0
        
        # Calculate pixel-wise similarity
        total_pixels = arr1.size
        matching_pixels = np.sum(arr1 == arr2)
        
        return matching_pixels / total_pixels
        
    except:
        return 0.0

def test_all_training_tasks():
    """اختبار جميع مهام التدريب"""
    
    print("🧪 اختبار النظام على جميع مهام التدريب")
    print("="*60)
    
    # Load data
    challenges, solutions = load_training_data()
    if not challenges or not solutions:
        print("❌ لا يمكن تحميل البيانات")
        return
    
    # Test parameters
    max_tasks_to_test = 50  # Test first 50 tasks for speed
    results = []
    
    print(f"🎯 اختبار أول {max_tasks_to_test} مهمة...")
    print("-" * 60)
    
    task_ids = list(challenges.keys())[:max_tasks_to_test]
    
    for i, task_id in enumerate(task_ids):
        print(f"{i+1:2d}. مهمة {task_id[:8]}...")
        
        try:
            challenge = challenges[task_id]
            solution = solutions[task_id]
            
            # Get first test case
            if not challenge.get('test'):
                print(f"     ⚠️  لا توجد حالات اختبار")
                continue
            
            test_case = challenge['test'][0]
            input_grid = test_case['input']
            expected_output = solution[0]  # First solution
            
            # Try EfficientZero first
            result = solve_with_efficient_zero(input_grid, task_id)
            
            # If EfficientZero fails, try fallback
            if not result['success']:
                result = solve_with_fallback_methods(input_grid, task_id)
            
            # Calculate similarity with expected output
            if result['success']:
                similarity = calculate_similarity(result['output_grid'], expected_output)
                result['similarity'] = similarity
                result['expected_output'] = expected_output
                
                # Determine if solved correctly
                result['solved_correctly'] = similarity >= 0.99
                
                # Print result
                status = "✅" if result['solved_correctly'] else f"📊 {similarity:.2f}"
                method = result['method']
                confidence = result.get('confidence', 0)
                
                print(f"     {status} {method} (ثقة: {confidence:.2f})")
                
            else:
                result['similarity'] = 0.0
                result['solved_correctly'] = False
                print(f"     ❌ فشل: {result.get('error', 'خطأ غير معروف')}")
            
            result['task_id'] = task_id
            results.append(result)
            
        except Exception as e:
            print(f"     ❌ خطأ في المهمة: {e}")
            results.append({
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'solved_correctly': False,
                'similarity': 0.0
            })
    
    # Calculate statistics
    print("\n" + "="*60)
    print("📊 إحصائيات النتائج:")
    print("-" * 60)
    
    total_tasks = len(results)
    successful_attempts = sum(1 for r in results if r.get('success', False))
    correctly_solved = sum(1 for r in results if r.get('solved_correctly', False))
    
    if total_tasks > 0:
        success_rate = successful_attempts / total_tasks * 100
        solve_rate = correctly_solved / total_tasks * 100
        
        # Calculate average similarity
        similarities = [r.get('similarity', 0) for r in results if r.get('success', False)]
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in results if r.get('success', False)]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Method breakdown
        methods = {}
        for r in results:
            if r.get('success', False):
                method = r.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
        
        print(f"📈 معدل المحاولات الناجحة: {successful_attempts}/{total_tasks} ({success_rate:.1f}%)")
        print(f"🎯 معدل الحلول الصحيحة: {correctly_solved}/{total_tasks} ({solve_rate:.1f}%)")
        print(f"📊 متوسط التشابه: {avg_similarity:.3f}")
        print(f"🔮 متوسط الثقة: {avg_confidence:.3f}")
        
        print(f"\n🔧 توزيع الطرق:")
        for method, count in methods.items():
            percentage = count / successful_attempts * 100 if successful_attempts > 0 else 0
            print(f"   - {method}: {count} ({percentage:.1f}%)")
        
        # Best results
        best_results = sorted([r for r in results if r.get('similarity', 0) > 0.5], 
                             key=lambda x: x.get('similarity', 0), reverse=True)[:5]
        
        if best_results:
            print(f"\n🏆 أفضل النتائج:")
            for i, r in enumerate(best_results):
                task_id = r['task_id'][:8]
                similarity = r.get('similarity', 0)
                method = r.get('method', 'unknown')
                print(f"   {i+1}. {task_id}: {similarity:.3f} ({method})")
    
    # Save results
    timestamp = int(time.time())
    results_file = f"all_training_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 النتائج محفوظة في: {results_file}")
    except Exception as e:
        print(f"\n⚠️  فشل في حفظ النتائج: {e}")
    
    return results

if __name__ == "__main__":
    results = test_all_training_tasks()
    
    if results:
        correctly_solved = sum(1 for r in results if r.get('solved_correctly', False))
        total = len(results)
        
        print(f"\n🎉 النتيجة النهائية: {correctly_solved}/{total} مهمة حُلت بشكل صحيح!")
        
        if correctly_solved > 0:
            print("✅ النظام قادر على حل بعض المهام!")
        else:
            print("⚠️  النظام يحتاج تحسينات لحل المهام بدقة أكبر")
    else:
        print("❌ لم يتم اختبار أي مهام")
