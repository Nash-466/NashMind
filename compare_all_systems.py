from __future__ import annotations
#!/usr/bin/env python3
"""
مقارنة شاملة لجميع أنظمة ARC
============================
"""
import json
import time
import numpy as np
from collections.abc import Callable
from typing import Dict, Any, List
import traceback

def load_training_data():
    """تحميل بيانات التدريب"""
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    return challenges, solutions

def arrays_equal(arr1: np.ndarray, arr2: List[List[int]]) -> bool:
    """مقارنة مصفوفة numpy مع قائمة ثنائية الأبعاد"""
    try:
        arr2_np = np.array(arr2)
        return np.array_equal(arr1, arr2_np)
    except:
        return False

def calculate_similarity(predicted: np.ndarray, expected: List[List[int]]) -> float:
    """حساب نسبة التشابه"""
    try:
        expected_np = np.array(expected)
        if predicted.shape != expected_np.shape:
            return 0.0
        matching_cells = np.sum(predicted == expected_np)
        return matching_cells / predicted.size
    except:
        return 0.0

def test_system(system_name: str, solve_func, challenges: Dict, solutions: Dict, max_problems: int = 10):
    """اختبار نظام واحد"""
    print(f"\n🧪 Testing {system_name}...")
    print("=" * 50)
    
    results = {
        "system_name": system_name,
        "total_problems": 0,
        "solved_correctly": 0,
        "partial_matches": 0,
        "failed": 0,
        "success_rate": 0.0,
        "average_similarity": 0.0,
        "average_time": 0.0,
        "total_time": 0.0,
        "problem_results": [],
        "errors": []
    }
    
    problem_count = 0
    total_similarity = 0
    total_time = 0
    
    for problem_id, challenge_data in list(challenges.items())[:max_problems]:
        problem_count += 1
        test_cases = challenge_data.get("test", [])
        if not test_cases:
            continue
            
        test_input = test_cases[0]["input"]
        expected_solutions = solutions.get(problem_id, [])
        
        print(f"🔄 {problem_count}/{max_problems}: {problem_id}")
        
        try:
            input_grid = np.array(test_input)
            start_time = time.time()
            
            # حل المهمة
            solution = solve_func(input_grid)
            solve_time = time.time() - start_time
            total_time += solve_time
            
            # تقييم الحل
            if hasattr(solution, 'solution_grid'):
                solution_grid = solution.solution_grid
            else:
                solution_grid = solution
                
            is_correct = evaluate_solution(solution_grid, expected_solutions)
            similarity = calculate_similarity(solution_grid, expected_solutions[0])
            total_similarity += similarity
            
            # تسجيل النتيجة
            problem_result = {
                "problem_id": problem_id,
                "correct": is_correct,
                "similarity": similarity,
                "solve_time": solve_time,
                "input_shape": input_grid.shape,
                "output_shape": solution_grid.shape,
                "expected_shape": np.array(expected_solutions[0]).shape
            }
            results["problem_results"].append(problem_result)
            
            if is_correct:
                results["solved_correctly"] += 1
                print(f"✅ {problem_id} - SOLVED! ({solve_time:.2f}s)")
            elif similarity > 0.5:
                results["partial_matches"] += 1
                print(f"🔶 {problem_id} - PARTIAL ({similarity:.2f}) ({solve_time:.2f}s)")
            else:
                results["failed"] += 1
                print(f"❌ {problem_id} - FAILED ({similarity:.2f}) ({solve_time:.2f}s)")
                
        except Exception as e:
            print(f"💥 {problem_id} - ERROR: {str(e)[:50]}...")
            results["failed"] += 1
            results["errors"].append({
                "problem_id": problem_id,
                "error": str(e)
            })
    
    # حساب الإحصائيات النهائية
    results["total_problems"] = problem_count
    if problem_count > 0:
        results["success_rate"] = results["solved_correctly"] / problem_count
        results["average_similarity"] = total_similarity / problem_count
        results["average_time"] = total_time / problem_count
    results["total_time"] = total_time
    
    return results

def evaluate_solution(predicted: np.ndarray, expected: List[List[List[int]]]) -> bool:
    """تقييم الحل"""
    for correct_solution in expected:
        if arrays_equal(predicted, correct_solution):
            return True
    return False

def main():
    """المقارنة الرئيسية"""
    print("🚀 مقارنة شاملة لجميع أنظمة ARC")
    print("=" * 60)
    
    # تحميل البيانات
    challenges, solutions = load_training_data()
    print(f"📚 تم تحميل {len(challenges)} مهمة تدريب")
    
    all_results = []
    
    # النظام 1: Ultra Advanced ARC System
    try:
        from ultra_advanced_arc_system import solve_arc_problem
        result1 = test_system("Ultra Advanced ARC System", solve_arc_problem, challenges, solutions, 10)
        all_results.append(result1)
    except Exception as e:
        print(f"❌ فشل تحميل Ultra Advanced ARC System: {e}")
    
    # النظام 2: Main System
    try:
        from main import build_orchestrator
        orchestrator = build_orchestrator('fast', 0)
        def solve_main(input_grid):
            # تحويل إلى تنسيق المهمة
            task_data = {
                'train': [],
                'test': [{'input': input_grid.tolist()}]
            }
            result = orchestrator.process_single_task(task_data)
            return result if result is not None else input_grid
            
        result2 = test_system("Main System (Orchestrator)", solve_main, challenges, solutions, 10)
        all_results.append(result2)
    except Exception as e:
        print(f"❌ فشل تحميل Main System: {e}")
    
    # النظام 3: MetaBrain System
    try:
        from burhan_meta_brain import MetaBrain
        meta_brain = MetaBrain()
        from main import build_orchestrator
        orchestrator = build_orchestrator('fast', 0)
        
        def solve_meta(input_grid):
            task_data = {
                'train': [],
                'test': [{'input': input_grid.tolist()}]
            }
            result = meta_brain.suggest_and_solve(task_data, orchestrator, mode='fast')
            return result if result is not None else input_grid
            
        result3 = test_system("MetaBrain System", solve_meta, challenges, solutions, 10)
        all_results.append(result3)
    except Exception as e:
        print(f"❌ فشل تحميل MetaBrain System: {e}")
    
    # النظام 4: Revolutionary ARC System
    try:
        from revolutionary_arc_system import solve_arc_problem as solve_revolutionary
        result4 = test_system("Revolutionary ARC System", solve_revolutionary, challenges, solutions, 10)
        all_results.append(result4)
    except Exception as e:
        print(f"❌ فشل تحميل Revolutionary ARC System: {e}")
    
    # النظام 5: Perfect ARC System
    try:
        from perfect_arc_system import solve_arc_problem as solve_perfect
        result5 = test_system("Perfect ARC System", solve_perfect, challenges, solutions, 10)
        all_results.append(result5)
    except Exception as e:
        print(f"❌ فشل تحميل Perfect ARC System: {e}")
    
    # النظام 6: Ultimate ARC System
    try:
        from ultimate_arc_system import solve_arc_problem as solve_ultimate
        result6 = test_system("Ultimate ARC System", solve_ultimate, challenges, solutions, 10)
        all_results.append(result6)
    except Exception as e:
        print(f"❌ فشل تحميل Ultimate ARC System: {e}")
    
    # النظام 7: Genius ARC Manager
    try:
        from genius_arc_manager import solve_arc_problem as solve_genius
        result7 = test_system("Genius ARC Manager", solve_genius, challenges, solutions, 10)
        all_results.append(result7)
    except Exception as e:
        print(f"❌ فشل تحميل Genius ARC Manager: {e}")
    
    # طباعة النتائج النهائية
    print("\n" + "="*80)
    print("🏆 نتائج المقارنة النهائية")
    print("="*80)
    
    for result in all_results:
        print(f"\n📊 {result['system_name']}:")
        print(f"   ✅ محلولة: {result['solved_correctly']}/{result['total_problems']}")
        print(f"   🔶 جزئياً: {result['partial_matches']}")
        print(f"   ❌ فاشلة: {result['failed']}")
        print(f"   📈 معدل النجاح: {result['success_rate']:.1%}")
        print(f"   📊 متوسط التشابه: {result['average_similarity']:.3f}")
        print(f"   ⏱️  متوسط الوقت: {result['average_time']:.2f}s")
        print(f"   🕐 إجمالي الوقت: {result['total_time']:.2f}s")
    
    # حفظ النتائج
    timestamp = int(time.time())
    filename = f"system_comparison_{timestamp}.json"
    
    # تحويل numpy arrays إلى lists للJSON
    json_results = []
    for result in all_results:
        json_result = result.copy()
        for problem_result in json_result.get("problem_results", []):
            for key, value in problem_result.items():
                if isinstance(value, np.ndarray):
                    problem_result[key] = value.tolist()
                elif isinstance(value, tuple):
                    problem_result[key] = list(value)
        json_results.append(json_result)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 النتائج محفوظة في: {filename}")

if __name__ == "__main__":
    main()
