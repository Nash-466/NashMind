from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 تقييم النظام المتقدم على مهام التدريب
=======================================

هذا السكريبت يقوم بتشغيل النظام المتقدم على جميع مهام التدريب
ومقارنة النتائج بالحلول الصحيحة لحساب معدل النجاح.
"""

import json
import numpy as np
import time
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from ultra_advanced_arc_system import solve_arc_problem
    SYSTEM_AVAILABLE = True
    logging.info("✅ Ultra Advanced ARC System loaded successfully")
except ImportError as e:
    logging.error(f"❌ Failed to import Ultra Advanced ARC System: {e}")
    SYSTEM_AVAILABLE = False


def load_training_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """تحميل بيانات التدريب والحلول"""
    
    challenges_file = Path("arc-agi_training_challenges.json")
    solutions_file = Path("arc-agi_training_solutions.json")
    
    if not challenges_file.exists():
        raise FileNotFoundError(f"Training challenges file not found: {challenges_file}")
    
    if not solutions_file.exists():
        raise FileNotFoundError(f"Training solutions file not found: {solutions_file}")
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    with open(solutions_file, 'r') as f:
        solutions = json.load(f)
    
    logging.info(f"📚 Loaded {len(challenges)} training challenges")
    logging.info(f"📝 Loaded {len(solutions)} training solutions")
    
    return challenges, solutions


def arrays_equal(arr1: np.ndarray, arr2: List[List[int]]) -> bool:
    """مقارنة مصفوفة numpy مع قائمة ثنائية الأبعاد"""
    try:
        arr2_np = np.array(arr2)
        return np.array_equal(arr1, arr2_np)
    except:
        return False


def evaluate_solution(predicted: np.ndarray, expected: List[List[List[int]]]) -> bool:
    """تقييم الحل المتوقع مقابل الحلول الصحيحة"""
    
    # ARC problems can have multiple correct solutions
    for correct_solution in expected:
        if arrays_equal(predicted, correct_solution):
            return True
    
    return False


def calculate_similarity(predicted: np.ndarray, expected: List[List[int]]) -> float:
    """حساب نسبة التشابه بين الحل المتوقع والصحيح"""
    try:
        expected_np = np.array(expected)
        
        # Check if shapes match
        if predicted.shape != expected_np.shape:
            return 0.0
        
        # Calculate percentage of matching cells
        matching_cells = np.sum(predicted == expected_np)
        total_cells = predicted.size
        
        return matching_cells / total_cells
    except:
        return 0.0


def run_evaluation(max_problems: int = None) -> Dict[str, Any]:
    """تشغيل التقييم الشامل"""
    
    if not SYSTEM_AVAILABLE:
        logging.error("❌ Cannot run evaluation - system not available")
        return {"error": "System not available"}
    
    # Load data
    try:
        challenges, solutions = load_training_data()
    except Exception as e:
        logging.error(f"❌ Failed to load training data: {e}")
        return {"error": f"Failed to load data: {e}"}
    
    # Initialize results
    results = {
        "total_problems": 0,
        "solved_correctly": 0,
        "partial_matches": 0,
        "failed": 0,
        "success_rate": 0.0,
        "average_similarity": 0.0,
        "average_time": 0.0,
        "problem_results": [],
        "errors": []
    }
    
    problem_count = 0
    total_time = 0
    total_similarity = 0

    # Progress file path
    progress_path = Path("training_eval_progress.json")
    
    # Process each problem
    for problem_id, challenge_data in challenges.items():
        
        if max_problems and problem_count >= max_problems:
            break
        
        problem_count += 1
        
        # Get test input
        test_cases = challenge_data.get("test", [])
        if not test_cases:
            logging.warning(f"⚠️  No test cases for problem {problem_id}")
            continue
        
        test_input = test_cases[0]["input"]  # Take first test case
        
        # Get expected solution
        if problem_id not in solutions:
            logging.warning(f"⚠️  No solution found for problem {problem_id}")
            continue
        
        expected_solutions = solutions[problem_id]
        
        logging.info(f"🧪 Processing problem {problem_count}: {problem_id}")
        print(f"🔄 Processing {problem_count}/50: {problem_id}")
        
        try:
            # Convert input to numpy array
            input_grid = np.array(test_input)
            
            # Solve using our system
            start_time = time.time()
            solution = solve_arc_problem(input_grid)
            solve_time = time.time() - start_time
            
            total_time += solve_time
            
            # Evaluate solution
            is_correct = evaluate_solution(solution.solution_grid, expected_solutions)
            
            # Calculate similarity with first expected solution
            similarity = calculate_similarity(solution.solution_grid, expected_solutions[0])
            total_similarity += similarity
            
            # Record result
            problem_result = {
                "problem_id": problem_id,
                "correct": is_correct,
                "similarity": similarity,
                "confidence": solution.confidence,
                "solve_time": solve_time,
                "input_shape": input_grid.shape,
                "output_shape": solution.solution_grid.shape,
                "expected_shape": np.array(expected_solutions[0]).shape
            }
            
            results["problem_results"].append(problem_result)
            
            if is_correct:
                results["solved_correctly"] += 1
                logging.info(f"✅ Problem {problem_id} solved correctly!")
                print(f"✅ {problem_count}/50: {problem_id} - SOLVED! (time: {solve_time:.2f}s)")
            elif similarity > 0.5:
                results["partial_matches"] += 1
                logging.info(f"🔶 Problem {problem_id} partially correct (similarity: {similarity:.2f})")
                print(f"🔶 {problem_count}/50: {problem_id} - PARTIAL ({similarity:.2f}) (time: {solve_time:.2f}s)")
            else:
                results["failed"] += 1
                logging.info(f"❌ Problem {problem_id} failed (similarity: {similarity:.2f})")
                print(f"❌ {problem_count}/50: {problem_id} - FAILED ({similarity:.2f}) (time: {solve_time:.2f}s)")

            # Write incremental progress after each problem
            try:
                progress = {
                    "processed": problem_count,
                    "total_target": max_problems or len(challenges),
                    "solved_correctly": results["solved_correctly"],
                    "partial_matches": results["partial_matches"],
                    "failed": results["failed"],
                    "avg_time_so_far": (total_time / problem_count) if problem_count else 0.0,
                    "avg_similarity_so_far": (total_similarity / problem_count) if problem_count else 0.0,
                    "last_problem_id": problem_id,
                }
                with open(progress_path, 'w', encoding='utf-8') as pf:
                    json.dump(progress, pf, ensure_ascii=False, indent=2)
            except Exception as pe:
                logging.debug(f"Progress write failed: {pe}")
        
        except Exception as e:
            logging.error(f"❌ Error processing problem {problem_id}: {e}")
            print(f"💥 {problem_count}/50: {problem_id} - ERROR: {str(e)[:50]}...")
            results["failed"] += 1
            results["errors"].append({
                "problem_id": problem_id,
                "error": str(e)
            })
            # Also attempt to write progress on error
            try:
                progress = {
                    "processed": problem_count,
                    "total_target": max_problems or len(challenges),
                    "solved_correctly": results["solved_correctly"],
                    "partial_matches": results["partial_matches"],
                    "failed": results["failed"],
                    "avg_time_so_far": (total_time / problem_count) if problem_count else 0.0,
                    "avg_similarity_so_far": (total_similarity / problem_count) if problem_count else 0.0,
                    "last_problem_id": problem_id,
                }
                with open(progress_path, 'w', encoding='utf-8') as pf:
                    json.dump(progress, pf, ensure_ascii=False, indent=2)
            except Exception:
                pass
    
    # Calculate final statistics
    results["total_problems"] = problem_count
    if problem_count > 0:
        results["success_rate"] = results["solved_correctly"] / problem_count
        results["average_similarity"] = total_similarity / problem_count
        results["average_time"] = total_time / problem_count
    
    return results


def print_results(results: Dict[str, Any]):
    """طباعة النتائج بشكل منسق"""
    
    print("\n" + "="*80)
    print("🏆 نتائج تقييم النظام المتقدم على مهام التدريب")
    print("="*80)
    
    if "error" in results:
        print(f"❌ خطأ: {results['error']}")
        return
    
    total = results["total_problems"]
    correct = results["solved_correctly"]
    partial = results["partial_matches"]
    failed = results["failed"]
    
    print(f"📊 إجمالي المهام المختبرة: {total}")
    print(f"✅ المهام المحلولة بشكل صحيح: {correct}")
    print(f"🔶 المهام المحلولة جزئياً: {partial}")
    print(f"❌ المهام الفاشلة: {failed}")
    print()
    
    print(f"🎯 معدل النجاح: {results['success_rate']:.1%}")
    print(f"📈 متوسط التشابه: {results['average_similarity']:.3f}")
    print(f"⏱️  متوسط وقت الحل: {results['average_time']:.3f} ثانية")
    print()
    
    if results["errors"]:
        print(f"⚠️  عدد الأخطاء: {len(results['errors'])}")
        print("أول 3 أخطاء:")
        for error in results["errors"][:3]:
            print(f"  - {error['problem_id']}: {error['error']}")
        print()
    
    # Show best and worst performing problems
    if results["problem_results"]:
        sorted_results = sorted(results["problem_results"], 
                              key=lambda x: x["similarity"], reverse=True)
        
        print("🏅 أفضل 5 نتائج:")
        for i, result in enumerate(sorted_results[:5], 1):
            status = "✅" if result["correct"] else f"🔶 ({result['similarity']:.2f})"
            print(f"  {i}. {result['problem_id']}: {status}")
        
        print("\n📉 أسوأ 5 نتائج:")
        for i, result in enumerate(sorted_results[-5:], 1):
            status = "❌" if result["similarity"] < 0.1 else f"🔶 ({result['similarity']:.2f})"
            print(f"  {i}. {result['problem_id']}: {status}")
    
    print("\n" + "="*80)


def save_detailed_results(results: Dict[str, Any], filename: str = None):
    """حفظ النتائج التفصيلية في ملف JSON"""
    
    if filename is None:
        timestamp = int(time.time())
        filename = f"training_evaluation_results_{timestamp}.json"
    
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = results.copy()
        for problem_result in json_results.get("problem_results", []):
            for key, value in problem_result.items():
                if isinstance(value, np.ndarray):
                    problem_result[key] = value.tolist()
                elif isinstance(value, tuple):
                    problem_result[key] = list(value)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"💾 Results saved to: {filename}")
        return filename
    
    except Exception as e:
        logging.error(f"❌ Failed to save results: {e}")
        return None


def main():
    """الدالة الرئيسية"""
    
    print("🚀 بدء تقييم النظام المتقدم على مهام التدريب...")
    print("=" * 60)
    
    # Run evaluation
    start_time = time.time()
    
    # Test on first 20 problems for quick evaluation
    # Remove max_problems=20 to test all problems
    results = run_evaluation(max_problems=20)
    
    total_time = time.time() - start_time
    
    # Print results
    print_results(results)
    
    # Save detailed results
    if "error" not in results:
        filename = save_detailed_results(results)
        if filename:
            print(f"📄 النتائج التفصيلية محفوظة في: {filename}")
    
    print(f"\n⏱️  إجمالي وقت التقييم: {total_time:.2f} ثانية")
    print("🎉 انتهى التقييم!")


if __name__ == "__main__":
    main()
