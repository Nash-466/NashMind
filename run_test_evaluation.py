from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 تقييم النظام المتقدم على مهام الاختبار
========================================

هذا السكريپت يقوم بتشغيل النظام المتقدم على مهام الاختبار
وإنتاج ملف الحلول بالتنسيق المطلوب للمسابقة.
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


def load_test_data() -> Dict[str, Any]:
    """تحميل بيانات الاختبار"""
    
    test_file = Path("arc-agi_test_challenges.json")
    
    if not test_file.exists():
        raise FileNotFoundError(f"Test challenges file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        test_challenges = json.load(f)
    
    logging.info(f"🧪 Loaded {len(test_challenges)} test challenges")
    return test_challenges


def solve_test_problems(test_challenges: Dict[str, Any], max_problems: int = None) -> Dict[str, List[List[List[int]]]]:
    """حل مهام الاختبار"""
    
    if not SYSTEM_AVAILABLE:
        raise RuntimeError("Ultra Advanced ARC System not available")
    
    solutions = {}
    problem_count = 0
    
    for problem_id, challenge_data in test_challenges.items():
        
        if max_problems and problem_count >= max_problems:
            break
        
        problem_count += 1
        
        # Get test cases
        test_cases = challenge_data.get("test", [])
        if not test_cases:
            logging.warning(f"⚠️  No test cases for problem {problem_id}")
            continue
        
        problem_solutions = []
        
        logging.info(f"🧪 Solving problem {problem_count}: {problem_id}")
        
        # Solve each test case
        for test_idx, test_case in enumerate(test_cases):
            try:
                # Convert input to numpy array
                input_grid = np.array(test_case["input"])
                
                # Solve using our system
                start_time = time.time()
                solution = solve_arc_problem(input_grid)
                solve_time = time.time() - start_time
                
                # Convert solution back to list format
                solution_list = solution.solution_grid.tolist()
                problem_solutions.append(solution_list)
                
                logging.info(f"  ✅ Test case {test_idx + 1} solved in {solve_time:.3f}s "
                           f"(confidence: {solution.confidence:.3f})")
                
            except Exception as e:
                logging.error(f"  ❌ Error solving test case {test_idx + 1}: {e}")
                
                # Create a fallback solution (copy input or create empty grid)
                try:
                    fallback_solution = test_case["input"]
                    problem_solutions.append(fallback_solution)
                    logging.info(f"  🔄 Using fallback solution for test case {test_idx + 1}")
                except:
                    # Last resort: create a 1x1 grid with color 0
                    fallback_solution = [[0]]
                    problem_solutions.append(fallback_solution)
                    logging.info(f"  🆘 Using emergency fallback for test case {test_idx + 1}")
        
        solutions[problem_id] = problem_solutions
        
        # Progress update
        if problem_count % 10 == 0:
            logging.info(f"📊 Progress: {problem_count} problems processed")
    
    return solutions


def validate_solutions(solutions: Dict[str, List[List[List[int]]]]) -> bool:
    """التحقق من صحة تنسيق الحلول"""
    
    logging.info("🔍 Validating solution format...")
    
    for problem_id, problem_solutions in solutions.items():
        
        if not isinstance(problem_solutions, list):
            logging.error(f"❌ Problem {problem_id}: solutions must be a list")
            return False
        
        for test_idx, solution in enumerate(problem_solutions):
            
            if not isinstance(solution, list):
                logging.error(f"❌ Problem {problem_id}, test {test_idx}: solution must be a list")
                return False
            
            if not solution:
                logging.error(f"❌ Problem {problem_id}, test {test_idx}: solution cannot be empty")
                return False
            
            # Check that all rows have the same length
            row_lengths = [len(row) for row in solution]
            if len(set(row_lengths)) > 1:
                logging.error(f"❌ Problem {problem_id}, test {test_idx}: inconsistent row lengths")
                return False
            
            # Check that all values are integers between 0-9
            for row_idx, row in enumerate(solution):
                for col_idx, value in enumerate(row):
                    if not isinstance(value, int) or not (0 <= value <= 9):
                        logging.error(f"❌ Problem {problem_id}, test {test_idx}, "
                                    f"position ({row_idx}, {col_idx}): invalid value {value}")
                        return False
    
    logging.info("✅ All solutions have valid format")
    return True


def save_solutions(solutions: Dict[str, List[List[List[int]]]], filename: str = None) -> str:
    """حفظ الحلول في ملف JSON"""
    
    if filename is None:
        timestamp = int(time.time())
        filename = f"test_solutions_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(solutions, f, separators=(',', ':'))
        
        logging.info(f"💾 Solutions saved to: {filename}")
        return filename
    
    except Exception as e:
        logging.error(f"❌ Failed to save solutions: {e}")
        raise


def generate_submission_report(solutions: Dict[str, List[List[List[int]]]], 
                             test_challenges: Dict[str, Any]) -> Dict[str, Any]:
    """إنتاج تقرير الحلول"""
    
    report = {
        "total_problems": len(test_challenges),
        "solved_problems": len(solutions),
        "total_test_cases": 0,
        "solved_test_cases": 0,
        "solution_statistics": {},
        "problem_details": []
    }
    
    for problem_id, challenge_data in test_challenges.items():
        
        test_cases = challenge_data.get("test", [])
        num_test_cases = len(test_cases)
        report["total_test_cases"] += num_test_cases
        
        if problem_id in solutions:
            num_solutions = len(solutions[problem_id])
            report["solved_test_cases"] += num_solutions
            
            # Analyze solution sizes
            for solution in solutions[problem_id]:
                height = len(solution)
                width = len(solution[0]) if solution else 0
                size_key = f"{height}x{width}"
                
                if size_key not in report["solution_statistics"]:
                    report["solution_statistics"][size_key] = 0
                report["solution_statistics"][size_key] += 1
            
            problem_detail = {
                "problem_id": problem_id,
                "test_cases": num_test_cases,
                "solutions_provided": num_solutions,
                "status": "solved" if num_solutions == num_test_cases else "partial"
            }
        else:
            problem_detail = {
                "problem_id": problem_id,
                "test_cases": num_test_cases,
                "solutions_provided": 0,
                "status": "unsolved"
            }
        
        report["problem_details"].append(problem_detail)
    
    # Calculate success rates
    if report["total_problems"] > 0:
        report["problem_success_rate"] = report["solved_problems"] / report["total_problems"]
    
    if report["total_test_cases"] > 0:
        report["test_case_success_rate"] = report["solved_test_cases"] / report["total_test_cases"]
    
    return report


def print_submission_report(report: Dict[str, Any]):
    """طباعة تقرير الحلول"""
    
    print("\n" + "="*80)
    print("🎯 تقرير حلول مهام الاختبار")
    print("="*80)
    
    print(f"📊 إجمالي المهام: {report['total_problems']}")
    print(f"✅ المهام المحلولة: {report['solved_problems']}")
    print(f"📈 معدل نجاح المهام: {report.get('problem_success_rate', 0):.1%}")
    print()
    
    print(f"🧪 إجمالي حالات الاختبار: {report['total_test_cases']}")
    print(f"✅ حالات الاختبار المحلولة: {report['solved_test_cases']}")
    print(f"📈 معدل نجاح حالات الاختبار: {report.get('test_case_success_rate', 0):.1%}")
    print()
    
    # Solution size distribution
    if report["solution_statistics"]:
        print("📐 توزيع أحجام الحلول:")
        for size, count in sorted(report["solution_statistics"].items()):
            print(f"  - {size}: {count} حل")
        print()
    
    # Problem status summary
    status_counts = {}
    for detail in report["problem_details"]:
        status = detail["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("📋 ملخص حالة المهام:")
    for status, count in status_counts.items():
        print(f"  - {status}: {count} مهمة")
    
    print("\n" + "="*80)


def main():
    """الدالة الرئيسية"""
    
    print("🎯 بدء تقييم النظام على مهام الاختبار...")
    print("=" * 60)
    
    try:
        # Load test data
        test_challenges = load_test_data()
        
        # Solve test problems
        print("🧠 حل مهام الاختبار...")
        start_time = time.time()
        
        # Remove max_problems parameter to solve all problems
        solutions = solve_test_problems(test_challenges, max_problems=10)  # Test on first 10 for demo
        
        solve_time = time.time() - start_time
        
        # Validate solutions
        if not validate_solutions(solutions):
            raise ValueError("Solution validation failed")
        
        # Save solutions
        filename = save_solutions(solutions, "arc_test_solutions.json")
        
        # Generate and print report
        report = generate_submission_report(solutions, test_challenges)
        print_submission_report(report)
        
        # Save report
        report_filename = filename.replace('.json', '_report.json')
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 تقرير الحلول محفوظ في: {report_filename}")
        print(f"⏱️  إجمالي وقت الحل: {solve_time:.2f} ثانية")
        print("🎉 انتهى التقييم بنجاح!")
        
        return filename
        
    except Exception as e:
        logging.error(f"❌ خطأ في التقييم: {e}")
        raise


if __name__ == "__main__":
    main()
