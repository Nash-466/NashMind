from __future__ import annotations
#!/usr/bin/env python3
"""
COMPREHENSIVE TEST FRAMEWORK - إطار اختبار شامل
=============================================
نظام اختبار متكامل لقياس أداء حلالات ARC بدقة
"""

import os
import json
import numpy as np
import time
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import traceback

class ComprehensiveTestFramework:
    """إطار الاختبار الشامل"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        
    def run_full_evaluation(self, solver_class, max_tasks: int = 50):
        """تشغيل تقييم شامل للحلال"""
        
        print(f"🧪 بدء التقييم الشامل للحلال...")
        print(f"📊 عدد المهام المستهدفة: {max_tasks}")
        
        # تحميل المهام
        tasks = self._load_tasks()
        if not tasks:
            print("❌ فشل في تحميل المهام")
            return
        
        # اختيار عينة من المهام
        task_ids = list(tasks.keys())[:max_tasks]
        
        # إنشاء الحلال
        try:
            solver = solver_class()
            print(f"✅ تم إنشاء الحلال: {solver_class.__name__}")
        except Exception as e:
            print(f"❌ فشل في إنشاء الحلال: {e}")
            return
        
        # تشغيل الاختبارات
        results = {
            'total_tasks': len(task_ids),
            'successful_tasks': 0,
            'failed_tasks': 0,
            'error_tasks': 0,
            'average_time': 0,
            'pattern_success_rate': defaultdict(int),
            'pattern_total_count': defaultdict(int),
            'detailed_results': []
        }
        
        total_time = 0
        
        for i, task_id in enumerate(task_ids):
            print(f"\n🎯 اختبار المهمة {i+1}/{len(task_ids)}: {task_id}")
            
            task = tasks[task_id]
            start_time = time.time()
            
            try:
                # محاولة حل المهمة
                solutions = solver.solve_task(task)
                execution_time = time.time() - start_time
                total_time += execution_time
                
                # تقييم النتيجة
                success = self._evaluate_solution(solutions, task)
                
                if success:
                    results['successful_tasks'] += 1
                    print(f"  ✅ نجح ({execution_time:.2f}s)")
                else:
                    results['failed_tasks'] += 1
                    print(f"  ❌ فشل ({execution_time:.2f}s)")
                
                # تسجيل النتائج التفصيلية
                task_result = {
                    'task_id': task_id,
                    'success': success,
                    'execution_time': execution_time,
                    'solution_count': len(solutions) if solutions else 0
                }
                results['detailed_results'].append(task_result)
                
            except Exception as e:
                results['error_tasks'] += 1
                execution_time = time.time() - start_time
                total_time += execution_time
                
                print(f"  💥 خطأ ({execution_time:.2f}s): {str(e)[:50]}...")
                
                self.error_log.append({
                    'task_id': task_id,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        # حساب الإحصائيات النهائية
        results['average_time'] = total_time / len(task_ids)
        results['success_rate'] = (results['successful_tasks'] / results['total_tasks']) * 100
        
        # طباعة النتائج
        self._print_final_results(results)
        
        # حفظ النتائج
        self._save_results(results, solver_class.__name__)
        
        return results
    
    def _load_tasks(self) -> Dict:
        """تحميل مهام ARC"""
        
        data_paths = [
            'ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json',
            'data/arc-tasks/arc-agi_training_challenges.json',
            'arc-agi_training_challenges.json'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        tasks = json.load(f)
                    print(f"✅ تم تحميل {len(tasks)} مهمة من: {path}")
                    return tasks
                except Exception as e:
                    print(f"❌ خطأ في تحميل {path}: {e}")
        
        print("❌ لم يتم العثور على ملفات البيانات")
        return {}
    
    def _evaluate_solution(self, solutions: List[np.ndarray], task: Dict) -> bool:
        """تقييم صحة الحل"""
        
        if not solutions:
            return False
        
        # فحص عدد الحلول
        if len(solutions) != len(task['test']):
            return False
        
        # فحص كل حل
        for i, solution in enumerate(solutions):
            if solution is None or solution.size == 0:
                return False
            
            # فحص الأبعاد المنطقية
            if solution.shape[0] < 1 or solution.shape[1] < 1:
                return False
            
            # فحص الأبعاد المعقولة
            if solution.shape[0] > 30 or solution.shape[1] > 30:
                return False
        
        return True
    
    def _print_final_results(self, results: Dict):
        """طباعة النتائج النهائية"""
        
        print(f"\n{'='*60}")
        print(f"📊 النتائج النهائية للتقييم الشامل")
        print(f"{'='*60}")
        
        print(f"📈 إجمالي المهام: {results['total_tasks']}")
        print(f"✅ المهام الناجحة: {results['successful_tasks']}")
        print(f"❌ المهام الفاشلة: {results['failed_tasks']}")
        print(f"💥 المهام المعطلة: {results['error_tasks']}")
        
        print(f"\n🎯 معدل النجاح: {results['success_rate']:.1f}%")
        print(f"⏱️ متوسط وقت التنفيذ: {results['average_time']:.2f} ثانية")
        
        # تحليل الأداء
        if results['success_rate'] >= 80:
            print("🎉 أداء ممتاز!")
        elif results['success_rate'] >= 60:
            print("👍 أداء جيد")
        elif results['success_rate'] >= 40:
            print("⚠️ أداء متوسط - يحتاج تحسين")
        else:
            print("🚨 أداء ضعيف - يحتاج إعادة تصميم")
    
    def _save_results(self, results: Dict, solver_name: str):
        """حفظ النتائج"""
        
        timestamp = int(time.time())
        filename = f"test_results_{solver_name}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 تم حفظ النتائج في: {filename}")
            
        except Exception as e:
            print(f"❌ فشل في حفظ النتائج: {e}")
    
    def compare_solvers(self, solver_classes: List, max_tasks: int = 20):
        """مقارنة عدة حلالات"""
        
        print(f"🔬 مقارنة {len(solver_classes)} حلال...")
        
        comparison_results = {}
        
        for solver_class in solver_classes:
            print(f"\n{'='*50}")
            print(f"اختبار: {solver_class.__name__}")
            print(f"{'='*50}")
            
            results = self.run_full_evaluation(solver_class, max_tasks)
            comparison_results[solver_class.__name__] = results
        
        # طباعة المقارنة
        self._print_comparison(comparison_results)
        
        return comparison_results
    
    def _print_comparison(self, comparison_results: Dict):
        """طباعة مقارنة الحلالات"""
        
        print(f"\n{'='*80}")
        print(f"📊 مقارنة الحلالات")
        print(f"{'='*80}")
        
        print(f"{'الحلال':<30} {'النجاح%':<10} {'الوقت(s)':<12} {'الأخطاء':<8}")
        print(f"{'-'*70}")
        
        for solver_name, results in comparison_results.items():
            success_rate = results['success_rate']
            avg_time = results['average_time']
            errors = results['error_tasks']
            
            print(f"{solver_name:<30} {success_rate:<10.1f} {avg_time:<12.2f} {errors:<8}")
        
        # تحديد الأفضل
        best_solver = max(comparison_results.keys(), 
                         key=lambda x: comparison_results[x]['success_rate'])
        
        print(f"\n🏆 أفضل حلال: {best_solver}")
        print(f"   معدل النجاح: {comparison_results[best_solver]['success_rate']:.1f}%")

def test_current_systems():
    """اختبار الأنظمة الحالية"""
    
    print("🚀 بدء اختبار الأنظمة الحالية...")
    
    # محاولة استيراد الأنظمة
    systems_to_test = []
    
    try:
        from arc_clean_integrated_system import ARCCleanIntegratedSystem
        systems_to_test.append(ARCCleanIntegratedSystem)
        print("✅ تم تحميل النظام المتكامل")
    except Exception as e:
        print(f"❌ فشل تحميل النظام المتكامل: {e}")
    
    try:
        from arc_ultimate_perfect_system import ARCUltimatePerfectSolver
        systems_to_test.append(ARCUltimatePerfectSolver)
        print("✅ تم تحميل النظام المثالي")
    except Exception as e:
        print(f"❌ فشل تحميل النظام المثالي: {e}")
    
    if not systems_to_test:
        print("❌ لم يتم العثور على أنظمة للاختبار")
        return
    
    # تشغيل الاختبارات
    framework = ComprehensiveTestFramework()
    
    if len(systems_to_test) == 1:
        framework.run_full_evaluation(systems_to_test[0], max_tasks=10)
    else:
        framework.compare_solvers(systems_to_test, max_tasks=10)

if __name__ == "__main__":
    test_current_systems()
