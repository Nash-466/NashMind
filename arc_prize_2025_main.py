from __future__ import annotations
# -*- coding: utf-8 -*-
"""
ARC PRIZE 2025 - MAIN INTEGRATION SYSTEM
========================================
النظام الرئيسي المتكامل لمسابقة ARC Prize 2025
يجمع بين الأنظمة الثلاثة مع واجهة موحدة

الأنظمة المدمجة:
1. MasterOrchestrator - نظام النظريات المتعددة
2. UltimateOrchestrator - نظام الاستدلال المعرفي المتقدم
3. UltimateSystem - نظام الوعي الذاتي والاستدلال السببي
4. ARCInteractiveSystem - النظام التفاعلي المتكامل

المؤلف: مساعد AI
التاريخ: 2025
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Dict, Any, List, Optional
import numpy as np

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# استيراد الأنظمة
try:
    from arc_interactive_system import ARCInteractiveSystem
    from arc_ultimate_mind_part7 import MasterOrchestrator
    from arc_ultimate_system import UltimateOrchestrator
    from arc_revolutionary_system import UltimateSystem
    from kaggle_io import load_arc_tasks_from_dir, load_arc_solutions_from_dir
    SYSTEMS_AVAILABLE = True
    logger.info("✅ جميع الأنظمة متاحة")
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    logger.error(f"❌ خطأ في استيراد الأنظمة: {e}")
    sys.exit(1)

class ARCPrize2025System:
    """النظام الرئيسي لمسابقة ARC Prize 2025"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """تهيئة النظام الرئيسي"""
        self.config = config or {}
        
        # تهيئة النظام التفاعلي المتكامل
        self.interactive_system = ARCInteractiveSystem(self.config)
        
        # إعدادات المسابقة
        self.competition_config = {
            'max_time_per_task': 30.0,
            'max_memory_mb': 8192,
            'output_format': 'submission',
            'enable_validation': True,
            'enable_learning': True
        }
        
        # إحصائيات المسابقة
        self.competition_stats = {
            'total_tasks': 0,
            'solved_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
            'average_consensus': 0.0,
            'system_performance': {}
        }
        
        logger.info("🏆 نظام ARC Prize 2025 جاهز للمسابقة!")
    
    def process_training_data(self, data_path: str) -> Dict[str, Any]:
        """معالجة بيانات التدريب"""
        logger.info(f"📚 معالجة بيانات التدريب من: {data_path}")
        
        try:
            # تحميل بيانات التدريب
            training_tasks = load_arc_tasks_from_dir(data_path, 'train')
            training_solutions = load_arc_solutions_from_dir(data_path, 'train')
            
            logger.info(f"✅ تم تحميل {len(training_tasks)} مهمة تدريب")
            
            # معالجة المهام التدريبية
            training_results = self._process_task_batch(training_tasks, training_solutions, 'training')
            
            # تحديث الإحصائيات
            self.competition_stats.update(training_results['stats'])
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة بيانات التدريب: {e}")
            return {'error': str(e), 'stats': self.competition_stats}
    
    def process_evaluation_data(self, data_path: str) -> Dict[str, Any]:
        """معالجة بيانات التقييم"""
        logger.info(f"📊 معالجة بيانات التقييم من: {data_path}")
        
        try:
            # تحميل بيانات التقييم
            eval_tasks = load_arc_tasks_from_dir(data_path, 'eval')
            eval_solutions = load_arc_solutions_from_dir(data_path, 'eval')
            
            logger.info(f"✅ تم تحميل {len(eval_tasks)} مهمة تقييم")
            
            # معالجة مهام التقييم
            eval_results = self._process_task_batch(eval_tasks, eval_solutions, 'evaluation')
            
            # تحديث الإحصائيات
            self.competition_stats.update(eval_results['stats'])
            
            return eval_results
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة بيانات التقييم: {e}")
            return {'error': str(e), 'stats': self.competition_stats}
    
    def process_test_data(self, data_path: str, output_path: str = 'submission.json') -> Dict[str, Any]:
        """معالجة بيانات الاختبار وإنشاء ملف التقديم"""
        logger.info(f"🧪 معالجة بيانات الاختبار من: {data_path}")
        
        try:
            # تحميل بيانات الاختبار
            test_tasks = load_arc_tasks_from_dir(data_path, 'test')
            
            logger.info(f"✅ تم تحميل {len(test_tasks)} مهمة اختبار")
            
            # معالجة مهام الاختبار
            test_results = self._process_task_batch(test_tasks, None, 'test')
            
            # إنشاء ملف التقديم
            submission_data = self._create_submission_file(test_results['solutions'], output_path)
            
            logger.info(f"📝 تم إنشاء ملف التقديم: {output_path}")
            
            return {
                'submission_file': output_path,
                'stats': test_results['stats'],
                'solutions_count': len(test_results['solutions'])
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة بيانات الاختبار: {e}")
            return {'error': str(e)}
    
    def _process_task_batch(self, tasks: Dict[str, Any], solutions: Dict[str, Any] = None, 
                           mode: str = 'test') -> Dict[str, Any]:
        """معالجة مجموعة من المهام"""
        logger.info(f"🔄 بدء معالجة {len(tasks)} مهمة في وضع: {mode}")
        
        batch_start_time = time.time()
        processed_solutions = {}
        batch_stats = {
            'total_tasks': len(tasks),
            'solved_tasks': 0,
            'failed_tasks': 0,
            'total_time': 0.0,
            'average_consensus': 0.0,
            'system_performance': {}
        }
        
        consensus_scores = []
        
        for task_id, task_data in tasks.items():
            logger.info(f"🎯 معالجة المهمة: {task_id}")
            task_start_time = time.time()
            
            try:
                # معالجة المهمة باستخدام النظام التفاعلي
                result = self.interactive_system.process_task_interactive(task_data, task_id)
                
                task_time = time.time() - task_start_time
                
                # حفظ الحل
                if result.final_solution is not None:
                    processed_solutions[task_id] = result.final_solution.tolist()
                    batch_stats['solved_tasks'] += 1
                else:
                    # حل افتراضي في حالة الفشل
                    processed_solutions[task_id] = [[0]]
                    batch_stats['failed_tasks'] += 1
                
                # تحديث الإحصائيات
                batch_stats['total_time'] += task_time
                consensus_scores.append(result.consensus_score)
                
                # التحقق من صحة الحل إذا كانت الحلول متاحة
                if solutions and task_id in solutions:
                    validation_result = self._validate_solution(
                        result.final_solution, 
                        solutions[task_id], 
                        task_id
                    )
                    logger.info(f"✅ التحقق من المهمة {task_id}: {validation_result}")
                
                logger.info(f"✅ اكتملت المهمة {task_id} في {task_time:.2f}s - الإجماع: {result.consensus_score:.3f}")
                
            except Exception as e:
                logger.error(f"❌ خطأ في معالجة المهمة {task_id}: {e}")
                processed_solutions[task_id] = [[0]]  # حل افتراضي
                batch_stats['failed_tasks'] += 1
        
        # حساب الإحصائيات النهائية
        batch_stats['average_consensus'] = np.mean(consensus_scores) if consensus_scores else 0.0
        batch_stats['success_rate'] = batch_stats['solved_tasks'] / batch_stats['total_tasks']
        batch_stats['average_time_per_task'] = batch_stats['total_time'] / batch_stats['total_tasks']
        
        # الحصول على ملخص أداء الأنظمة
        batch_stats['system_performance'] = self.interactive_system.get_system_performance_summary()
        
        total_batch_time = time.time() - batch_start_time
        logger.info(f"🏁 انتهت معالجة الدفعة في {total_batch_time:.2f}s")
        logger.info(f"📊 النتائج: {batch_stats['solved_tasks']}/{batch_stats['total_tasks']} مهام محلولة")
        logger.info(f"📈 معدل النجاح: {batch_stats['success_rate']:.1%}")
        logger.info(f"🤝 متوسط الإجماع: {batch_stats['average_consensus']:.3f}")
        
        return {
            'solutions': processed_solutions,
            'stats': batch_stats
        }
    
    def _validate_solution(self, predicted_solution: np.ndarray, expected_solution: Any, task_id: str) -> str:
        """التحقق من صحة الحل"""
        try:
            if predicted_solution is None:
                return "فشل - لا يوجد حل"
            
            # تحويل الحل المتوقع إلى numpy array إذا لزم الأمر
            if isinstance(expected_solution, list):
                expected_array = np.array(expected_solution)
            else:
                expected_array = expected_solution
            
            # مقارنة الحلول
            if np.array_equal(predicted_solution, expected_array):
                return "صحيح ✅"
            else:
                # حساب نسبة التطابق
                matches = np.sum(predicted_solution == expected_array)
                total = predicted_solution.size
                accuracy = matches / total if total > 0 else 0.0
                return f"خطأ - دقة: {accuracy:.1%}"
                
        except Exception as e:
            return f"خطأ في التحقق: {e}"
    
    def _create_submission_file(self, solutions: Dict[str, Any], output_path: str) -> str:
        """إنشاء ملف التقديم بالتنسيق المطلوب"""
        logger.info(f"📝 إنشاء ملف التقديم: {output_path}")
        
        # تنسيق التقديم المطلوب
        submission_data = {}
        
        for task_id, solution in solutions.items():
            submission_data[task_id] = [{'attempt_1': solution}]
        
        # حفظ الملف
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2)
        
        logger.info(f"✅ تم حفظ ملف التقديم: {output_path}")
        return output_path
    
    def get_competition_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص المسابقة"""
        return {
            'competition_stats': self.competition_stats,
            'system_performance': self.interactive_system.get_system_performance_summary(),
            'interaction_config': self.interactive_system.interaction_config,
            'timestamp': time.time()
        }
    
    def optimize_system(self):
        """تحسين النظام بناءً على الأداء السابق"""
        logger.info("⚡ تحسين النظام")
        
        # تحسين إعدادات التفاعل
        self.interactive_system.optimize_interaction_config()
        
        logger.info("✅ تم تحسين النظام")


def parse_arguments():
    """تحليل وسيطات سطر الأوامر"""
    parser = argparse.ArgumentParser(
        description='ARC Prize 2025 - النظام المتكامل للأنظمة الثلاثة',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  python arc_prize_2025_main.py --train-data ./data --mode training
  python arc_prize_2025_main.py --eval-data ./data --mode evaluation  
  python arc_prize_2025_main.py --test-data ./data --mode test --output submission.json
  python arc_prize_2025_main.py --single-task ./data/task_001.json --mode single
        """
    )
    
    parser.add_argument('--data-path', '-d', required=True,
                       help='مسار مجلد البيانات')
    parser.add_argument('--mode', '-m', required=True,
                       choices=['training', 'evaluation', 'test', 'single'],
                       help='وضع التشغيل')
    parser.add_argument('--output', '-o', default='submission.json',
                       help='مسار ملف الإخراج (للاختبار)')
    parser.add_argument('--single-task', '-s',
                       help='مسار مهمة واحدة للاختبار')
    parser.add_argument('--config', '-c',
                       help='مسار ملف الإعدادات JSON')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='مستوى التسجيل')
    
    return parser.parse_args()


def main():
    """الدالة الرئيسية"""
    args = parse_arguments()
    
    # تحديث مستوى التسجيل
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("🚀 بدء تشغيل نظام ARC Prize 2025")
    
    # تحميل الإعدادات
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"✅ تم تحميل الإعدادات من: {args.config}")
    
    # إنشاء النظام الرئيسي
    arc_system = ARCPrize2025System(config)
    
    try:
        if args.mode == 'training':
            # وضع التدريب
            logger.info("📚 بدء وضع التدريب")
            results = arc_system.process_training_data(args.data_path)
            
        elif args.mode == 'evaluation':
            # وضع التقييم
            logger.info("📊 بدء وضع التقييم")
            results = arc_system.process_evaluation_data(args.data_path)
            
        elif args.mode == 'test':
            # وضع الاختبار
            logger.info("🧪 بدء وضع الاختبار")
            results = arc_system.process_test_data(args.data_path, args.output)
            
        elif args.mode == 'single':
            # مهمة واحدة
            logger.info("🎯 معالجة مهمة واحدة")
            if args.single_task:
                with open(args.single_task, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                result = arc_system.interactive_system.process_task_interactive(task_data)
                logger.info(f"✅ النتيجة: {'نجح' if result.final_solution is not None else 'فشل'}")
            else:
                logger.error("❌ يجب تحديد مسار المهمة باستخدام --single-task")
                return
        
        # عرض ملخص النتائج
        summary = arc_system.get_competition_summary()
        logger.info("📈 ملخص المسابقة:")
        logger.info(f"   • إجمالي المهام: {summary['competition_stats']['total_tasks']}")
        logger.info(f"   • المهام المحلولة: {summary['competition_stats']['solved_tasks']}")
        logger.info(f"   • معدل النجاح: {summary['competition_stats'].get('success_rate', 0):.1%}")
        logger.info(f"   • متوسط الإجماع: {summary['competition_stats']['average_consensus']:.3f}")
        
        # تحسين النظام
        arc_system.optimize_system()
        
        logger.info("🏆 انتهى تشغيل نظام ARC Prize 2025 بنجاح!")
        
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

