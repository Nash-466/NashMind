from __future__ import annotations
# -*- coding: utf-8 -*-
"""
ARC Simple Test - اختبار بسيط للنظام المتكامل
============================================
نسخة مبسطة تعمل بدون المكتبات الخارجية للاختبار السريع
"""

import json
import time
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Optional
import numpy as np

# إعداد نظام التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleARCAgent:
    """وكيل ARC بسيط للاختبار"""
    
    def __init__(self, name: str):
        self.name = name
        self.strategies = {
            'identity': lambda g: g,
            'flip_horizontal': lambda g: np.fliplr(g),
            'flip_vertical': lambda g: np.flipud(g),
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'transpose': lambda g: g.T,
        }
    
    def solve_task(self, task: Dict[str, Any]) -> Optional[np.ndarray]:
        """حل المهمة باستخدام استراتيجيات بسيطة"""
        try:
            train_pairs = task['train']
            test_input = np.array(task['test'][0]['input'])
            
            # تجربة كل استراتيجية
            for strategy_name, strategy_func in self.strategies.items():
                if self._test_strategy(train_pairs, strategy_func):
                    # تطبيق الاستراتيجية على المدخل
                    result = strategy_func(test_input)
                    logger.info(f"✅ {self.name}: نجح باستخدام {strategy_name}")
                    return result
            
            logger.warning(f"❌ {self.name}: لم يجد حل مناسب")
            return None
            
        except Exception as e:
            logger.error(f"❌ {self.name}: خطأ - {e}")
            return None
    
    def _test_strategy(self, train_pairs: List[Dict], strategy_func) -> bool:
        """اختبار استراتيجية على بيانات التدريب"""
        try:
            for pair in train_pairs:
                input_grid = np.array(pair['input'])
                expected_output = np.array(pair['output'])
                
                # تطبيق الاستراتيجية
                actual_output = strategy_func(input_grid)
                
                # مقارنة النتائج
                if not np.array_equal(actual_output, expected_output):
                    return False
            
            return True
            
        except Exception:
            return False

class SimpleInteractiveSystem:
    """نظام تفاعلي بسيط يجمع بين عدة وكلاء"""
    
    def __init__(self):
        self.agents = {
            'agent_1': SimpleARCAgent('النظام الأول'),
            'agent_2': SimpleARCAgent('النظام الثاني'),
            'agent_3': SimpleARCAgent('النظام الثالث')
        }
        logger.info("🚀 النظام التفاعلي البسيط جاهز!")
    
    def process_task(self, task: Dict[str, Any], task_id: str = None) -> Dict[str, Any]:
        """معالجة المهمة باستخدام الأنظمة الثلاثة"""
        if task_id is None:
            task_id = f"task_{int(time.time())}"
        
        logger.info(f"🎯 بدء معالجة المهمة: {task_id}")
        start_time = time.time()
        
        # تشغيل الأنظمة الثلاثة
        results = {}
        for agent_name, agent in self.agents.items():
            logger.info(f"🔄 تشغيل {agent_name}...")
            solution = agent.solve_task(task)
            results[agent_name] = {
                'solution': solution,
                'success': solution is not None,
                'agent_name': agent.name
            }
        
        # تحليل النتائج
        successful_results = [r for r in results.values() if r['success']]
        
        if successful_results:
            # اختيار أول حل ناجح
            final_solution = successful_results[0]['solution']
            consensus_score = len(successful_results) / len(self.agents)
            logger.info(f"✅ تم العثور على حل من {len(successful_results)} نظام")
        else:
            final_solution = None
            consensus_score = 0.0
            logger.warning("❌ لم يجد أي نظام حلاً")
        
        processing_time = time.time() - start_time
        
        return {
            'task_id': task_id,
            'final_solution': final_solution,
            'consensus_score': consensus_score,
            'processing_time': processing_time,
            'system_results': results,
            'success': final_solution is not None
        }

def test_with_sample_data():
    """اختبار النظام مع بيانات عينة"""
    logger.info("🧪 بدء الاختبار مع بيانات العينة")
    
    # إنشاء النظام
    system = SimpleInteractiveSystem()
    
    # مهمة عينة بسيطة
    sample_task = {
        "id": "test_001",
        "train": [
            {
                "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                "output": [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
            }
        ],
        "test": [
            {
                "input": [[0, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]
            }
        ]
    }
    
    # معالجة المهمة
    result = system.process_task(sample_task)
    
    # عرض النتائج
    logger.info("📊 نتائج الاختبار:")
    logger.info(f"   • معرف المهمة: {result['task_id']}")
    logger.info(f"   • نجح: {'نعم' if result['success'] else 'لا'}")
    logger.info(f"   • درجة الإجماع: {result['consensus_score']:.3f}")
    logger.info(f"   • وقت المعالجة: {result['processing_time']:.2f}s")
    
    if result['final_solution'] is not None:
        logger.info(f"   • الحل النهائي:")
        for row in result['final_solution']:
            logger.info(f"     {row}")
    
    # عرض نتائج الأنظمة الفردية
    logger.info("🔍 نتائج الأنظمة الفردية:")
    for agent_name, agent_result in result['system_results'].items():
        logger.info(f"   • {agent_result['agent_name']}: {'نجح' if agent_result['success'] else 'فشل'}")
    
    return result

def test_with_real_data():
    """اختبار النظام مع البيانات الفعلية"""
    logger.info("📚 اختبار النظام مع البيانات الفعلية")
    
    try:
        # محاولة تحميل بيانات التدريب
        with open('arc-agi_training_challenges.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        logger.info(f"✅ تم تحميل {len(training_data)} مهمة تدريب")
        
        # إنشاء النظام
        system = SimpleInteractiveSystem()
        
        # اختبار على أول 3 مهام
        test_results = []
        for i, (task_id, task_data) in enumerate(list(training_data.items())[:3]):
            logger.info(f"🎯 اختبار المهمة {i+1}/3: {task_id}")
            
            result = system.process_task(task_data, task_id)
            test_results.append(result)
            
            logger.info(f"   النتيجة: {'نجح' if result['success'] else 'فشل'} - الإجماع: {result['consensus_score']:.3f}")
        
        # ملخص النتائج
        successful_tasks = sum(1 for r in test_results if r['success'])
        avg_consensus = sum(r['consensus_score'] for r in test_results) / len(test_results)
        avg_time = sum(r['processing_time'] for r in test_results) / len(test_results)
        
        logger.info("📈 ملخص الاختبار:")
        logger.info(f"   • المهام الناجحة: {successful_tasks}/{len(test_results)}")
        logger.info(f"   • متوسط الإجماع: {avg_consensus:.3f}")
        logger.info(f"   • متوسط وقت المعالجة: {avg_time:.2f}s")
        
        return test_results
        
    except FileNotFoundError:
        logger.error("❌ ملف البيانات غير موجود: arc-agi_training_challenges.json")
        return None
    except Exception as e:
        logger.error(f"❌ خطأ في اختبار البيانات الفعلية: {e}")
        return None

def main():
    """الدالة الرئيسية للاختبار"""
    logger.info("🚀 بدء اختبار النظام البسيط")
    
    # اختبار مع بيانات العينة
    sample_result = test_with_sample_data()
    
    print("\n" + "="*50)
    
    # اختبار مع البيانات الفعلية
    real_data_result = test_with_real_data()
    
    logger.info("🏁 انتهى الاختبار")

if __name__ == "__main__":
    main()

