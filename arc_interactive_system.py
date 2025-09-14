from __future__ import annotations
# -*- coding: utf-8 -*-
"""
ARC INTERACTIVE INTEGRATED SYSTEM - ARC Prize 2025
==================================================
نظام تفاعلي متكامل يجمع بين الأنظمة الثلاثة لحل مشاكل ARC

الأنظمة الثلاثة:
1. MasterOrchestrator (arc_ultimate_mind_part7.py) - نظام النظريات المتعددة
2. UltimateOrchestrator (arc_ultimate_system.py) - نظام الاستدلال المعرفي المتقدم  
3. UltimateSystem (arc_revolutionary_system.py) - نظام الوعي الذاتي والاستدلال السببي

المؤلف: مساعد AI
التاريخ: 2025
"""

import numpy as np
import json
import time
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# استيراد الأنظمة الثلاثة
try:
    from arc_ultimate_mind_part7 import MasterOrchestrator
    from arc_ultimate_system import UltimateOrchestrator
    from arc_revolutionary_system import UltimateSystem
    SYSTEMS_AVAILABLE = True
    logger.info("✅ جميع الأنظمة الثلاثة متاحة")
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    logger.error(f"❌ خطأ في استيراد الأنظمة: {e}")
    # إنشاء فئات وهمية للاختبار
    class MasterOrchestrator:
        def process_single_task(self, task): return None
    class UltimateOrchestrator:
        def process_arc_task(self, task, task_id=None): return {}
    class UltimateSystem:
        def process_task(self, task): return {}

@dataclass
class SystemResult:
    """نتيجة من نظام واحد"""
    system_name: str
    solution: Optional[np.ndarray]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class InteractiveResult:
    """النتيجة النهائية من النظام التفاعلي"""
    final_solution: Optional[np.ndarray]
    system_results: List[SystemResult]
    consensus_score: float
    total_processing_time: float
    interaction_summary: Dict[str, Any]

class ARCInteractiveSystem:
    """النظام التفاعلي المتكامل للأنظمة الثلاثة"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """تهيئة النظام التفاعلي"""
        self.config = config or {}
        
        # تهيئة الأنظمة الثلاثة
        self.systems = {
            'theory_based': MasterOrchestrator(),
            'cognitive_reasoning': UltimateOrchestrator(),
            'causal_awareness': UltimateSystem()
        }
        
        # إعدادات التفاعل
        self.interaction_config = {
            'max_parallel_time': 30.0,  # أقصى وقت للمعالجة المتوازية
            'consensus_threshold': 0.7,  # عتبة الإجماع
            'confidence_weight': 0.4,   # وزن الثقة في التقييم
            'time_weight': 0.2,         # وزن الوقت في التقييم
            'quality_weight': 0.4,      # وزن الجودة في التقييم
            'enable_cross_validation': True,  # تفعيل التحقق المتبادل
            'enable_learning': True,    # تفعيل التعلم من التفاعل
        }
        
        # ذاكرة التفاعل
        self.interaction_memory = defaultdict(list)
        self.performance_history = defaultdict(list)
        
        logger.info("🚀 النظام التفاعلي المتكامل جاهز للعمل!")
    
    def process_task_interactive(self, task: Dict[str, Any], task_id: str = None) -> InteractiveResult:
        """معالجة المهمة بشكل تفاعلي باستخدام الأنظمة الثلاثة"""
        if task_id is None:
            task_id = f"task_{int(time.time())}"
        
        logger.info(f"🎯 بدء المعالجة التفاعلية للمهمة: {task_id}")
        start_time = time.time()
        
        # المرحلة 1: المعالجة المتوازية للأنظمة الثلاثة
        system_results = self._parallel_processing(task, task_id)
        
        # المرحلة 2: تحليل النتائج والتفاعل بين الأنظمة
        interaction_analysis = self._analyze_interactions(system_results, task_id)
        
        # المرحلة 3: توليد الحل النهائي بالإجماع
        final_solution, consensus_score = self._generate_consensus_solution(
            system_results, interaction_analysis, task_id
        )
        
        # المرحلة 4: التعلم من التفاعل
        if self.interaction_config['enable_learning']:
            self._learn_from_interaction(system_results, interaction_analysis, task_id)
        
        total_time = time.time() - start_time
        
        result = InteractiveResult(
            final_solution=final_solution,
            system_results=system_results,
            consensus_score=consensus_score,
            total_processing_time=total_time,
            interaction_summary=interaction_analysis
        )
        
        logger.info(f"✅ انتهت المعالجة التفاعلية في {total_time:.2f}s - درجة الإجماع: {consensus_score:.3f}")
        return result
    
    def _parallel_processing(self, task: Dict[str, Any], task_id: str) -> List[SystemResult]:
        """معالجة متوازية للأنظمة الثلاثة"""
        logger.info("🔄 بدء المعالجة المتوازية للأنظمة الثلاثة")
        
        system_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # إرسال المهام للأنظمة الثلاثة
            future_to_system = {}
            
            for system_name, system in self.systems.items():
                future = executor.submit(self._run_single_system, system_name, system, task, task_id)
                future_to_system[future] = system_name
            
            # جمع النتائج
            for future in as_completed(future_to_system, timeout=self.interaction_config['max_parallel_time']):
                system_name = future_to_system[future]
                try:
                    result = future.result()
                    system_results.append(result)
                    logger.info(f"✅ {system_name}: اكتمل في {result.processing_time:.2f}s")
                except Exception as e:
                    logger.error(f"❌ {system_name}: خطأ - {e}")
                    # إضافة نتيجة فارغة في حالة الخطأ
                    system_results.append(SystemResult(
                        system_name=system_name,
                        solution=None,
                        confidence=0.0,
                        processing_time=0.0,
                        metadata={'error': str(e)}
                    ))
        
        return system_results
    
    def _run_single_system(self, system_name: str, system: Any, task: Dict[str, Any], task_id: str) -> SystemResult:
        """تشغيل نظام واحد"""
        start_time = time.time()
        
        try:
            if system_name == 'theory_based':
                solution = system.process_single_task(task)
            elif system_name == 'cognitive_reasoning':
                result = system.process_arc_task(task, task_id)
                solution = result.get('solution')
            elif system_name == 'causal_awareness':
                result = system.process_task(task)
                solution = result.get('solution')
            else:
                solution = None
            
            processing_time = time.time() - start_time
            
            # حساب الثقة بناءً على وجود الحل وجودته
            confidence = self._calculate_confidence(solution, processing_time)
            
            return SystemResult(
                system_name=system_name,
                solution=solution,
                confidence=confidence,
                processing_time=processing_time,
                metadata={'task_id': task_id, 'timestamp': time.time()}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"خطأ في النظام {system_name}: {e}")
            return SystemResult(
                system_name=system_name,
                solution=None,
                confidence=0.0,
                processing_time=processing_time,
                metadata={'error': str(e), 'task_id': task_id}
            )
    
    def _analyze_interactions(self, system_results: List[SystemResult], task_id: str) -> Dict[str, Any]:
        """تحليل التفاعل بين الأنظمة"""
        logger.info("🔍 تحليل التفاعل بين الأنظمة")
        
        analysis = {
            'successful_systems': [r for r in system_results if r.solution is not None],
            'failed_systems': [r for r in system_results if r.solution is None],
            'solution_similarity': self._calculate_solution_similarity(system_results),
            'confidence_distribution': [r.confidence for r in system_results],
            'time_distribution': [r.processing_time for r in system_results],
            'consensus_indicators': self._identify_consensus_indicators(system_results)
        }
        
        # حساب درجة التفاعل الإجمالية
        analysis['interaction_score'] = self._calculate_interaction_score(analysis)
        
        return analysis
    
    def _calculate_solution_similarity(self, system_results: List[SystemResult]) -> Dict[str, float]:
        """حساب التشابه بين الحلول"""
        solutions = [r.solution for r in system_results if r.solution is not None]
        
        if len(solutions) < 2:
            return {'average_similarity': 0.0, 'max_similarity': 0.0}
        
        similarities = []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                sim = self._grid_similarity(solutions[i], solutions[j])
                similarities.append(sim)
        
        return {
            'average_similarity': np.mean(similarities) if similarities else 0.0,
            'max_similarity': np.max(similarities) if similarities else 0.0,
            'min_similarity': np.min(similarities) if similarities else 0.0
        }
    
    def _grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """حساب التشابه بين شبكتين"""
        if grid1 is None or grid2 is None:
            return 0.0
        
        try:
            # التأكد من نفس الشكل
            if grid1.shape != grid2.shape:
                return 0.0
            
            # حساب نسبة التطابق
            matches = np.sum(grid1 == grid2)
            total = grid1.size
            return matches / total if total > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _identify_consensus_indicators(self, system_results: List[SystemResult]) -> Dict[str, Any]:
        """تحديد مؤشرات الإجماع"""
        successful_results = [r for r in system_results if r.solution is not None]
        
        if len(successful_results) == 0:
            return {'consensus_level': 'none', 'agreement_count': 0}
        
        # حساب متوسط الثقة
        avg_confidence = np.mean([r.confidence for r in successful_results])
        
        # حساب عدد الأنظمة التي توافق
        high_confidence_count = sum(1 for r in successful_results if r.confidence > 0.7)
        
        if high_confidence_count >= 2:
            consensus_level = 'strong'
        elif high_confidence_count == 1:
            consensus_level = 'weak'
        else:
            consensus_level = 'none'
        
        return {
            'consensus_level': consensus_level,
            'agreement_count': high_confidence_count,
            'average_confidence': avg_confidence
        }
    
    def _calculate_interaction_score(self, analysis: Dict[str, Any]) -> float:
        """حساب درجة التفاعل الإجمالية"""
        # وزن النجاح
        success_weight = len(analysis['successful_systems']) / 3.0
        
        # وزن التشابه
        similarity_weight = analysis['solution_similarity']['average_similarity']
        
        # وزن الثقة
        confidence_weight = np.mean(analysis['confidence_distribution'])
        
        # وزن الوقت (كلما كان أسرع كان أفضل)
        time_weight = 1.0 - min(np.mean(analysis['time_distribution']) / 30.0, 1.0)
        
        # حساب الدرجة النهائية
        interaction_score = (
            0.4 * success_weight +
            0.3 * similarity_weight +
            0.2 * confidence_weight +
            0.1 * time_weight
        )
        
        return min(interaction_score, 1.0)
    
    def _generate_consensus_solution(self, system_results: List[SystemResult], 
                                   interaction_analysis: Dict[str, Any], task_id: str) -> Tuple[Optional[np.ndarray], float]:
        """توليد الحل بالإجماع"""
        logger.info("🤝 توليد الحل بالإجماع")
        
        successful_results = [r for r in system_results if r.solution is not None]
        
        if not successful_results:
            logger.warning("❌ لا توجد حلول ناجحة من أي نظام")
            return None, 0.0
        
        if len(successful_results) == 1:
            logger.info("✅ حل واحد متاح من نظام واحد")
            return successful_results[0].solution, successful_results[0].confidence
        
        # اختيار الحل الأفضل بناءً على الثقة والجودة
        best_solution = None
        best_score = -1.0
        
        for result in successful_results:
            # حساب النقاط المركبة
            score = (
                self.interaction_config['confidence_weight'] * result.confidence +
                self.interaction_config['quality_weight'] * self._evaluate_solution_quality(result.solution) +
                self.interaction_config['time_weight'] * (1.0 - min(result.processing_time / 30.0, 1.0))
            )
            
            if score > best_score:
                best_score = score
                best_solution = result.solution
        
        consensus_score = best_score
        logger.info(f"🎯 تم اختيار الحل الأفضل بدرجة إجماع: {consensus_score:.3f}")
        
        return best_solution, consensus_score
    
    def _calculate_confidence(self, solution: Optional[np.ndarray], processing_time: float) -> float:
        """حساب الثقة في الحل"""
        if solution is None:
            return 0.0
        
        # الثقة الأساسية بناءً على وجود الحل
        base_confidence = 0.5
        
        # تعديل بناءً على وقت المعالجة (أسرع = أفضل)
        time_factor = max(0.0, 1.0 - processing_time / 30.0)
        
        # تعديل بناءً على جودة الحل
        quality_factor = self._evaluate_solution_quality(solution)
        
        # حساب الثقة النهائية
        confidence = base_confidence + 0.3 * time_factor + 0.2 * quality_factor
        
        return min(confidence, 1.0)
    
    def _evaluate_solution_quality(self, solution: np.ndarray) -> float:
        """تقييم جودة الحل"""
        if solution is None:
            return 0.0
        
        try:
            # تقييمات بسيطة للجودة
            quality_score = 0.0
            
            # حجم الحل (ليس صغير جداً أو كبير جداً)
            size_score = 1.0 - abs(solution.size - 25) / 25.0
            quality_score += 0.3 * max(0.0, size_score)
            
            # تنوع الألوان (ليس أحادي اللون)
            unique_colors = len(np.unique(solution))
            color_score = min(unique_colors / 5.0, 1.0)
            quality_score += 0.3 * color_score
            
            # تعقيد الشكل (ليس فارغ أو ممتلئ بالكامل)
            non_zero_ratio = np.sum(solution > 0) / solution.size
            complexity_score = 1.0 - abs(non_zero_ratio - 0.5) * 2
            quality_score += 0.4 * max(0.0, complexity_score)
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.5  # قيمة افتراضية في حالة الخطأ
    
    def _learn_from_interaction(self, system_results: List[SystemResult], 
                               interaction_analysis: Dict[str, Any], task_id: str):
        """التعلم من التفاعل"""
        logger.info("🧠 التعلم من التفاعل")
        
        # حفظ نتائج التفاعل في الذاكرة
        interaction_record = {
            'task_id': task_id,
            'timestamp': time.time(),
            'system_results': [
                {
                    'system_name': r.system_name,
                    'confidence': r.confidence,
                    'processing_time': r.processing_time,
                    'success': r.solution is not None
                }
                for r in system_results
            ],
            'interaction_score': interaction_analysis['interaction_score'],
            'consensus_level': interaction_analysis['consensus_indicators']['consensus_level']
        }
        
        self.interaction_memory[task_id].append(interaction_record)
        
        # تحديث إحصائيات الأداء
        for result in system_results:
            self.performance_history[result.system_name].append({
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'success': result.solution is not None,
                'timestamp': time.time()
            })
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص أداء الأنظمة"""
        summary = {}
        
        for system_name, history in self.performance_history.items():
            if not history:
                continue
            
            recent_history = history[-10:]  # آخر 10 محاولات
            
            summary[system_name] = {
                'total_attempts': len(history),
                'recent_success_rate': sum(1 for h in recent_history if h['success']) / len(recent_history),
                'average_confidence': np.mean([h['confidence'] for h in recent_history]),
                'average_processing_time': np.mean([h['processing_time'] for h in recent_history]),
                'last_updated': max([h['timestamp'] for h in recent_history])
            }
        
        return summary
    
    def optimize_interaction_config(self):
        """تحسين إعدادات التفاعل بناءً على الأداء السابق"""
        logger.info("⚡ تحسين إعدادات التفاعل")
        
        performance_summary = self.get_system_performance_summary()
        
        # تحسين الأوزان بناءً على أداء الأنظمة
        for system_name, perf in performance_summary.items():
            if perf['recent_success_rate'] > 0.8:
                # زيادة وزن هذا النظام
                if system_name == 'theory_based':
                    self.interaction_config['confidence_weight'] += 0.05
                elif system_name == 'cognitive_reasoning':
                    self.interaction_config['quality_weight'] += 0.05
                elif system_name == 'causal_awareness':
                    self.interaction_config['time_weight'] += 0.05
        
        # تطبيع الأوزان
        total_weight = (
            self.interaction_config['confidence_weight'] +
            self.interaction_config['quality_weight'] +
            self.interaction_config['time_weight']
        )
        
        if total_weight > 0:
            self.interaction_config['confidence_weight'] /= total_weight
            self.interaction_config['quality_weight'] /= total_weight
            self.interaction_config['time_weight'] /= total_weight
        
        logger.info("✅ تم تحسين إعدادات التفاعل")


def main():
    """دالة الاختبار الرئيسية"""
    print("🚀 اختبار النظام التفاعلي المتكامل")
    
    # إنشاء النظام التفاعلي
    interactive_system = ARCInteractiveSystem()
    
    # مثال على مهمة ARC بسيطة
    sample_task = {
        "id": "test_task_001",
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
    result = interactive_system.process_task_interactive(sample_task)
    
    # عرض النتائج
    print(f"\n📊 نتائج المعالجة التفاعلية:")
    print(f"   • الحل النهائي: {'موجود' if result.final_solution is not None else 'غير موجود'}")
    print(f"   • درجة الإجماع: {result.consensus_score:.3f}")
    print(f"   • وقت المعالجة الإجمالي: {result.total_processing_time:.2f}s")
    
    print(f"\n🔍 نتائج الأنظمة الفردية:")
    for system_result in result.system_results:
        print(f"   • {system_result.system_name}:")
        print(f"     - الثقة: {system_result.confidence:.3f}")
        print(f"     - وقت المعالجة: {system_result.processing_time:.2f}s")
        print(f"     - الحل: {'موجود' if system_result.solution is not None else 'غير موجود'}")
    
    # عرض ملخص الأداء
    performance_summary = interactive_system.get_system_performance_summary()
    print(f"\n📈 ملخص الأداء:")
    for system_name, perf in performance_summary.items():
        print(f"   • {system_name}:")
        print(f"     - معدل النجاح: {perf['recent_success_rate']:.1%}")
        print(f"     - متوسط الثقة: {perf['average_confidence']:.3f}")
        print(f"     - متوسط وقت المعالجة: {perf['average_processing_time']:.2f}s")


if __name__ == "__main__":
    main()

