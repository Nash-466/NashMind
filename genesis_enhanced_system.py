from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام Genesis المحسّن - دمج لغة Genesis مع الأنظمة الناجحة
"""

import numpy as np
import sys
import os
from pathlib import Path
from collections.abc import Callable
from typing import Dict, List, Optional, Any
import logging

# إضافة مسار ملف Genesis - استخدام النسخة المحلية
# sys.path.insert(0, r'C:\Users\Lenovo\Downloads')

# استيراد Genesis Language
try:
    from genesis_language import (
        Grid, GridObject, ObjectSet,
        extract_objects_from_grid,
        RelationshipAnalyzer,
        ActionLanguage,
        Principle, SymmetryCompletion, PatternContinuation,
        OutlierRemoval, Homogenization,
        IntentClassifier
    )
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("⚠️ Genesis Language غير متاح")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenesisEnhancedSolver:
    """حل متقدم باستخدام Genesis Language"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier() if GENESIS_AVAILABLE else None
        self.principles = {
            'symmetry': SymmetryCompletion(),
            'pattern': PatternContinuation(),
            'outlier': OutlierRemoval(),
            'homogenize': Homogenization()
        } if GENESIS_AVAILABLE else {}
        
    def analyze_task(self, task_data: Dict) -> Dict:
        """تحليل المهمة باستخدام Genesis"""
        if not GENESIS_AVAILABLE:
            return {}
            
        analysis = {
            'objects': [],
            'relationships': [],
            'patterns': [],
            'transformations': []
        }
        
        # تحليل أمثلة التدريب
        for example in task_data.get('train', []):
            input_grid = Grid(np.array(example['input']))
            output_grid = Grid(np.array(example['output']))
            
            # استخراج الكائنات
            input_objects = extract_objects_from_grid(input_grid)
            output_objects = extract_objects_from_grid(output_grid)
            
            analysis['objects'].append({
                'input_count': len(input_objects),
                'output_count': len(output_objects),
                'input_objects': input_objects,
                'output_objects': output_objects
            })
            
            # تحليل العلاقات
            if len(input_objects) > 1:
                for i, obj1 in enumerate(input_objects):
                    for obj2 in input_objects[i+1:]:
                        rel = {
                            'touching': RelationshipAnalyzer.is_touching(obj1, obj2),
                            'aligned': RelationshipAnalyzer.get_alignment(obj1, obj2),
                            'distance': RelationshipAnalyzer.get_distance(obj1, obj2),
                            'same_color': RelationshipAnalyzer.share_property(obj1, obj2, 'color'),
                            'same_size': RelationshipAnalyzer.share_property(obj1, obj2, 'size')
                        }
                        analysis['relationships'].append(rel)
            
            # تحديد المبدأ/التحويل
            principle = self.intent_classifier.classify(input_grid, output_grid)
            if principle:
                analysis['transformations'].append(principle)
        
        return analysis
    
    def solve_with_genesis(self, task_data: Dict) -> Optional[np.ndarray]:
        """حل المهمة باستخدام Genesis Language"""
        if not GENESIS_AVAILABLE:
            return None
            
        try:
            # تحليل المهمة
            analysis = self.analyze_task(task_data)
            
            # الحصول على الإدخال للاختبار
            test_input = np.array(task_data['test'][0]['input'])
            test_grid = Grid(test_input)
            test_objects = extract_objects_from_grid(test_grid)
            
            # تحديد المبدأ الأكثر احتمالاً
            if analysis['transformations']:
                from collections import Counter
                most_common_principle = Counter(analysis['transformations']).most_common(1)[0][0]
                
                # تطبيق المبدأ
                if most_common_principle == Principle.SYMMETRY_COMPLETION:
                    actions = self.principles['symmetry'].get_candidate_actions(test_grid, test_objects)
                elif most_common_principle == Principle.PATTERN_CONTINUATION:
                    actions = self.principles['pattern'].get_candidate_actions(test_grid, test_objects)
                elif most_common_principle == Principle.OUTLIER_REMOVAL:
                    actions = self.principles['outlier'].get_candidate_actions(test_grid, test_objects)
                elif most_common_principle == Principle.HOMOGENIZATION:
                    actions = self.principles['homogenize'].get_candidate_actions(test_grid, test_objects)
                else:
                    actions = []
                
                # تطبيق الإجراءات
                if actions:
                    # تطبيق أول إجراء مقترح
                    action_func, *args = actions[0]
                    result_grid = action_func(*args)
                    return result_grid.data
            
            # إذا لم نتمكن من تحديد مبدأ، نحاول كل المبادئ
            for principle in self.principles.values():
                actions = principle.get_candidate_actions(test_grid, test_objects)
                if actions:
                    action_func, *args = actions[0]
                    result_grid = action_func(*args)
                    
                    # التحقق من معقولية النتيجة
                    if self._is_reasonable_solution(result_grid.data, test_input):
                        return result_grid.data
            
        except Exception as e:
            logger.error(f"خطأ في Genesis solver: {e}")
        
        return None
    
    def _is_reasonable_solution(self, solution: np.ndarray, input_grid: np.ndarray) -> bool:
        """التحقق من معقولية الحل"""
        # الحل يجب أن يكون ليس فارغ تماماً
        if np.all(solution == 0):
            return False
        
        # الحل يجب أن يكون مختلف عن الإدخال
        if np.array_equal(solution, input_grid):
            return False
        
        return True

class HybridGenesisSystem:
    """نظام هجين يجمع Genesis مع الأنظمة الناجحة"""
    
    def __init__(self):
        self.genesis_solver = GenesisEnhancedSolver()
        self.load_successful_systems()
        
    def load_successful_systems(self):
        """تحميل الأنظمة الناجحة"""
        self.systems = []
        
        # الأنظمة الناجحة من الاختبار السابق
        successful_systems = [
            'perfect_arc_system_v2',
            'enhanced_efficient_zero',
            'symbolic_rule_engine',
            'neural_pattern_learner'
        ]
        
        for system_name in successful_systems:
            try:
                module = __import__(system_name)
                if hasattr(module, 'solve_task'):
                    self.systems.append({
                        'name': system_name,
                        'solve': module.solve_task
                    })
                    logger.info(f"✓ تم تحميل {system_name}")
            except Exception as e:
                logger.warning(f"✗ فشل تحميل {system_name}: {e}")
    
    def solve_task(self, task_data: Dict) -> np.ndarray:
        """حل المهمة باستخدام النظام الهجين"""
        
        # 1. جرب Genesis أولاً
        if GENESIS_AVAILABLE:
            logger.info("🧬 محاولة حل باستخدام Genesis Language...")
            genesis_solution = self.genesis_solver.solve_with_genesis(task_data)
            if genesis_solution is not None:
                logger.info("✅ نجح Genesis في إيجاد حل!")
                return genesis_solution
        
        # 2. جرب الأنظمة الناجحة
        for system in self.systems:
            try:
                logger.info(f"🔧 محاولة {system['name']}...")
                solution = system['solve'](task_data)
                if solution is not None:
                    
                    # تحسين الحل باستخدام Genesis إذا أمكن
                    if GENESIS_AVAILABLE:
                        enhanced_solution = self.enhance_with_genesis(solution, task_data)
                        if enhanced_solution is not None:
                            logger.info(f"✨ تم تحسين حل {system['name']} باستخدام Genesis")
                            return enhanced_solution
                    
                    return solution
            except Exception as e:
                logger.warning(f"خطأ في {system['name']}: {e}")
        
        # 3. حل احتياطي
        return self.fallback_solution(task_data)
    
    def enhance_with_genesis(self, solution: np.ndarray, task_data: Dict) -> Optional[np.ndarray]:
        """تحسين الحل باستخدام Genesis"""
        if not GENESIS_AVAILABLE:
            return None
            
        try:
            # تحليل الحل الحالي
            solution_grid = Grid(solution)
            solution_objects = extract_objects_from_grid(solution_grid)
            
            # البحث عن تحسينات ممكنة
            for principle_name, principle in self.genesis_solver.principles.items():
                actions = principle.get_candidate_actions(solution_grid, solution_objects)
                
                for action in actions[:1]:  # جرب أول إجراء فقط
                    action_func, *args = action
                    enhanced_grid = action_func(*args)
                    
                    # التحقق من التحسين
                    if self._is_improvement(enhanced_grid.data, solution, task_data):
                        return enhanced_grid.data
        
        except Exception as e:
            logger.debug(f"لا يمكن تحسين الحل: {e}")
        
        return None
    
    def _is_improvement(self, enhanced: np.ndarray, original: np.ndarray, task_data: Dict) -> bool:
        """التحقق من أن التحسين أفضل من الأصل"""
        # معايير بسيطة للتحسين
        
        # 1. يجب أن يكون مختلف
        if np.array_equal(enhanced, original):
            return False
        
        # 2. يجب أن يحافظ على بعض الخصائص
        if enhanced.shape != original.shape:
            return False
        
        # 3. التحقق من التناسق مع أمثلة التدريب
        if task_data.get('train'):
            # هنا يمكن إضافة فحوصات أكثر تطوراً
            pass
        
        return True
    
    def fallback_solution(self, task_data: Dict) -> np.ndarray:
        """حل احتياطي"""
        test_input = np.array(task_data['test'][0]['input'])
        
        # حلول احتياطية بسيطة
        # 1. إرجاع الإدخال كما هو
        # 2. تطبيق تحويل بسيط
        
        # هنا نعيد الإدخال مع تحويل بسيط
        return np.rot90(test_input)

# دالة الواجهة الرئيسية
def solve_task(task_data: Dict) -> np.ndarray:
    """دالة الحل الرئيسية مع Genesis"""
    global hybrid_system
    
    try:
        hybrid_system
    except NameError:
        hybrid_system = HybridGenesisSystem()
    
    return hybrid_system.solve_task(task_data)

# اختبار النظام
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          Genesis Enhanced System                         ║
    ║    دمج Genesis Language مع الأنظمة الناجحة              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if GENESIS_AVAILABLE:
        print("✅ Genesis Language متاح وجاهز!")
    else:
        print("⚠️ Genesis Language غير متاح - تأكد من وجود الملف")
    
    # اختبار بسيط
    test_task = {
        'train': [
            {
                'input': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'output': [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
            }
        ],
        'test': [
            {
                'input': [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            }
        ]
    }
    
    system = HybridGenesisSystem()
    solution = system.solve_task(test_task)
    print(f"\nالحل: \n{solution}")
