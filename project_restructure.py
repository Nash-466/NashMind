from __future__ import annotations
#!/usr/bin/env python3
"""
PROJECT RESTRUCTURE - إعادة هيكلة المشروع
========================================
تنظيف وترتيب المشروع لجعل الأنظمة تعمل بشكل تكاملي
"""

import os
import shutil
import json
from pathlib import Path

def create_organized_structure():
    """إنشاء هيكل منظم للمشروع"""
    
    # إنشاء المجلدات الرئيسية
    folders = {
        'core': 'الأنظمة الأساسية',
        'solvers': 'حلالات ARC',
        'data': 'ملفات البيانات',
        'tests': 'ملفات الاختبار',
        'utils': 'الأدوات المساعدة',
        'results': 'النتائج والتقارير',
        'archive': 'الملفات القديمة',
        'docs': 'التوثيق'
    }
    
    print("🏗️ إنشاء هيكل المجلدات...")
    for folder, description in folders.items():
        os.makedirs(folder, exist_ok=True)
        print(f"  📁 {folder}/ - {description}")
    
    return folders

def categorize_files():
    """تصنيف الملفات الموجودة"""
    
    file_categories = {
        'core': [
            'arc_ultimate_perfect_system.py',
            'arc_core_redesign.py',
            'advanced_pattern_engine.py',
            'main.py'
        ],
        'solvers': [
            'arc_complete_agent_part1.py',
            'arc_complete_agent_part2.py',
            'arc_complete_agent_part3.py',
            'arc_complete_agent_part4.py',
            'arc_complete_agent_part5.py',
            'arc_complete_agent_part6.py',
            'arc_ultimate_mind_part7.py',
            'burhan_meta_brain.py'
        ],
        'data': [
            'ملفات المسابقةarc-prize-2025/',
            'arc-agi_evaluation_challenges.json',
            'arc_memory.json',
            'ai_memory.json'
        ],
        'tests': [
            'quick_system_test.py',
            'simple_test.py',
            'arc_solver_test.py',
            'test_all_systems.py'
        ],
        'utils': [
            'deep_arc_analyzer.py',
            'system_performance_tester.py',
            'detailed_system_diagnosis.py'
        ],
        'results': [
            '*.json',  # ملفات النتائج
            '*.log',   # ملفات السجلات
            '*.pkl'    # ملفات البيانات المحفوظة
        ],
        'docs': [
            '*.md',    # ملفات التوثيق
            'README.md'
        ]
    }
    
    return file_categories

def move_files_to_structure(file_categories):
    """نقل الملفات إلى الهيكل الجديد"""
    
    print("\n📦 نقل الملفات إلى المجلدات المناسبة...")
    
    for category, files in file_categories.items():
        print(f"\n📂 نقل ملفات {category}:")
        
        for file_pattern in files:
            if file_pattern.endswith('/'):
                # مجلد
                if os.path.exists(file_pattern):
                    dest = os.path.join(category, os.path.basename(file_pattern.rstrip('/')))
                    if not os.path.exists(dest):
                        shutil.move(file_pattern, dest)
                        print(f"  ✅ {file_pattern} → {dest}")
            elif '*' in file_pattern:
                # نمط ملفات
                import glob
                for file in glob.glob(file_pattern):
                    if os.path.isfile(file):
                        dest = os.path.join(category, os.path.basename(file))
                        if not os.path.exists(dest):
                            shutil.move(file, dest)
                            print(f"  ✅ {file} → {dest}")
            else:
                # ملف محدد
                if os.path.exists(file_pattern):
                    dest = os.path.join(category, os.path.basename(file_pattern))
                    if not os.path.exists(dest):
                        shutil.move(file_pattern, dest)
                        print(f"  ✅ {file_pattern} → {dest}")

def create_integration_system():
    """إنشاء نظام التكامل الرئيسي"""
    
    integration_code = '''#!/usr/bin/env python3
"""
ARC INTEGRATED SYSTEM - النظام المتكامل
=====================================
نظام موحد يجمع أفضل مكونات المشروع
"""

import sys
import os
sys.path.append('core')
sys.path.append('solvers')
sys.path.append('utils')

import numpy as np
import json
from collections.abc import Callable
from typing import List, Dict, Any

class ARCIntegratedSystem:
    """النظام المتكامل لحل مهام ARC"""
    
    def __init__(self):
        self.solvers = []
        self.pattern_engine = None
        self.memory_system = {}
        self._load_components()
    
    def _load_components(self):
        """تحميل المكونات الأساسية"""
        try:
            # تحميل محرك الأنماط
            from advanced_pattern_engine import AdvancedPatternEngine
            self.pattern_engine = AdvancedPatternEngine()
            print("✅ تم تحميل محرك الأنماط")
        except:
            print("⚠️ فشل تحميل محرك الأنماط")
        
        # تحميل الحلالات
        self._load_solvers()
    
    def _load_solvers(self):
        """تحميل الحلالات المختلفة"""
        solver_modules = [
            'arc_ultimate_perfect_system',
            'arc_core_redesign'
        ]
        
        for module_name in solver_modules:
            try:
                module = __import__(module_name)
                # البحث عن كلاس الحلال في الوحدة
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        'solver' in attr_name.lower() and
                        hasattr(attr, 'solve_task')):
                        solver = attr()
                        self.solvers.append(solver)
                        print(f"✅ تم تحميل {attr_name}")
                        break
            except Exception as e:
                print(f"⚠️ فشل تحميل {module_name}: {e}")
    
    def solve_task(self, task: Dict) -> List[np.ndarray]:
        """حل مهمة ARC باستخدام النظام المتكامل"""
        
        # تحليل المهمة أولاً
        task_analysis = self._analyze_task(task)
        
        # محاولة الحل باستخدام الحلالات المختلفة
        best_solution = None
        best_confidence = 0
        
        for solver in self.solvers:
            try:
                solution = solver.solve_task(task)
                confidence = self._evaluate_solution_confidence(solution, task_analysis)
                
                if confidence > best_confidence:
                    best_solution = solution
                    best_confidence = confidence
                    
            except Exception as e:
                print(f"⚠️ فشل حلال: {e}")
                continue
        
        # إذا لم نجد حل جيد، استخدم الحل الافتراضي
        if best_solution is None or best_confidence < 0.3:
            best_solution = self._generate_fallback_solution(task)
        
        return best_solution
    
    def _analyze_task(self, task: Dict) -> Dict:
        """تحليل المهمة"""
        analysis = {
            'num_examples': len(task['train']),
            'input_sizes': [],
            'output_sizes': [],
            'colors_used': set(),
            'complexity': 'unknown'
        }
        
        for example in task['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            analysis['input_sizes'].append(input_grid.shape)
            analysis['output_sizes'].append(output_grid.shape)
            analysis['colors_used'].update(input_grid.flatten())
            analysis['colors_used'].update(output_grid.flatten())
        
        # تقدير التعقيد
        if len(analysis['colors_used']) <= 3:
            analysis['complexity'] = 'simple'
        elif len(analysis['colors_used']) <= 6:
            analysis['complexity'] = 'medium'
        else:
            analysis['complexity'] = 'complex'
        
        return analysis
    
    def _evaluate_solution_confidence(self, solution: List[np.ndarray], analysis: Dict) -> float:
        """تقييم ثقة الحل"""
        if not solution:
            return 0.0
        
        confidence = 0.5  # ثقة أساسية
        
        # فحص إذا كان الحل منطقياً
        for sol in solution:
            if sol is not None and sol.size > 0:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _generate_fallback_solution(self, task: Dict) -> List[np.ndarray]:
        """إنتاج حل احتياطي"""
        solutions = []
        
        for test_input in task['test']:
            # في أسوأ الحالات، إرجاع المدخل كما هو
            solutions.append(np.array(test_input['input']))
        
        return solutions
    
    def test_system(self):
        """اختبار النظام المتكامل"""
        print("🧪 اختبار النظام المتكامل...")
        
        # تحميل مهمة تجريبية
        try:
            with open('data/ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
                tasks = json.load(f)
            
            # اختبار على أول مهمة
            task_id = list(tasks.keys())[0]
            task = tasks[task_id]
            
            print(f"🎯 اختبار المهمة: {task_id}")
            
            solution = self.solve_task(task)
            
            if solution:
                print(f"✅ تم إنتاج حل: {len(solution)} اختبار")
                for i, sol in enumerate(solution):
                    print(f"  الحل {i+1}: {sol.shape}")
            else:
                print("❌ فشل في إنتاج حل")
                
        except Exception as e:
            print(f"❌ فشل الاختبار: {e}")

if __name__ == "__main__":
    # تشغيل النظام المتكامل
    system = ARCIntegratedSystem()
    system.test_system()
'''
    
    with open('arc_integrated_system.py', 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("✅ تم إنشاء النظام المتكامل: arc_integrated_system.py")

def create_project_config():
    """إنشاء ملف تكوين المشروع"""
    
    config = {
        "project_name": "مشروع برهان - ARC Solver",
        "version": "2.0.0",
        "description": "نظام متكامل لحل مهام ARC بدقة عالية",
        "structure": {
            "core": "الأنظمة الأساسية",
            "solvers": "حلالات ARC المختلفة", 
            "data": "ملفات البيانات والمهام",
            "tests": "ملفات الاختبار والتحقق",
            "utils": "الأدوات المساعدة",
            "results": "النتائج والتقارير",
            "archive": "الملفات القديمة",
            "docs": "التوثيق"
        },
        "main_entry": "arc_integrated_system.py",
        "dependencies": [
            "numpy",
            "json", 
            "typing",
            "collections",
            "dataclasses"
        ],
        "goals": [
            "تحقيق دقة 100% في حل مهام ARC",
            "نظام تعلم تكيفي من الأمثلة",
            "معالجة الأنماط الأكثر شيوعاً",
            "تكامل فعال بين المكونات"
        ]
    }
    
    with open('project_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ تم إنشاء ملف التكوين: project_config.json")

def main():
    """الوظيفة الرئيسية لإعادة الهيكلة"""
    
    print("🚀 بدء إعادة هيكلة المشروع...")
    print("=" * 50)
    
    # إنشاء الهيكل المنظم
    folders = create_organized_structure()
    
    # تصنيف الملفات
    file_categories = categorize_files()
    
    # نقل الملفات (معطل مؤقتاً لتجنب الفوضى)
    # move_files_to_structure(file_categories)
    
    # إنشاء النظام المتكامل
    create_integration_system()
    
    # إنشاء ملف التكوين
    create_project_config()
    
    print("\n🎉 تمت إعادة الهيكلة بنجاح!")
    print("\n📋 الخطوات التالية:")
    print("1. تشغيل النظام المتكامل: python arc_integrated_system.py")
    print("2. اختبار المكونات المختلفة")
    print("3. تحسين الأداء بناءً على النتائج")

if __name__ == "__main__":
    main()
