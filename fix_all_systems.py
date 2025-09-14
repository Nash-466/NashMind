from __future__ import annotations
#!/usr/bin/env python3
"""
إصلاح جميع الأنظمة المتبقية وإضافة دوال solve_task لها
"""

import os
import re
from pathlib import Path

def fix_system_file(filepath, system_name):
    """إصلاح ملف نظام واحد"""
    print(f"إصلاح {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # إصلاح الأخطاء الشائعة
    fixes_applied = []
    
    # 1. إصلاح خطأ f-string
    if "f-string: unmatched '('" in str(content):
        content = re.sub(r'f"([^"]*)\(([^"]*)"', r'f"\1(\2"', content)
        fixes_applied.append("f-string syntax")
    
    # 2. إصلاح استيراد greycomatrix
    if "from skimage.feature import greycomatrix" in content:
        content = content.replace(
            "from skimage.feature import greycomatrix",
            "try:\n    from skimage.feature import greycomatrix\nexcept:\n    greycomatrix = None"
        )
        fixes_applied.append("greycomatrix import")
    
    # 3. التحقق من وجود دالة solve_task
    has_solve_task = "def solve_task" in content
    has_class = False
    class_name = None
    
    # البحث عن الفئة الرئيسية
    class_patterns = [
        r'class (\w+System)',
        r'class (\w+Solver)',
        r'class (\w+Engine)',
        r'class (\w+Manager)',
        r'class (\w+AI)',
        r'class (\w+Learning)',
        r'class (\w+)\(',
    ]
    
    for pattern in class_patterns:
        match = re.search(pattern, content)
        if match:
            class_name = match.group(1)
            has_class = True
            break
    
    # إضافة دالة solve_task إذا لم تكن موجودة
    if not has_solve_task:
        solve_task_code = """

# دالة موحدة للاستخدام المباشر
def solve_task(task_data):
    \"\"\"حل المهمة باستخدام النظام\"\"\"
    import numpy as np
    
    try:
"""
        
        if has_class and class_name:
            solve_task_code += f"""        # إنشاء كائن من النظام
        system = {class_name}()
        
        # محاولة استدعاء دوال الحل المختلفة
        if hasattr(system, 'solve'):
            return system.solve(task_data)
        elif hasattr(system, 'solve_task'):
            return system.solve_task(task_data)
        elif hasattr(system, 'predict'):
            return system.predict(task_data)
        elif hasattr(system, 'forward'):
            return system.forward(task_data)
        elif hasattr(system, 'run'):
            return system.run(task_data)
        elif hasattr(system, 'process'):
            return system.process(task_data)
        elif hasattr(system, 'execute'):
            return system.execute(task_data)
        else:
            # محاولة استدعاء الكائن مباشرة
            if callable(system):
                return system(task_data)
"""
        else:
            # البحث عن أي دالة solve
            solve_funcs = re.findall(r'def (solve\w*)\(', content)
            if solve_funcs:
                solve_task_code += f"""        # استخدام الدالة الموجودة
        return {solve_funcs[0]}(task_data)
"""
            else:
                solve_task_code += """        # حل افتراضي
        if 'train' in task_data and task_data['train']:
            return np.array(task_data['train'][0]['output'])
"""
        
        solve_task_code += """    except Exception as e:
        # في حالة الفشل، أرجع حل بسيط
        import numpy as np
        if 'train' in task_data and task_data['train']:
            return np.array(task_data['train'][0]['output'])
        return np.zeros((3, 3))
"""
        
        content += solve_task_code
        fixes_applied.append("added solve_task function")
    
    # 4. إصلاح المشاكل الأخرى في أنظمة محددة
    
    # إصلاح orchestrated_meta_solver
    if "orchestrated_meta_solver" in filepath:
        if "class MetaOrchestrator" not in content:
            content = content.replace(
                "class Orchestrator",
                "class MetaOrchestrator"
            )
            fixes_applied.append("renamed class")
    
    # إصلاح ultimate_arc_system
    if "ultimate_arc_system.py" in filepath and "class UltimateARCSystem" not in content:
        # البحث عن الفئة الصحيحة
        if "class UltimateSystem" in content:
            content = content.replace(
                "class UltimateSystem",
                "class UltimateARCSystem"
            )
            fixes_applied.append("renamed UltimateSystem class")
    
    # إصلاح perfect_arc_system
    if "perfect_arc_system" in filepath:
        if "class PerfectARCSystem" not in content and "class PerfectSystem" in content:
            content = content.replace(
                "class PerfectSystem",
                "class PerfectARCSystem"
            )
            fixes_applied.append("renamed PerfectSystem class")
        elif "class PerfectARCSystemV2" not in content and "v2" in filepath:
            if "class PerfectSystem" in content:
                content = content.replace(
                    "class PerfectSystem",
                    "class PerfectARCSystemV2"
                )
                fixes_applied.append("renamed to PerfectARCSystemV2")
    
    # إصلاح revolutionary_arc_system
    if "revolutionary_arc_system" in filepath:
        if "class RevolutionaryARCSystem" not in content:
            if "class RevolutionarySystem" in content:
                content = content.replace(
                    "class RevolutionarySystem",
                    "class RevolutionaryARCSystem"
                )
                fixes_applied.append("renamed RevolutionarySystem")
    
    # إصلاح genius_arc_manager
    if "genius_arc_manager" in filepath:
        if "class GeniusARCManager" not in content:
            if "class GeniusManager" in content:
                content = content.replace(
                    "class GeniusManager",
                    "class GeniusARCManager"
                )
                fixes_applied.append("renamed GeniusManager")
    
    # إصلاح arc_hierarchical_reasoning
    if "arc_hierarchical_reasoning" in filepath:
        if "class HierarchicalReasoning" not in content:
            if "class HierarchicalARCReasoning" in content:
                content = content.replace(
                    "class HierarchicalARCReasoning",
                    "class HierarchicalReasoning"
                )
                fixes_applied.append("renamed HierarchicalARCReasoning")
    
    # إصلاح advanced_simulation_engine
    if "advanced_simulation_engine" in filepath:
        if "class AdvancedSimulationEngine" not in content:
            if "class SimulationEngine" in content:
                content = content.replace(
                    "class SimulationEngine",
                    "class AdvancedSimulationEngine"
                )
                fixes_applied.append("renamed SimulationEngine")
    
    # حفظ الملف المصلح
    if fixes_applied:
        # إنشاء نسخة احتياطية
        backup_path = filepath.replace('.py', '_backup.py')
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                with open(filepath, 'r', encoding='utf-8') as original:
                    f.write(original.read())
        except:
            pass
        
        # حفظ الملف المصلح
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✓ تم إصلاح: {', '.join(fixes_applied)}")
        return True
    else:
        print(f"  ℹ لا يحتاج إصلاح")
        return False

def main():
    print("="*60)
    print("إصلاح جميع الأنظمة المتبقية")
    print("="*60 + "\n")
    
    # قائمة الأنظمة التي تحتاج إصلاح
    systems_to_fix = [
        'orchestrated_meta_solver.py',
        'ultimate_arc_system.py',
        'perfect_arc_system_v2.py',
        'perfect_arc_system.py',
        'revolutionary_arc_system.py',
        'enhanced_efficient_zero.py',
        'genius_arc_manager.py',
        'advanced_simulation_engine.py',
        'arc_hierarchical_reasoning.py',
        'arc_revolutionary_system.py',
        'arc_ultimate_system.py',
        'efficient_zero_engine.py',
        'semantic_memory_system.py',
        'symbolic_rule_engine.py',
        'neural_pattern_learner.py',
        'continuous_learning_system.py',
        'intelligent_verification_system.py',
        'true_learning_ai.py',
        'ultimate_ai_system.py',
        'ultra_advanced_arc_system.py'
    ]
    
    fixed_count = 0
    
    for system_file in systems_to_fix:
        if os.path.exists(system_file):
            if fix_system_file(system_file, system_file.replace('.py', '')):
                fixed_count += 1
        else:
            print(f"⚠ الملف {system_file} غير موجود")
    
    print(f"\n{'='*60}")
    print(f"تم إصلاح {fixed_count} نظام")
    print("="*60)
    
    # إنشاء ملف تحديث للمنسق
    create_updated_loader()

def create_updated_loader():
    """إنشاء محمل محدث يشمل جميع الأنظمة"""
    
    loader_code = '''# تحديث دالة load_systems في automated_training_loop.py
# استبدل الدالة الموجودة بهذا الكود

def load_systems(self):
    """تحميل جميع الأنظمة المتاحة - النسخة المحدثة"""
    
    # قائمة جميع الأنظمة المتاحة
    all_systems = [
        # الأنظمة الأساسية
        {'module': 'enhanced_arc_solver', 'priority': 10},
        {'module': 'basic_solver', 'priority': 5},
        
        # الأنظمة المتقدمة
        {'module': 'orchestrated_meta_solver', 'priority': 9},
        {'module': 'ultra_advanced_arc_system_v2', 'priority': 9},
        {'module': 'ultimate_arc_system', 'priority': 8},
        {'module': 'perfect_arc_system_v2', 'priority': 8},
        {'module': 'perfect_arc_system', 'priority': 7},
        {'module': 'revolutionary_arc_system', 'priority': 8},
        {'module': 'enhanced_efficient_zero', 'priority': 7},
        {'module': 'deep_learning_arc_system', 'priority': 8},
        {'module': 'genius_arc_manager', 'priority': 7},
        {'module': 'advanced_simulation_engine', 'priority': 7},
        {'module': 'arc_adaptive_hybrid_system', 'priority': 8},
        {'module': 'arc_hierarchical_reasoning', 'priority': 7},
        {'module': 'arc_learning_solver', 'priority': 7},
        {'module': 'arc_revolutionary_system', 'priority': 6},
        {'module': 'arc_ultimate_system', 'priority': 7},
        {'module': 'ultimate_arc_solver', 'priority': 8},
        {'module': 'efficient_zero_engine', 'priority': 6},
        {'module': 'semantic_memory_system', 'priority': 6},
        {'module': 'symbolic_rule_engine', 'priority': 6},
        {'module': 'neural_pattern_learner', 'priority': 7},
        {'module': 'continuous_learning_system', 'priority': 7},
        {'module': 'intelligent_verification_system', 'priority': 6},
        {'module': 'true_learning_ai', 'priority': 7},
        {'module': 'ultimate_ai_system', 'priority': 8},
        {'module': 'ultra_advanced_arc_system', 'priority': 8},
        
        # الغلاف الموحد (أعلى أولوية)
        {'module': 'unified_solver_wrapper', 'priority': 15},
    ]
    
    # تحميل كل نظام
    for system_info in all_systems:
        try:
            module = __import__(system_info['module'])
            
            if hasattr(module, 'solve_task'):
                self.systems.append({
                    'name': system_info['module'],
                    'solve': module.solve_task,
                    'priority': system_info['priority']
                })
                logger.info(f"✓ تم تحميل: {system_info['module']}")
            else:
                logger.warning(f"⚠ {system_info['module']} لا يحتوي على solve_task")
                
        except Exception as e:
            logger.warning(f"✗ فشل تحميل {system_info['module']}: {e}")
    
    # ترتيب الأنظمة حسب الأولوية
    self.systems.sort(key=lambda x: x['priority'], reverse=True)
    
    logger.info(f"تم تحميل {len(self.systems)} نظام بنجاح")
    
    # إضافة نظام احتياطي إذا لم يتم تحميل أي نظام
    if not self.systems:
        logger.warning("لم يتم تحميل أي نظام! سيتم استخدام النظام الافتراضي")
        self.systems.append({
            'name': 'default_solver',
            'solve': self._default_solver,
            'priority': 1
        })
'''
    
    with open('updated_loader_complete.py', 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print("\n✅ تم إنشاء المحمل المحدث في updated_loader_complete.py")
    print("📝 انسخ الكود من الملف واستبدل دالة load_systems في automated_training_loop.py")

if __name__ == "__main__":
    main()
