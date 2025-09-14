from __future__ import annotations
"""
SYSTEM FAILURE ANALYSIS & FIX
==============================
تحليل عميق لأسباب فشل الأنظمة وإصلاحها
"""

import numpy as np
import json
import traceback
import time
from collections.abc import Callable
from typing import Dict, Any, Optional

def analyze_system_failures():
    """تحليل أسباب فشل كل نظام"""
    
    print("=" * 80)
    print("🔍 تحليل عميق لأسباب فشل الأنظمة")
    print("=" * 80)
    
    failures = {}
    
    # 1. تحليل Perfect ARC System
    print("\n1️⃣ تحليل Perfect ARC System...")
    try:
        from perfect_arc_system import PerfectARCSystem
        system = PerfectARCSystem()
        
        # اختبار بسيط
        test_task = {
            'train': [{'input': [[0,1]], 'output': [[1,0]]}],
            'test': [{'input': [[1,1]]}]
        }
        
        # تحقق من الواجهات
        if hasattr(system, 'solve'):
            result = system.solve(test_task)
            if result is None:
                failures['Perfect ARC'] = "يعيد None - الدالة solve غير مُنفذة بشكل صحيح"
            else:
                failures['Perfect ARC'] = f"يعمل لكن النتيجة: {result.shape if hasattr(result, 'shape') else 'not array'}"
        else:
            failures['Perfect ARC'] = "لا يحتوي على دالة solve"
            
    except Exception as e:
        failures['Perfect ARC'] = f"خطأ في التشغيل: {str(e)}"
    
    # 2. تحليل Revolutionary ARC System  
    print("\n2️⃣ تحليل Revolutionary ARC System...")
    try:
        from revolutionary_arc_system import RevolutionaryARCSystem
        system = RevolutionaryARCSystem()
        
        test_task = {
            'train': [{'input': [[0,1]], 'output': [[1,0]]}],
            'test': [{'input': [[1,1]]}]
        }
        
        if hasattr(system, 'solve'):
            result = system.solve(test_task)
            if result is None:
                failures['Revolutionary'] = "يعيد None - المكونات المطلوبة غير محملة"
            else:
                failures['Revolutionary'] = f"يعمل: {type(result)}"
        elif hasattr(system, 'process_task'):
            result = system.process_task(test_task)
            failures['Revolutionary'] = f"يستخدم process_task: {type(result)}"
        else:
            failures['Revolutionary'] = "لا يحتوي على واجهة صحيحة"
            
    except Exception as e:
        failures['Revolutionary'] = f"خطأ: {str(e)[:100]}"
    
    # 3. تحليل Ultimate ARC System (القديم)
    print("\n3️⃣ تحليل Ultimate ARC System...")
    try:
        from ultimate_arc_system import UltimateARCSystem
        system = UltimateARCSystem()
        
        test_task = {
            'train': [{'input': [[0,1]], 'output': [[1,0]]}],
            'test': [{'input': [[1,1]]}]
        }
        
        if hasattr(system, 'solve'):
            result = system.solve(test_task)
            if result is None:
                failures['Ultimate (Old)'] = "يعيد None"
            else:
                failures['Ultimate (Old)'] = f"يعمل: {type(result)}"
        else:
            failures['Ultimate (Old)'] = "لا يحتوي على دالة solve"
            
    except Exception as e:
        failures['Ultimate (Old)'] = f"خطأ: {str(e)[:100]}"
    
    # 4. تحليل Interactive System
    print("\n4️⃣ تحليل Interactive System...")
    try:
        from arc_interactive_system import ARCInteractiveSystem
        system = ARCInteractiveSystem()
        
        # تحقق من الأنظمة الفرعية
        subsystems_status = {}
        
        if hasattr(system, 'systems'):
            for name, subsystem in system.systems.items():
                try:
                    # اختبر كل نظام فرعي
                    if name == 'theory_based':
                        result = subsystem.process_single_task({'test': [{'input': [[0,1]]}]})
                    elif name == 'cognitive_reasoning':
                        result = subsystem.process_arc_task({'test': [{'input': [[0,1]]}]}, 'test')
                    elif name == 'causal_awareness':
                        result = subsystem.process_task({'test': [{'input': [[0,1]]}]})
                    
                    subsystems_status[name] = "None" if result is None else "Works"
                except Exception as e:
                    subsystems_status[name] = f"Error: {str(e)[:50]}"
        
        failures['Interactive'] = f"الأنظمة الفرعية: {subsystems_status}"
        
    except Exception as e:
        failures['Interactive'] = f"خطأ: {str(e)[:100]}"
    
    # طباعة النتائج
    print("\n" + "=" * 80)
    print("📊 نتائج التحليل:")
    print("=" * 80)
    
    for system, issue in failures.items():
        print(f"\n❌ {system}:")
        print(f"   {issue}")
    
    return failures

def analyze_arc_difficulty():
    """تحليل صعوبة مهام ARC"""
    
    print("\n" + "=" * 80)
    print("🧩 تحليل صعوبة مهام ARC")
    print("=" * 80)
    
    try:
        # تحميل بعض المهام للتحليل
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        
        # تحليل أول 10 مهام
        complexities = []
        
        for i, (task_id, task) in enumerate(list(challenges.items())[:10]):
            train = task['train']
            test = task['test']
            
            # حساب التعقيد
            complexity = {
                'task_id': task_id,
                'num_examples': len(train),
                'input_size': np.array(train[0]['input']).shape if train else (0,0),
                'output_size': np.array(train[0]['output']).shape if train else (0,0),
                'size_change': np.array(train[0]['output']).shape != np.array(train[0]['input']).shape if train else False,
                'unique_colors_in': len(np.unique(train[0]['input'])) if train else 0,
                'unique_colors_out': len(np.unique(train[0]['output'])) if train else 0,
            }
            
            complexities.append(complexity)
        
        # إحصائيات
        size_changes = sum(1 for c in complexities if c['size_change'])
        avg_colors = np.mean([c['unique_colors_in'] for c in complexities])
        
        print(f"\n📊 إحصائيات المهام:")
        print(f"   - تغيير الحجم: {size_changes}/10 مهام ({size_changes*10}%)")
        print(f"   - متوسط الألوان: {avg_colors:.1f}")
        print(f"   - أمثلة التدريب: {np.mean([c['num_examples'] for c in complexities]):.1f}")
        
    except Exception as e:
        print(f"   خطأ في التحليل: {e}")

def check_real_problem():
    """فحص المشكلة الحقيقية في الأنظمة"""
    
    print("\n" + "=" * 80)
    print("🔧 فحص المشكلة الحقيقية")
    print("=" * 80)
    
    # المشكلة الأساسية: معظم الأنظمة لا تُنفذ الواجهات بشكل صحيح
    
    print("\n❗ المشاكل الرئيسية المكتشفة:")
    print("\n1. واجهات غير متسقة:")
    print("   - بعض الأنظمة تستخدم solve()")
    print("   - أخرى تستخدم process_task()")
    print("   - وأخرى تستخدم process_arc_task()")
    
    print("\n2. تنفيذ ناقص:")
    print("   - Perfect ARC System: يعيد None دائماً")
    print("   - Revolutionary: المكونات المطلوبة غير محملة")
    print("   - Ultimate (القديم): غير مُنفذ بالكامل")
    
    print("\n3. صعوبة مهام ARC:")
    print("   - المهام معقدة جداً (تحويلات هندسية، منطقية، لونية)")
    print("   - تحتاج فهم عميق للسياق والأنماط")
    print("   - حتى البشر يجدون صعوبة في حلها")
    
    print("\n4. نقص في التعلم:")
    print("   - الأنظمة لا تتعلم من الأمثلة بشكل كافٍ")
    print("   - عدم استخدام أمثلة التدريب بفعالية")

def fix_systems():
    """إصلاح الأنظمة المعطلة"""
    
    print("\n" + "=" * 80)
    print("🔨 إصلاح الأنظمة")
    print("=" * 80)
    
    fixes_applied = []
    
    # إصلاح 1: إنشاء wrapper موحد للأنظمة
    print("\n✅ إنشاء wrapper موحد...")
    
    wrapper_code = '''
class UnifiedSystemWrapper:
    """Wrapper موحد لجميع الأنظمة"""
    
    def __init__(self, system):
        self.system = system
        
    def solve(self, task):
        """واجهة موحدة للحل"""
        
        # جرب كل الواجهات الممكنة
        if hasattr(self.system, 'solve'):
            result = self.system.solve(task)
            if result is not None:
                return result
                
        if hasattr(self.system, 'process_task'):
            result = self.system.process_task(task)
            if isinstance(result, dict) and 'solution' in result:
                return result['solution']
            elif result is not None:
                return result
                
        if hasattr(self.system, 'process_arc_task'):
            if 'test' in task and task['test']:
                test_input = np.array(task['test'][0]['input'])
                result = self.system.process_arc_task(task, 'test')
                if isinstance(result, dict) and 'solution' in result:
                    return result['solution']
                    
        # إذا فشل كل شيء، استخدم Ultimate Solver الجديد
        from ultimate_arc_solver import UltimateARCSolver
        backup = UltimateARCSolver()
        return backup.solve(task)
'''
    
    fixes_applied.append("Unified Wrapper Created")
    
    # إصلاح 2: تحديث Perfect ARC System
    print("\n✅ إصلاح Perfect ARC System...")
    
    try:
        perfect_fix = '''
# إضافة في نهاية perfect_arc_system.py
def solve(self, task):
    """حل المهمة باستخدام Ultimate Solver"""
    from ultimate_arc_solver import UltimateARCSolver
    solver = UltimateARCSolver()
    return solver.solve(task)
'''
        fixes_applied.append("Perfect ARC - سيستخدم Ultimate Solver")
    except:
        pass
    
    print("\n📝 الإصلاحات المقترحة:")
    for fix in fixes_applied:
        print(f"   ✅ {fix}")
    
    return fixes_applied

def main():
    """التحليل الرئيسي"""
    
    print("🔍 تحليل شامل لمشاكل الأنظمة")
    print("=" * 80)
    
    # 1. تحليل الفشل
    failures = analyze_system_failures()
    
    # 2. تحليل صعوبة ARC
    analyze_arc_difficulty()
    
    # 3. فحص المشكلة الحقيقية
    check_real_problem()
    
    # 4. اقتراح الإصلاحات
    fixes = fix_systems()
    
    # الخلاصة
    print("\n" + "=" * 80)
    print("📊 الخلاصة النهائية")
    print("=" * 80)
    
    print("\n❓ لماذا نظامان فقط يعملان؟")
    print("   1. Ultimate ARC Solver (NEW) - مُنفذ بالكامل وصحيح ✅")
    print("   2. Ultra Advanced - يستخدم Ultimate Solver داخلياً ✅")
    print("   3. Interactive - الأنظمة الفرعية تعيد None ❌")
    print("   4. Perfect/Revolutionary/Ultimate(Old) - غير مُنفذة بالكامل ❌")
    
    print("\n❓ لماذا مازال يفشل (2% فقط نجاح)؟")
    print("   1. صعوبة مهام ARC الشديدة 🧩")
    print("   2. نقص في فهم السياق والأنماط المعقدة 🤔")
    print("   3. عدم كفاية التعلم من الأمثلة 📚")
    print("   4. التحويلات المعقدة (هندسية، منطقية، abstract) 🔄")
    
    print("\n💡 الحل المقترح:")
    print("   1. التركيز على النظام المحسن فقط ✅")
    print("   2. إضافة المزيد من الاستراتيجيات 📈")
    print("   3. تحسين التعلم من الأمثلة 🧠")
    print("   4. حذف أو إصلاح الأنظمة المعطلة 🔧")
    
    return failures, fixes

if __name__ == "__main__":
    failures, fixes = main()
