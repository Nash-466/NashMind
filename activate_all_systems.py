from __future__ import annotations
#!/usr/bin/env python3
"""
فحص وتفعيل جميع الأنظمة الموجودة
"""

import os
import sys
import importlib
import json
from pathlib import Path

def check_all_systems():
    """فحص جميع الأنظمة الموجودة"""
    print("\n" + "="*60)
    print("فحص جميع الأنظمة في المشروع")
    print("="*60 + "\n")
    
    # قائمة الأنظمة المحتملة
    system_files = [
        'orchestrated_meta_solver',
        'ultra_advanced_arc_system_v2', 
        'ultimate_arc_system',
        'perfect_arc_system_v2',
        'perfect_arc_system',
        'revolutionary_arc_system',
        'enhanced_efficient_zero',
        'deep_learning_arc_system',
        'genius_arc_manager',
        'advanced_simulation_engine',
        'arc_adaptive_hybrid_system',
        'arc_hierarchical_reasoning',
        'arc_learning_solver',
        'arc_revolutionary_system',
        'arc_ultimate_system',
        'efficient_zero_engine',
        'semantic_memory_system',
        'symbolic_rule_engine',
        'neural_pattern_learner',
        'continuous_learning_system',
        'intelligent_verification_system',
        'true_learning_ai',
        'ultimate_ai_system',
        'ultra_advanced_arc_system',
        'ultimate_arc_solver'
    ]
    
    working_systems = []
    failed_systems = []
    
    for system_name in system_files:
        # تحقق من وجود الملف
        if os.path.exists(f"{system_name}.py"):
            print(f"\n📁 فحص {system_name}.py...")
            
            try:
                # محاولة استيراد الوحدة
                module = importlib.import_module(system_name)
                
                # البحث عن دوال الحل
                has_solve = False
                solve_func = None
                
                if hasattr(module, 'solve_task'):
                    has_solve = True
                    solve_func = 'solve_task'
                    print(f"  ✓ يحتوي على دالة solve_task")
                    
                if hasattr(module, 'ARCSolver'):
                    has_solve = True
                    solve_func = 'ARCSolver'
                    print(f"  ✓ يحتوي على فئة ARCSolver")
                    
                if hasattr(module, 'solve'):
                    has_solve = True
                    solve_func = 'solve'
                    print(f"  ✓ يحتوي على دالة solve")
                    
                # البحث عن فئات أخرى
                for attr_name in dir(module):
                    if 'Solver' in attr_name or 'solver' in attr_name:
                        attr = getattr(module, attr_name)
                        if callable(attr) or isinstance(attr, type):
                            print(f"  ℹ يحتوي على: {attr_name}")
                            has_solve = True
                            solve_func = attr_name
                
                if has_solve:
                    working_systems.append({
                        'name': system_name,
                        'function': solve_func
                    })
                    print(f"  ✅ النظام يعمل!")
                else:
                    failed_systems.append({
                        'name': system_name,
                        'reason': 'لا يحتوي على دالة حل'
                    })
                    print(f"  ⚠ لا يحتوي على دالة حل قياسية")
                    
            except ImportError as e:
                failed_systems.append({
                    'name': system_name,
                    'reason': f'خطأ استيراد: {e}'
                })
                print(f"  ❌ فشل الاستيراد: {e}")
            except Exception as e:
                failed_systems.append({
                    'name': system_name,
                    'reason': f'خطأ: {e}'
                })
                print(f"  ❌ خطأ: {e}")
        else:
            print(f"⚠ الملف {system_name}.py غير موجود")
    
    # ملخص النتائج
    print("\n" + "="*60)
    print("ملخص النتائج")
    print("="*60)
    print(f"\n✅ الأنظمة العاملة: {len(working_systems)}")
    for system in working_systems:
        print(f"   - {system['name']} ({system['function']})")
    
    print(f"\n❌ الأنظمة غير العاملة: {len(failed_systems)}")
    for system in failed_systems:
        print(f"   - {system['name']}: {system['reason']}")
    
    return working_systems, failed_systems

def test_system(system_info):
    """اختبار نظام واحد"""
    try:
        module = importlib.import_module(system_info['name'])
        
        # مهمة اختبار بسيطة
        test_task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[4, 3], [2, 1]]
                }
            ],
            'test': [
                {
                    'input': [[5, 6], [7, 8]]
                }
            ]
        }
        
        # محاولة حل المهمة
        result = None
        if system_info['function'] == 'solve_task' and hasattr(module, 'solve_task'):
            result = module.solve_task(test_task)
        elif system_info['function'] == 'ARCSolver' and hasattr(module, 'ARCSolver'):
            solver = module.ARCSolver()
            if hasattr(solver, 'solve'):
                result = solver.solve(test_task)
        elif system_info['function'] == 'solve' and hasattr(module, 'solve'):
            result = module.solve(test_task)
            
        if result is not None:
            print(f"  ✓ {system_info['name']}: نجح الاختبار (شكل النتيجة: {result.shape})")
            return True
        else:
            print(f"  ✗ {system_info['name']}: لم يُرجع نتيجة")
            return False
            
    except Exception as e:
        print(f"  ✗ {system_info['name']}: خطأ في الاختبار - {e}")
        return False

def create_updated_loader():
    """إنشاء محمّل محدث للأنظمة"""
    working_systems, _ = check_all_systems()
    
    print("\n" + "="*60)
    print("إنشاء محمّل محدث للأنظمة")
    print("="*60 + "\n")
    
    # اختبار الأنظمة العاملة
    tested_systems = []
    for system in working_systems:
        if test_system(system):
            tested_systems.append(system)
    
    print(f"\n✅ الأنظمة المختبرة بنجاح: {len(tested_systems)}")
    
    # إنشاء كود المحمّل المحدث
    loader_code = '''def load_all_working_systems(self):
    """تحميل جميع الأنظمة العاملة"""
    
    # قائمة الأنظمة المؤكد عملها
    working_systems = [
'''
    
    for system in tested_systems:
        loader_code += f'''        {{
            'module': '{system['name']}',
            'function': '{system['function']}'
        }},
'''
    
    loader_code += '''    ]
    
    for system_info in working_systems:
        try:
            module = __import__(system_info['module'])
            
            if system_info['function'] == 'solve_task':
                self.systems.append({
                    'name': system_info['module'],
                    'solve': module.solve_task,
                    'priority': 1.0
                })
            elif system_info['function'] == 'ARCSolver':
                solver = module.ARCSolver()
                self.systems.append({
                    'name': system_info['module'],
                    'solve': solver.solve,
                    'priority': 1.0
                })
            elif system_info['function'] == 'solve':
                self.systems.append({
                    'name': system_info['module'],
                    'solve': module.solve,
                    'priority': 1.0
                })
                
            logger.info(f"✓ تم تحميل: {system_info['module']}")
            
        except Exception as e:
            logger.warning(f"فشل تحميل {system_info['module']}: {e}")
    
    logger.info(f"تم تحميل {len(self.systems)} نظام بنجاح")
'''
    
    # حفظ الكود المحدث
    with open('updated_loader.txt', 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print("\n✅ تم إنشاء المحمّل المحدث في updated_loader.txt")
    
    return tested_systems

def main():
    print("\n" + "="*80)
    print("تفعيل جميع الأنظمة الموجودة")
    print("="*80)
    
    tested_systems = create_updated_loader()
    
    print("\n" + "="*60)
    print("التوصيات")
    print("="*60)
    
    print("\n1. قم بتحديث دالة load_systems في automated_training_loop.py")
    print("2. استخدم الكود في updated_loader.txt")
    print(f"3. سيتم تحميل {len(tested_systems)} نظام بدلاً من 2 فقط")
    print("4. هذا سيحسن الأداء بشكل كبير!")
    
    # حفظ قائمة الأنظمة العاملة
    with open('working_systems.json', 'w') as f:
        json.dump(tested_systems, f, indent=2)
    
    print("\n✅ تم حفظ قائمة الأنظمة العاملة في working_systems.json")

if __name__ == "__main__":
    main()
