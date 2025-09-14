from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار شامل لجميع الأنظمة مع عرض النتائج مباشرة
"""

import os
import sys
import json
import numpy as np
import time
import traceback
from collections.abc import Callable
from typing import Dict, List, Any

def test_system_safely(system_name: str, system_class, test_task: Dict) -> Dict:
    """اختبار نظام واحد بأمان"""
    
    result = {
        'system_name': system_name,
        'status': 'unknown',
        'execution_time': 0,
        'solutions_count': 0,
        'error': None,
        'success': False
    }
    
    print(f"🧪 اختبار {system_name}...")
    
    start_time = time.time()
    
    try:
        # إنشاء النظام
        system = system_class()
        
        # حل المهمة
        solutions = system.solve_task(test_task)
        
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        if solutions and len(solutions) > 0:
            result['status'] = 'نجح'
            result['success'] = True
            result['solutions_count'] = len(solutions)
            print(f"   ✅ نجح في {execution_time:.2f}s - {len(solutions)} حل")
            
            # عرض الحل الأول
            if solutions[0] is not None:
                print(f"   📋 الحل: {solutions[0].tolist()}")
        else:
            result['status'] = 'فشل - لا حلول'
            print(f"   ❌ فشل في {execution_time:.2f}s - لا حلول")
            
    except Exception as e:
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        result['status'] = 'خطأ'
        result['error'] = str(e)
        print(f"   💥 خطأ في {execution_time:.2f}s: {str(e)[:50]}...")
    
    return result

def run_comprehensive_system_test():
    """اختبار شامل لجميع الأنظمة"""
    
    print("🚀 اختبار شامل لجميع أنظمة ARC")
    print("=" * 60)
    
    # مهمة اختبار بسيطة
    test_task = {
        'train': [
            {
                'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                'output': [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
            }
        ],
        'test': [
            {'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}
        ]
    }
    
    # قائمة الأنظمة للاختبار
    systems_to_test = [
        # الأنظمة الجديدة
        ('final_arc_system', 'FinalARCSystem'),
        ('arc_clean_integrated_system', 'ARCCleanIntegratedSystem'),
        ('arc_ultimate_perfect_system', 'ARCUltimatePerfectSolver'),
        ('arc_core_redesign', 'ARCCoreSolver'),
        
        # الأنظمة الكبيرة السابقة
        ('perfect_arc_system', 'PerfectARCSystem'),
        ('ultimate_ai_system', 'UltimateAISystem'),
        ('ultra_advanced_arc_system', 'UltraAdvancedARCSystem'),
        ('genius_arc_manager', 'GeniusARCManager'),
    ]
    
    results = []
    successful_systems = []
    failed_systems = []
    
    for system_file, class_name in systems_to_test:
        print(f"\n{'='*50}")
        print(f"🎯 اختبار: {system_file}")
        print(f"{'='*50}")
        
        try:
            # محاولة استيراد النظام
            if os.path.exists(f"{system_file}.py"):
                # استيراد ديناميكي
                spec = __import__(system_file)
                
                if hasattr(spec, class_name):
                    system_class = getattr(spec, class_name)
                    
                    # اختبار النظام
                    result = test_system_safely(system_file, system_class, test_task)
                    results.append(result)
                    
                    if result['success']:
                        successful_systems.append(system_file)
                    else:
                        failed_systems.append(system_file)
                        
                else:
                    print(f"   ❌ الكلاس {class_name} غير موجود")
                    results.append({
                        'system_name': system_file,
                        'status': 'كلاس غير موجود',
                        'success': False
                    })
                    failed_systems.append(system_file)
            else:
                print(f"   ❌ الملف {system_file}.py غير موجود")
                results.append({
                    'system_name': system_file,
                    'status': 'ملف غير موجود',
                    'success': False
                })
                failed_systems.append(system_file)
                
        except Exception as e:
            print(f"   💥 خطأ في الاستيراد: {str(e)[:50]}...")
            results.append({
                'system_name': system_file,
                'status': 'خطأ استيراد',
                'error': str(e),
                'success': False
            })
            failed_systems.append(system_file)
    
    # النتائج النهائية
    print(f"\n{'='*60}")
    print(f"📊 النتائج النهائية")
    print(f"{'='*60}")
    
    print(f"📈 إجمالي الأنظمة المختبرة: {len(results)}")
    print(f"✅ الأنظمة الناجحة: {len(successful_systems)}")
    print(f"❌ الأنظمة الفاشلة: {len(failed_systems)}")
    
    success_rate = (len(successful_systems) / len(results)) * 100 if results else 0
    print(f"🎯 معدل النجاح: {success_rate:.1f}%")
    
    # تفاصيل الأنظمة الناجحة
    if successful_systems:
        print(f"\n✅ الأنظمة الناجحة:")
        print("-" * 40)
        for system in successful_systems:
            result = next(r for r in results if r['system_name'] == system)
            print(f"   🎉 {system} ({result['execution_time']:.2f}s)")
    
    # تفاصيل الأنظمة الفاشلة
    if failed_systems:
        print(f"\n❌ الأنظمة الفاشلة:")
        print("-" * 40)
        for system in failed_systems:
            result = next(r for r in results if r['system_name'] == system)
            print(f"   💔 {system} - {result['status']}")
    
    # جدول مقارنة
    print(f"\n📋 جدول المقارنة:")
    print("-" * 70)
    print(f"{'النظام':<35} {'الحالة':<15} {'الوقت':<10}")
    print("-" * 70)
    
    for result in results:
        status = result['status']
        time_str = f"{result.get('execution_time', 0):.2f}s" if result.get('execution_time') else "N/A"
        print(f"{result['system_name']:<35} {status:<15} {time_str:<10}")
    
    # التوصيات
    print(f"\n💡 التوصيات:")
    print("-" * 20)
    
    if successful_systems:
        best_system = min([r for r in results if r['success']], 
                         key=lambda x: x.get('execution_time', float('inf')))
        print(f"🏆 أفضل نظام: {best_system['system_name']}")
        print(f"   ⚡ الوقت: {best_system['execution_time']:.2f}s")
    
    if success_rate >= 50:
        print("🎉 أداء جيد! معظم الأنظمة تعمل")
    else:
        print("⚠️ يحتاج تحسين - معظم الأنظمة لا تعمل")
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_system_test()
    
    # حفظ النتائج
    timestamp = int(time.time())
    with open(f'comprehensive_test_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 تم حفظ النتائج في: comprehensive_test_results_{timestamp}.json")
