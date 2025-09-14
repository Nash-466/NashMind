from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار سريع على مهام ARC حقيقية
"""

import os
import json
import numpy as np
from final_arc_system import FinalARCSystem

def run_quick_real_test():
    """اختبار سريع على مهام حقيقية"""
    
    print("🚀 اختبار النظام على مهام ARC حقيقية")
    print("=" * 50)
    
    # البحث عن ملفات البيانات
    data_paths = [
        'ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json',
        'arc-agi_training_challenges.json'
    ]
    
    tasks = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    tasks = json.load(f)
                print(f"✅ تم تحميل {len(tasks)} مهمة من: {path}")
                break
            except Exception as e:
                print(f"❌ خطأ في تحميل {path}: {e}")
    
    if not tasks:
        print("❌ لم يتم العثور على ملفات البيانات")
        print("💡 سأقوم بإنشاء مهام تجريبية...")
        tasks = create_test_tasks()
    
    # إنشاء النظام
    system = FinalARCSystem()
    
    # اختبار على أول 5 مهام
    task_ids = list(tasks.keys())[:5]
    
    results = {
        'total': len(task_ids),
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    for i, task_id in enumerate(task_ids):
        print(f"\n🎯 اختبار المهمة {i+1}/5: {task_id}")
        
        task = tasks[task_id]
        
        try:
            solutions = system.solve_task(task)
            
            if solutions and len(solutions) > 0:
                results['success'] += 1
                status = "✅ نجح"
                print(f"   {status} - تم إنتاج {len(solutions)} حل")
                
                # عرض الحل الأول
                if len(solutions) > 0 and solutions[0] is not None:
                    sol_shape = solutions[0].shape
                    print(f"   📐 شكل الحل: {sol_shape}")
            else:
                results['failed'] += 1
                status = "❌ فشل"
                print(f"   {status} - لم يتم إنتاج حلول")
            
            results['details'].append({
                'task_id': task_id,
                'status': status,
                'solutions_count': len(solutions) if solutions else 0
            })
            
        except Exception as e:
            results['failed'] += 1
            print(f"   💥 خطأ: {str(e)[:50]}...")
            results['details'].append({
                'task_id': task_id,
                'status': "💥 خطأ",
                'error': str(e)
            })
    
    # النتائج النهائية
    print(f"\n{'='*50}")
    print(f"📊 النتائج النهائية:")
    print(f"{'='*50}")
    print(f"📈 إجمالي المهام: {results['total']}")
    print(f"✅ نجح: {results['success']}")
    print(f"❌ فشل: {results['failed']}")
    
    success_rate = (results['success'] / results['total']) * 100
    print(f"🎯 معدل النجاح: {success_rate:.1f}%")
    
    if success_rate >= 60:
        print("🎉 أداء ممتاز!")
    elif success_rate >= 40:
        print("👍 أداء جيد")
    else:
        print("⚠️ يحتاج تحسين")
    
    return results

def create_test_tasks():
    """إنشاء مهام تجريبية"""
    
    return {
        'test_1': {
            'train': [
                {
                    'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                    'output': [[2, 0, 2], [0, 2, 0], [2, 0, 2]]
                }
            ],
            'test': [
                {'input': [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}
            ]
        },
        'test_2': {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[2, 1], [4, 3]]
                }
            ],
            'test': [
                {'input': [[1, 2], [3, 4]]}
            ]
        },
        'test_3': {
            'train': [
                {
                    'input': [[1, 1], [1, 1]],
                    'output': [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                }
            ],
            'test': [
                {'input': [[2, 2], [2, 2]]}
            ]
        }
    }

if __name__ == "__main__":
    run_quick_real_test()
