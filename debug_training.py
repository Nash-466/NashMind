from __future__ import annotations
#!/usr/bin/env python3
"""
سكربت تشخيصي لفحص نظام التدريب
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
import logging

# إعداد السجلات
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_files():
    """فحص الملفات المطلوبة"""
    print("\n=== فحص الملفات ===")
    
    files = [
        'arc-agi_training_challenges.json',
        'arc-agi_evaluation_challenges.json',
        'arc-agi_training_solutions.json',
        'arc-agi_evaluation_solutions.json'
    ]
    
    for filename in files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                print(f"✓ {filename}: {len(data)} عنصر")
            except Exception as e:
                print(f"✗ {filename}: خطأ - {e}")
        else:
            print(f"✗ {filename}: غير موجود")

def test_simple_training():
    """اختبار تدريب بسيط"""
    print("\n=== اختبار التدريب البسيط ===")
    
    try:
        # استيراد النظام
        from automated_training_loop import AutomatedTrainingLoop
        
        print("إنشاء نظام التدريب...")
        loop = AutomatedTrainingLoop()
        
        print(f"عدد بيانات التدريب: {len(loop.training_data)}")
        print(f"عدد بيانات التقييم: {len(loop.evaluation_data)}")
        print(f"عدد الأنظمة المحملة: {len(loop.orchestrator.systems)}")
        
        if loop.orchestrator.systems:
            print("\nالأنظمة المحملة:")
            for system in loop.orchestrator.systems:
                print(f"  - {system['name']}")
        
        # اختبار دورة تدريب واحدة
        if loop.training_data:
            print("\nتشغيل دورة تدريب واحدة...")
            accuracy, results = loop.train_iteration()
            print(f"الدقة: {accuracy:.2%}")
            print(f"عدد النتائج: {len(results)}")
        else:
            print("⚠ لا توجد بيانات تدريب!")
            
    except Exception as e:
        print(f"خطأ: {e}")
        import traceback
        traceback.print_exc()

def test_orchestrator():
    """اختبار المنسق"""
    print("\n=== اختبار المنسق ===")
    
    try:
        from automated_training_loop import SmartOrchestrator
        
        print("إنشاء المنسق...")
        orchestrator = SmartOrchestrator()
        
        print(f"عدد الأنظمة: {len(orchestrator.systems)}")
        
        # اختبار حل مهمة بسيطة
        test_task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[4, 3], [2, 1]]
                }
            ],
            'test': [
                {
                    'input': [[5, 6], [7, 8]],
                    'output': [[8, 7], [6, 5]]
                }
            ]
        }
        
        print("اختبار حل مهمة...")
        solution = orchestrator.solve_with_orchestration(test_task, 'test_task')
        print(f"شكل الحل: {solution.shape}")
        print(f"الحل:\n{solution}")
        
    except Exception as e:
        print(f"خطأ: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("="*60)
    print("تشخيص نظام التدريب التلقائي")
    print("="*60)
    
    check_files()
    test_orchestrator()
    test_simple_training()
    
    print("\n" + "="*60)
    print("انتهى التشخيص")
    print("="*60)

if __name__ == "__main__":
    main()
