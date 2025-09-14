from __future__ import annotations
#!/usr/bin/env python3
"""
فحص مشكلة دقة التقييم 0%
"""

import json
import numpy as np
from pathlib import Path

def check_evaluation_data():
    """فحص بيانات التقييم"""
    print("\n=== فحص بيانات التقييم ===\n")
    
    # تحميل البيانات
    eval_path = Path("arc-agi_evaluation_challenges.json")
    solutions_path = Path("arc-agi_evaluation_solutions.json")
    
    if not eval_path.exists():
        print("✗ لا يوجد ملف تقييم!")
        return
        
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)
    
    print(f"عدد مهام التقييم: {len(eval_data)}")
    
    # تحميل الحلول
    solutions = {}
    if solutions_path.exists():
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
        print(f"عدد الحلول المتاحة: {len(solutions)}")
    else:
        print("⚠ لا توجد حلول للمقارنة")
    
    # فحص أول 5 مهام
    print("\n=== فحص أول 5 مهام ===\n")
    
    for idx, (task_id, task_data) in enumerate(list(eval_data.items())[:5]):
        print(f"\n{idx+1}. المهمة: {task_id}")
        
        # فحص هيكل المهمة
        if 'train' in task_data:
            print(f"   - عدد أمثلة التدريب: {len(task_data['train'])}")
            if task_data['train']:
                first_train = task_data['train'][0]
                input_shape = np.array(first_train['input']).shape
                output_shape = np.array(first_train['output']).shape
                print(f"   - شكل الإدخال: {input_shape}")
                print(f"   - شكل الإخراج: {output_shape}")
        
        if 'test' in task_data:
            print(f"   - عدد أمثلة الاختبار: {len(task_data['test'])}")
            if task_data['test']:
                test_example = task_data['test'][0]
                test_input_shape = np.array(test_example['input']).shape
                print(f"   - شكل إدخال الاختبار: {test_input_shape}")
                
                # هل يوجد حل في الاختبار؟
                if 'output' in test_example:
                    test_output = np.array(test_example['output'])
                    print(f"   - ⚠ يوجد حل في الاختبار! شكله: {test_output.shape}")
                else:
                    print(f"   - ✓ لا يوجد حل في الاختبار (صحيح)")
        
        # فحص الحل المتوقع
        if task_id in solutions:
            solution = np.array(solutions[task_id][0])
            print(f"   - الحل المتوقع: شكل {solution.shape}")
            print(f"   - عينة من الحل:\n{solution[:3, :3] if solution.shape[0] >= 3 and solution.shape[1] >= 3 else solution}")

def test_solver_on_evaluation():
    """اختبار الحل على مهام التقييم"""
    print("\n=== اختبار الحل على التقييم ===\n")
    
    # استيراد الحل
    from enhanced_arc_solver import solve_task
    
    # تحميل البيانات
    with open("arc-agi_evaluation_challenges.json", 'r') as f:
        eval_data = json.load(f)
    
    solutions = {}
    if Path("arc-agi_evaluation_solutions.json").exists():
        with open("arc-agi_evaluation_solutions.json", 'r') as f:
            solutions = json.load(f)
    
    # اختبار أول 5 مهام
    correct = 0
    total = 0
    
    for idx, (task_id, task_data) in enumerate(list(eval_data.items())[:5]):
        print(f"\nاختبار المهمة {idx+1}: {task_id}")
        
        # حل المهمة
        solution = solve_task(task_data)
        print(f"  - شكل الحل: {solution.shape}")
        
        # المقارنة مع الحل الصحيح
        if task_id in solutions:
            expected = np.array(solutions[task_id][0])
            print(f"  - الشكل المتوقع: {expected.shape}")
            
            if solution.shape == expected.shape:
                accuracy = np.mean(solution == expected)
                print(f"  - التطابق: {accuracy:.2%}")
                if accuracy == 1.0:
                    correct += 1
                    print("  - ✓ حل صحيح!")
                else:
                    print("  - ✗ حل خاطئ")
            else:
                print("  - ✗ الأشكال غير متطابقة!")
        else:
            print("  - ⚠ لا يوجد حل للمقارنة")
        
        total += 1
    
    print(f"\n=== النتيجة النهائية ===")
    print(f"صحيح: {correct}/{total} ({correct/total*100:.1f}%)")

def check_evaluation_training_mismatch():
    """فحص الاختلاف بين التدريب والتقييم"""
    print("\n=== فحص الاختلاف بين التدريب والتقييم ===\n")
    
    # تحميل البيانات
    with open("arc-agi_training_challenges.json", 'r') as f:
        train_data = json.load(f)
    
    with open("arc-agi_evaluation_challenges.json", 'r') as f:
        eval_data = json.load(f)
    
    # فحص التداخل
    train_ids = set(train_data.keys())
    eval_ids = set(eval_data.keys())
    
    overlap = train_ids & eval_ids
    
    print(f"عدد مهام التدريب: {len(train_ids)}")
    print(f"عدد مهام التقييم: {len(eval_ids)}")
    print(f"التداخل: {len(overlap)} مهمة")
    
    if overlap:
        print(f"⚠ يوجد تداخل! المهام المشتركة: {list(overlap)[:5]}")
    else:
        print("✓ لا يوجد تداخل (جيد)")
    
    # فحص الخصائص
    print("\n=== مقارنة الخصائص ===\n")
    
    # حجم الشبكات
    train_sizes = []
    eval_sizes = []
    
    for task in list(train_data.values())[:100]:
        if 'train' in task and task['train']:
            grid = np.array(task['train'][0]['input'])
            train_sizes.append(grid.shape)
    
    for task in eval_data.values():
        if 'train' in task and task['train']:
            grid = np.array(task['train'][0]['input'])
            eval_sizes.append(grid.shape)
    
    print(f"أحجام شبكات التدريب (عينة): {train_sizes[:5]}")
    print(f"أحجام شبكات التقييم (عينة): {eval_sizes[:5]}")
    
    # متوسط الأحجام
    if train_sizes:
        avg_train_h = np.mean([s[0] for s in train_sizes])
        avg_train_w = np.mean([s[1] for s in train_sizes])
        print(f"متوسط حجم التدريب: {avg_train_h:.1f} × {avg_train_w:.1f}")
    
    if eval_sizes:
        avg_eval_h = np.mean([s[0] for s in eval_sizes])
        avg_eval_w = np.mean([s[1] for s in eval_sizes])
        print(f"متوسط حجم التقييم: {avg_eval_h:.1f} × {avg_eval_w:.1f}")

def main():
    print("="*60)
    print("تشخيص مشكلة دقة التقييم 0%")
    print("="*60)
    
    check_evaluation_data()
    check_evaluation_training_mismatch()
    test_solver_on_evaluation()
    
    print("\n" + "="*60)
    print("انتهى التشخيص")
    print("="*60)

if __name__ == "__main__":
    main()
