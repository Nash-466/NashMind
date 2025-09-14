from __future__ import annotations
#!/usr/bin/env python3
"""
اختبار يدوي سريع مع عرض النتائج مباشرة
"""

import json
import numpy as np
from final_arc_system import FinalARCSystem

# تحميل مهمة واحدة للاختبار
with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_tasks = json.load(f)

with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
    solutions = json.load(f)

# أول مهمة
task_id = '0934a4d8'
task = eval_tasks[task_id]
official_solution = solutions[task_id]

print(f"🧪 اختبار المهمة: {task_id}")
print(f"📊 عدد أمثلة التدريب: {len(task['train'])}")
print(f"📊 عدد مهام الاختبار: {len(task['test'])}")
print(f"🎯 الحل الرسمي: {len(official_solution)} حل")

# اختبار النظام
system = FinalARCSystem()
solutions_generated = system.solve_task(task)

print(f"✅ تم إنتاج {len(solutions_generated)} حل")

# مقارنة
if len(solutions_generated) == len(official_solution):
    match = True
    for i, (gen, off) in enumerate(zip(solutions_generated, official_solution)):
        if not np.array_equal(gen, np.array(off)):
            match = False
            break
    
    if match:
        print("🎉 الحل صحيح 100%!")
    else:
        print("❌ الحل غير صحيح")
        print(f"الحل المولد: {solutions_generated[0].shape}")
        print(f"الحل الرسمي: {np.array(official_solution[0]).shape}")
else:
    print(f"❌ عدد الحلول مختلف: {len(solutions_generated)} vs {len(official_solution)}")
