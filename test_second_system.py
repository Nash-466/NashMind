from __future__ import annotations
#!/usr/bin/env python3
import json
import numpy as np
from arc_clean_integrated_system import ARCCleanIntegratedSystem

with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_tasks = json.load(f)
with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
    solutions = json.load(f)

task_id = '0934a4d8'
task = eval_tasks[task_id]
official_solution = solutions[task_id]

print(f"🧪 اختبار النظام المتكامل النظيف على: {task_id}")

system = ARCCleanIntegratedSystem()
solutions_generated = system.solve_task(task)

print(f"✅ تم إنتاج {len(solutions_generated)} حل")

if len(solutions_generated) > 0:
    print(f"📐 شكل الحل المولد: {solutions_generated[0].shape}")
    print(f"📐 شكل الحل الرسمي: {np.array(official_solution[0]).shape}")
    
    if np.array_equal(solutions_generated[0], np.array(official_solution[0])):
        print("🎉 الحل صحيح 100%!")
    else:
        print("❌ الحل غير صحيح")
else:
    print("❌ لم يتم إنتاج أي حل")
