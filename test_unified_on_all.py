import json
import numpy as np
from unified_solver_wrapper import solve_task

with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
    eval_tasks = json.load(f)

with open('ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
    official_solutions = json.load(f)

correct = 0
total = len(eval_tasks)
current = 0

for task_id, task in eval_tasks.items():
    current += 1
    print(f"Processing task {current}/{total} ({current/total*100:.2f}%): {task_id}")
    
    predicted = solve_task(task)
    official = official_solutions.get(task_id, [])
    
    is_correct = len(predicted) == len(official) and all(np.array_equal(p, np.array(o)) for p, o in zip(predicted, official))
    if is_correct:
        correct += 1
        print("  - Correct")
    else:
        print("  - Incorrect")
    
    print(f"Current accuracy: {correct/current*100:.2f}%")

print(f"\nFinal Results: {correct}/{total} ({correct/total*100:.2f}%)")