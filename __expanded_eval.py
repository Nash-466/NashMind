from __future__ import annotations
import json, numpy as np
from evaluation import evaluate_tasks
from arc_ultimate_mind_part7 import MasterOrchestrator
try:
    from burhan_meta_brain import MetaBrain
    meta = MetaBrain()
except Exception:
    meta = None

with open('arc-agi_training_challenges.json','r',encoding='utf-8') as f:
    tasks = json.load(f)
with open('arc-agi_training_solutions.json','r',encoding='utf-8') as f:
    solutions = json.load(f)

items = list(tasks.items())[:500]
subset = {k:v for k,v in items}

orch = MasterOrchestrator({'min_validation_score':0.99})
report = evaluate_tasks(subset, solutions, orch, meta_brain=meta, mode='fast')

overall = report.get('overall', {})
by_task = report.get('by_task', {})
exact_acc = overall.get('exact_match_accuracy', 0.0)
mean_sim = overall.get('mean_similarity', 0.0)
num_eval = overall.get('total_evaluated', 0)
solved = [tid for tid,res in by_task.items() if res.get('status')=='ok' and res.get('exact')==1]
failed = [tid for tid,res in by_task.items() if res.get('status')=='ok' and res.get('exact')==0]
no_pred = [tid for tid,res in by_task.items() if res.get('status')=='no_prediction']

print('\n=== Expanded Evaluation (first 500 training tasks) ===')
print('Evaluated:', num_eval)
print('Solved (exact):', len(solved))
print('Failed (exact):', len(failed))
print('No prediction:', len(no_pred))
print(f'Exact accuracy: {exact_acc:.4f}')
print(f'Mean similarity: {mean_sim:.4f}')
if solved:
    print('Solved task IDs (up to 30):')
    print(', '.join(solved[:30]))
if failed:
    print('Failed task IDs (up to 30):')
    print(', '.join(failed[:30]))
