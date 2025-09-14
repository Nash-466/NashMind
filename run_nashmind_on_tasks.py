from __future__ import annotations
import json
import os
import sys
from collections.abc import Callable
from typing import Dict, Any
import numpy as np

# make NashMind importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NM_DIR = os.path.join(BASE_DIR, 'NashMind')
if NM_DIR not in sys.path:
    sys.path.insert(0, NM_DIR)

try:
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
except Exception:
    pass
from NashMind.aces_system import ACES  # type: ignore
from evaluation import evaluate_tasks, _extract_ground_truth  # type: ignore


def run_on_tasks(tasks_path: str, solutions_path: str, limit: int = 50) -> Dict[str, Any]:
    with open(tasks_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    with open(solutions_path, 'r', encoding='utf-8') as f:
        solutions = json.load(f)

    items = list(tasks.items())[:limit]
    subset = {k: v for k, v in items}

    agent = ACES()

    results: Dict[str, Any] = {}
    solved = 0
    failed = 0
    for task_id, task_data in subset.items():
        # build training examples for ACES
        train_pairs = task_data.get('train', [])
        training_examples = [{'input': np.array(p['input']), 'output': np.array(p['output'])} for p in train_pairs]
        test_input = np.array(task_data['test'][0]['input'])

        try:
            sol = agent.solve_arc_problem(training_examples, test_input)
        except Exception as e:
            results[task_id] = {'status': f'error: {e}'}
            continue

        # normalize prediction to ndarray
        pred = np.array(sol) if not isinstance(sol, np.ndarray) else sol
        gt = _extract_ground_truth(task_id, solutions, tasks)
        if gt is None:
            results[task_id] = {'status': 'no_ground_truth'}
            continue
        exact = int(pred.shape == gt.shape and np.array_equal(pred, gt))
        sim = float(np.mean(pred == gt)) if pred.shape == gt.shape else 0.0
        solved += exact
        failed += int(exact == 0)
        results[task_id] = {'status': 'ok', 'exact': exact, 'similarity': sim}

    overall = {
        'total': len(subset),
        'solved': solved,
        'failed': failed,
    }
    return {'overall': overall, 'by_task': results}


if __name__ == '__main__':
    tasks_file = os.path.join(BASE_DIR, 'arc-agi_training_challenges.json')
    solutions_file = os.path.join(BASE_DIR, 'arc-agi_training_solutions.json')
    report = run_on_tasks(tasks_file, solutions_file, limit=50)
    print('\n=== NashMind on 50 training tasks ===')
    print('Total:', report['overall']['total'])
    print('Solved:', report['overall']['solved'])
    print('Failed:', report['overall']['failed'])
    failed_ids = [tid for tid, r in report['by_task'].items() if r.get('status')=='ok' and r.get('exact')==0]
    if failed_ids:
        print('Failed IDs (up to 30):', ', '.join(failed_ids[:30]))
