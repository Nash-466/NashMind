from __future__ import annotations
import json
from collections.abc import Callable
from typing import Dict, Any, Optional, Tuple

import numpy as np


def _extract_ground_truth(task_id: str, solutions: Dict[str, Any], tasks: Dict[str, Any]) -> Optional[np.ndarray]:
    """Try to extract the expected test output grid for a task.

    Handles a few common shapes of solutions JSON; returns first test output.
    """
    if task_id not in solutions:
        # If running on 'train' split without a solutions file, some datasets include outputs inside tasks
        try:
            test_list = tasks[task_id].get('test', [])
            if test_list and 'output' in test_list[0]:
                return np.array(test_list[0]['output'])
        except Exception:
            return None
        return None

    entry = solutions[task_id]
    try:
        # Case 1: entry is list of dicts with 'output'
        if isinstance(entry, list):
            if not entry:
                return None
            first = entry[0]
            if isinstance(first, dict) and 'output' in first:
                return np.array(first['output'])
            # Case 2: entry is list of 2D list grid(s)
            if isinstance(first, list):
                # Treat the first grid as the GT
                return np.array(first)
        # Case 3: dict with 'output' or 'attempt_1'
        if isinstance(entry, dict):
            if 'output' in entry:
                return np.array(entry['output'])
            if 'attempt_1' in entry:
                return np.array(entry['attempt_1'])
    except Exception:
        return None
    return None


def evaluate_tasks(tasks: Dict[str, Any], solutions: Dict[str, Any], predictor, meta_brain=None, mode: str = 'fast') -> Dict[str, Any]:
    """Evaluates predictions against provided solutions.

    predictor: object with process_single_task(task) -> np.ndarray
    meta_brain: optional MetaBrain with suggest_and_solve(task, predictor, mode)
    Returns a report dict with per-task and aggregate metrics.
    """
    results = {}
    exact_matches = 0
    similarities = []

    for task_id, task_data in tasks.items():
        gt = _extract_ground_truth(task_id, solutions, tasks)
        if gt is None:
            results[task_id] = {'status': 'no_ground_truth'}
            continue
        try:
            if meta_brain is not None:
                pred = meta_brain.suggest_and_solve(task_data, predictor, mode=mode)
            else:
                pred = predictor.process_single_task(task_data)
        except Exception as e:
            results[task_id] = {'status': f'predict_error: {e}'}
            continue

        if pred is None:
            results[task_id] = {'status': 'no_prediction', 'exact': 0, 'similarity': 0.0}
            continue

        pred_arr = np.array(pred)
        if pred_arr.shape != gt.shape:
            exact = 0
            sim = 0.0
        else:
            exact = int(np.array_equal(pred_arr, gt))
            sim = float(np.mean(pred_arr == gt))

        exact_matches += exact
        similarities.append(sim)
        results[task_id] = {
            'status': 'ok',
            'exact': exact,
            'similarity': sim,
            'pred_shape': list(pred_arr.shape),
            'gt_shape': list(gt.shape),
        }

    total = len([k for k in tasks.keys() if k in solutions or tasks[k].get('test', [{}])[0].get('output') is not None])
    overall = {
        'total_evaluated': total,
        'exact_match_accuracy': (exact_matches / total) if total else 0.0,
        'mean_similarity': (sum(similarities) / len(similarities)) if similarities else 0.0,
    }
    return {'overall': overall, 'by_task': results}

