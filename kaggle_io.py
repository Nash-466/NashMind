from __future__ import annotations
import json
import os
from collections.abc import Callable
from typing import Dict, Any


TEST_FILE = "arc-agi_test_challenges.json"
EVAL_FILE = "arc-agi_evaluation_challenges.json"
TRAIN_FILE = "arc-agi_training_challenges.json"
EVAL_SOL_FILE = "arc-agi_evaluation_solutions.json"
TRAIN_SOL_FILE = "arc-agi_training_solutions.json"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_arc_tasks_from_dir(dir_path: str, split: str = 'test') -> Dict[str, Any]:
    """Reads ARC tasks JSON from Kaggle-style directory.

    - split: 'test'|'eval'|'train' chooses which challenges file to read.
    - Returns a tasks dict: {task_id: {train: [...], test: [...]}}
    """
    split = (split or 'test').lower()
    if split == 'test':
        fname = TEST_FILE
    elif split in ('eval', 'evaluation'):
        fname = EVAL_FILE
    elif split == 'train':
        fname = TRAIN_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    path = os.path.join(dir_path, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ARC file not found: {path}")
    data = _load_json(path)

    # Data is already in expected structure for our orchestrator
    return data


def load_arc_solutions_from_dir(dir_path: str, split: str = 'eval') -> Dict[str, Any]:
    """Reads ARC solutions JSON for evaluation or training splits.

    - split: 'eval'|'train'
    - Returns a solutions dict keyed by task_id.
    """
    split = (split or 'eval').lower()
    if split in ('eval', 'evaluation'):
        fname = EVAL_SOL_FILE
    elif split == 'train':
        fname = TRAIN_SOL_FILE
    else:
        raise ValueError("Solutions are available only for 'eval' or 'train' splits")

    path = os.path.join(dir_path, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ARC solutions file not found: {path}")
    return _load_json(path)


def save_submission(submission: Dict[str, Any], filepath: str) -> None:
    """Writes submission JSON compatible with Kaggle sample format.
    Expects values like: [{'attempt_1': grid, 'attempt_2': grid}] per task.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=2)
