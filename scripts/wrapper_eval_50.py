from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import sys

# Ensure project root on sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from unified_solver_wrapper import UnifiedSolverWrapper
import argparse
import io
import contextlib
import logging


def load_tasks(challenges_path: Path, solutions_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    solutions = json.loads(solutions_path.read_text(encoding="utf-8"))
    tasks: List[Dict[str, Any]] = []
    for task_id in list(challenges.keys())[:limit]:
        if task_id not in solutions:
            continue
        task = challenges[task_id]
        task["id"] = task_id
        task["expected"] = solutions[task_id]
        tasks.append(task)
    return tasks


def evaluate_solution(pred: np.ndarray, expected: List[List[List[int]]]) -> Tuple[bool, float]:
    # expected may be a list of outputs (one per test case). We check first one
    exp = np.array(expected[0]) if expected else None
    if exp is None or not isinstance(pred, np.ndarray):
        return False, 0.0
    if pred.shape != exp.shape:
        return False, 0.0
    exact = bool(np.array_equal(pred, exp))
    sim = float(np.mean(pred == exp))
    return exact, sim


def run_eval(challenges: str, solutions: str, out_name: str, limit: int) -> Dict[str, Any]:
    tasks = load_tasks(Path(challenges), Path(solutions), limit)
    w = UnifiedSolverWrapper()
    systems = w.systems

    results: Dict[str, Any] = {
        "meta": {
            "challenges": challenges,
            "solutions": solutions,
            "num_tasks": len(tasks),
            "num_systems": len(systems),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "systems": {}
    }

    for sysdef in systems:
        name = sysdef["name"]
        solve = sysdef["solve"]
        stats = {
            "total": 0,
            "exact": 0,
            "avg_similarity": 0.0,
            "fail": 0,
            "times": []  # not measuring per-task time precisely to keep simple
        }
        sims: List[float] = []

        for task in tasks:
            sub_task = {
                "train": task.get("train", []),
                "test": [{"input": task["test"][0]["input"]}] if task.get("test") else []
            }
            try:
                # Suppress verbose prints/logs during solve
                logging.getLogger().setLevel(logging.ERROR)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    pred = solve(sub_task)
                if isinstance(pred, list) and pred:
                    pred = pred[0]
                exact, sim = evaluate_solution(pred, task.get("expected", []))
                stats["total"] += 1
                stats["exact"] += int(exact)
                sims.append(sim)
            except Exception as e:
                stats["total"] += 1
                stats["fail"] += 1
                sims.append(0.0)

        stats["avg_similarity"] = float(np.mean(sims)) if sims else 0.0
        results["systems"][name] = stats

    outdir = Path("reports")
    outdir.mkdir(parents=True, exist_ok=True)
    Path(outdir / out_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate all systems via UnifiedSolverWrapper on ARC datasets.")
    parser.add_argument("--limit", type=int, default=50, help="Number of tasks to evaluate (default: 50)")
    args = parser.parse_args()

    limit = args.limit

    # Training set
    tr_results = run_eval("arc-agi_training_challenges.json", "arc-agi_training_solutions.json", f"wrapper_eval_{limit}_training.json", limit)
    print("Training summary (top-5 by exact):")
    ranked_tr = sorted(tr_results["systems"].items(), key=lambda x: (x[1]["exact"], x[1]["avg_similarity"]), reverse=True)
    for name, s in ranked_tr[:5]:
        print(f"- {name}: exact={s['exact']}/{s['total']} sim={s['avg_similarity']:.1%}")

    # Evaluation set
    ev_results = run_eval("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json", f"wrapper_eval_{limit}_evaluation.json", limit)
    print("\nEvaluation summary (top-5 by exact):")
    ranked_ev = sorted(ev_results["systems"].items(), key=lambda x: (x[1]["exact"], x[1]["avg_similarity"]), reverse=True)
    for name, s in ranked_ev[:5]:
        print(f"- {name}: exact={s['exact']}/{s['total']} sim={s['avg_similarity']:.1%}")


if __name__ == "__main__":
    main()
