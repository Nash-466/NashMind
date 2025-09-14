import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

import numpy as np

# Import error management system
from error_manager import error_manager, safe_execute, ErrorSeverity

def _self_heal_callable_issue():
    """Ensure arc_ultimate_mind_part7.py is resilient to Callable import issues.
    Injects `from __future__ import annotations` and `from collections.abc import Callable` if missing,
    then retries the import once.
    """
    target = os.path.join(os.path.dirname(__file__), 'arc_ultimate_mind_part7.py')
    if not os.path.exists(target):
        return False
    try:
        with open(target, 'r', encoding='utf-8') as f:
            src = f.read()
        changed = False
        if 'from __future__ import annotations' not in src:
            src = 'from __future__ import annotations\n' + src
            changed = True
        if 'from collections.abc import Callable' not in src:
            if 'from typing import' in src:
                src = src.replace('from typing import', 'from collections.abc import Callable\nfrom typing import', 1)
                changed = True
            else:
                # Fallback: add after logging import
                src = src.replace('import logging', 'import logging\nfrom collections.abc import Callable', 1)
                changed = True
        if changed:
            with open(target, 'w', encoding='utf-8') as f:
                f.write(src)
        return changed
    except Exception:
        return False

def _self_heal_repo_callable():
    """Scan local .py files and ensure future annotations + Callable import are present.
    This increases resilience on Kaggle/varied Python versions without manual edits.
    """
    base_dir = os.path.dirname(__file__)
    changed_any = False
    try:
        for name in os.listdir(base_dir):
            if not name.endswith('.py'):
                continue
            path = os.path.join(base_dir, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    src = f.read()
                changed = False
                if 'from __future__ import annotations' not in src:
                    src = 'from __future__ import annotations\n' + src
                    changed = True
                # Add Callable import only if file references Callable or typing import exists
                if ('Callable' in src or 'from typing import' in src) and 'from collections.abc import Callable' not in src:
                    if 'from typing import' in src:
                        src = src.replace('from typing import', 'from collections.abc import Callable\nfrom typing import', 1)
                        changed = True
                    else:
                        src = src.replace('import logging', 'import logging\nfrom collections.abc import Callable', 1)
                        changed = True
                if changed:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(src)
                    changed_any = True
            except Exception:
                continue
    except Exception:
        return False
    return changed_any

# Proactive self-heal to maximize portability across environments
_self_heal_repo_callable()

try:
    from arc_ultimate_mind_part7 import MasterOrchestrator
except Exception as e:
    # One-time self-heal for Callable NameError in older Python/type configurations
    if 'Callable' in str(e):
        if _self_heal_callable_issue():
            try:
                from arc_ultimate_mind_part7 import MasterOrchestrator
            except Exception as e2:
                print(f"Error importing MasterOrchestrator after self-heal: {e2}")
                sys.exit(1)
        else:
            print(f"Error importing MasterOrchestrator (no self-heal): {e}")
            sys.exit(1)
    else:
        print(f"Error importing MasterOrchestrator: {e}")
        sys.exit(1)


@safe_execute("main", "load_tasks", ErrorSeverity.CRITICAL, {})
def load_tasks(filepath: str) -> Dict[str, Any]:
    """Load tasks from JSON file with error handling."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


@safe_execute("main", "save_submission", ErrorSeverity.HIGH)
def save_submission(submission: Dict[str, Any], filepath: str) -> None:
    """Save submission to JSON file with error handling."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(submission, f, indent=4)
    logging.info(f"Submission successfully saved to {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Burhan ARC Orchestrator: unified entry point')

    parser.add_argument('-t', '--tasks', required=True,
                        help='Path to tasks JSON file')
    parser.add_argument('-o', '--output', default='submission.json',
                        help='Path to save submission JSON (default: submission.json)')
    parser.add_argument('--mode', choices=['fast', 'deep'], default='fast',
                        help='Execution mode: fast (quick heuristics) or deep (enable heavy search)')
    parser.add_argument('--task-id', default=None,
                        help='Run a single task by id for debugging')
    parser.add_argument('--optimize-every', type=int, default=0,
                        help='Trigger self-optimization every N tasks via performance window')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--meta', action='store_true', help='Enable MetaBrain layer for hypothesis market and compositions')
    parser.add_argument('--kb-path', default='_kb/meta_kb.json', help='Path to MetaBrain knowledge base file')
    parser.add_argument('--kaggle-format', action='store_true',
                        help='Output submission with both attempt_1 and attempt_2 keys per task')
    parser.add_argument('--kaggle-data', default=None,
                        help='Path to Kaggle ARC directory (e.g., /kaggle/input/arc-prize-2025) to load tasks directly')
    parser.add_argument('--kaggle-split', default='test', choices=['test', 'eval', 'train'],
                        help='Which Kaggle split to load when --kaggle-data is provided')
    parser.add_argument('--enable-plugins', action='store_true',
                        help='Dynamically import local .py modules and register Theory subclasses found')
    parser.add_argument('--smoke-import', action='store_true',
                        help='Import all local .py modules and print OK/FAILED for each (diagnostics)')
    parser.add_argument('--evaluate', action='store_true',
                        help='If provided with --kaggle-data and split eval/train, computes personal evaluation report')
    parser.add_argument('--eval-report', default='evaluation_report.json',
                        help='Path to save evaluation report JSON when --evaluate is set')
    return parser.parse_args()


def build_orchestrator(mode: str, optimize_every: int) -> MasterOrchestrator:
    # Configure orchestrator and solver behavior from CLI
    solver_cfg = {
        'max_time_per_task': 8 if mode == 'fast' else 60,
        'enable_adaptive_learning': mode == 'deep',
        'enable_meta_reasoning': True,
        'enable_solution_caching': True,
    }
    orch_cfg = {
        'solver_config': solver_cfg,
    }
    if optimize_every and optimize_every > 0:
        # Use performance_window as the cadence for optimization checks
        orch_cfg['performance_window'] = max(1, int(optimize_every))

    return MasterOrchestrator(orch_cfg)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.kaggle_data:
        try:
            from kaggle_io import load_arc_tasks_from_dir
            tasks = load_arc_tasks_from_dir(args.kaggle_data, split=args.kaggle_split)
            logging.info(f"Loaded tasks from Kaggle dir {args.kaggle_data} split={args.kaggle_split}")
        except Exception as e:
            logging.error(f"Failed to load Kaggle data: {e}")
            sys.exit(1)
    else:
        tasks = load_tasks(args.tasks)
    orchestrator = build_orchestrator(args.mode, args.optimize_every)
    meta_brain = None
    if args.meta:
        try:
            from burhan_meta_brain import MetaBrain
            meta_brain = MetaBrain(kb_path=args.kb_path)
            logging.info('MetaBrain enabled')
        except Exception as e:
            logging.warning(f"Failed to initialize MetaBrain: {e}")

    # Process tasks
    submission_dict: Dict[str, Any] = {}

    items = list(tasks.items())
    if args.task_id is not None:
        # Only run a single task if present
        items = [(args.task_id, tasks.get(args.task_id))] if args.task_id in tasks else []
        if not items:
            logging.error(f"Task id {args.task_id} not found in {args.tasks}")
            sys.exit(1)

    # Optional plugin loading and smoke import
    if args.enable_plugins or args.smoke_import:
        _load_plugins_and_smoke(orchestrator, smoke=args.smoke_import)

    for task_id, task_data in items:
        logging.info("--- Processing Task: %s (mode=%s) ---" % (task_id, args.mode))
        grid_list = None
        try:
            if meta_brain is not None:
                prediction = meta_brain.suggest_and_solve(task_data, orchestrator, mode=args.mode)
            else:
                prediction = orchestrator.process_single_task(task_data)
            if prediction is not None:
                grid_list = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
        except Exception as e:
            logging.exception(f"Error while processing task {task_id}: {e}")

        if grid_list is None:
            logging.warning(f"No solution found for task {task_id}. Submitting default placeholder.")
            try:
                test_shape = np.array(task_data['test'][0]['input']).shape
                grid_list = [[0 for _ in range(test_shape[1])] for __ in range(test_shape[0])]
            except Exception:
                grid_list = [[0]]

        if args.kaggle_format:
            submission_dict[task_id] = [{
                'attempt_1': grid_list,
                'attempt_2': grid_list
            }]
        else:
            submission_dict[task_id] = [{'attempt_1': grid_list}]

    save_submission(submission_dict, args.output)

    # Optional personal evaluation
    if args.evaluate and args.kaggle_data and args.kaggle_split in ('eval', 'train'):
        try:
            from kaggle_io import load_arc_solutions_from_dir
            from evaluation import evaluate_tasks
            solutions = load_arc_solutions_from_dir(args.kaggle_data, split=args.kaggle_split)
            report = evaluate_tasks(tasks, solutions, orchestrator, meta_brain=meta_brain, mode=args.mode)
            with open(args.eval_report, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            overall = report.get('overall', {})
            by_task = report.get('by_task', {})
            exact_acc = overall.get('exact_match_accuracy', 0.0)
            total_eval = overall.get('total_evaluated', 0)
            failed_ids = [tid for tid, res in by_task.items() if res.get('status') == 'ok' and res.get('exact') == 0]
            solved_ids = [tid for tid, res in by_task.items() if res.get('status') == 'ok' and res.get('exact') == 1]

            #      
            print("\n=== Evaluation Summary ===")
            print(f"Total evaluated: {total_eval}")
            print(f"Solved (exact): {len(solved_ids)}")
            print(f"Failed (exact): {len(failed_ids)}")
            if failed_ids:
                print("Failed task IDs:")
                print(", ".join(failed_ids))
            print(f"Exact accuracy: {exact_acc:.4f}")
            print("Saved report to:", args.eval_report)
            logging.info(f"Saved evaluation report to {args.eval_report}. Exact accuracy: {exact_acc:.4f}")
        except Exception as e:
            logging.error(f"Failed to run evaluation: {e}")

def _load_plugins_and_smoke(orchestrator: MasterOrchestrator, smoke: bool = False) -> None:
    import os, sys, importlib, inspect
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    skip = {os.path.basename(__file__).replace('.py',''), 'kaggle_io', 'evaluation', 'burhan_meta_brain'}
    try:
        from arc_ultimate_mind_part7 import Theory
    except Exception:
        Theory = None  # type: ignore
    for fname in os.listdir(base_dir):
        if not fname.endswith('.py'):
            continue
        modname = fname[:-3]
        if modname.startswith('_') or modname in skip:
            continue
        try:
            mod = importlib.import_module(modname)
            if smoke:
                print('[OK] import ' + modname)
        except Exception as ie:
            if smoke:
                print('[FAILED] import ' + modname + ': ' + str(ie))
            continue
        if Theory is None:
            continue
        try:
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if cls.__module__ != mod.__name__:
                    continue
                try:
                    is_sub = issubclass(cls, Theory)
                except Exception:
                    is_sub = False
                if (not is_sub) or cls is Theory:
                    continue
                if any(t.__class__.__name__ == cls.__name__ for t in getattr(orchestrator, 'theory_library', [])):
                    continue
                try:
                    inst = cls(getattr(orchestrator, 'pattern_analyzer', None),
                               getattr(orchestrator, 'strategy_manager', None))
                    orchestrator.theory_library.append(inst)
                    if smoke:
                        print('[PLUGIN] Registered theory: ' + cls.__name__)
                except Exception as ce:
                    if smoke:
                        print('[PLUGIN-ERR] ' + cls.__name__ + ': ' + str(ce))
                    continue
        except Exception:
            continue


if __name__ == '__main__':
    main()

