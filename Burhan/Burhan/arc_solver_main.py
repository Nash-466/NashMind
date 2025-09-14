#!/usr/bin/env python3
"""
ARC Solver Main Entry Point
Command-line interface for solving ARC Prize 2025 tasks
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.arc_agent import ARCAgent, AgentMode
from src.arc.task_loader import TaskLoader, ARCTask
from src.utils.evaluator import SolutionEvaluator
from src.arc.grid_operations import Grid


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_grid(grid: np.ndarray, title: str = ""):
    """Pretty print a grid"""
    if title:
        print(f"\n{title}")
    
    # Use colors for better visualization
    color_map = {
        0: '⬜',  # White (background)
        1: '🟦',  # Blue
        2: '🟥',  # Red
        3: '🟩',  # Green
        4: '🟨',  # Yellow
        5: '⬜',  # Gray (using white)
        6: '🟪',  # Purple
        7: '🟧',  # Orange
        8: '🟫',  # Brown
        9: '⬛'   # Black
    }
    
    for row in grid:
        row_str = ' '.join(color_map.get(int(cell), str(cell)) for cell in row)
        print(row_str)
    
    print(f"Shape: {grid.shape}")


def solve_single_task(agent: ARCAgent, 
                     task: ARCTask, 
                     evaluator: SolutionEvaluator,
                     verbose: bool = False) -> Dict:
    """
    Solve a single ARC task
    
    Args:
        agent: ARC agent instance
        task: Task to solve
        evaluator: Solution evaluator
        verbose: Print detailed output
        
    Returns:
        Results dictionary
    """
    
    print(f"\n{'='*70}")
    print(f"Solving Task: {task.task_id}")
    print(f"{'='*70}")
    
    if verbose:
        print(f"Training examples: {task.num_train}")
        print(f"Test examples: {task.num_test}")
        
        # Show first training example
        if task.num_train > 0:
            train_input, train_output = task.get_train_pair(0)
            print_grid(train_input, "First Training Input:")
            print_grid(train_output, "First Training Output:")
    
    # Solve the task
    start_time = time.time()
    solution = agent.solve_task(task)
    solve_time = time.time() - start_time
    
    print(f"\nSolution found in {solve_time:.2f} seconds")
    print(f"Strategy used: {solution.strategy_used}")
    print(f"Confidence: {solution.confidence:.2%}")
    
    # Evaluate if ground truth available
    results = {
        'task_id': task.task_id,
        'solved': len(solution.test_solutions),
        'time': solve_time,
        'confidence': solution.confidence,
        'strategy': solution.strategy_used
    }
    
    if any(ex.output is not None for ex in task.test_examples):
        # We have ground truth
        correct = 0
        for i, (test_ex, pred) in enumerate(zip(task.test_examples, solution.test_solutions)):
            if test_ex.output is not None and pred is not None:
                metrics = evaluator.evaluate_solution(
                    pred, test_ex.output,
                    task_id=f"{task.task_id}_test_{i}",
                    solver_used=solution.strategy_used,
                    execution_time=solve_time,
                    confidence=solution.confidence
                )
                
                if metrics.is_correct:
                    correct += 1
                    print(f"✓ Test {i+1}: CORRECT")
                else:
                    print(f"✗ Test {i+1}: INCORRECT (Pixel accuracy: {metrics.pixel_accuracy:.2%})")
                
                if verbose:
                    print_grid(pred, f"Test {i+1} Prediction:")
                    if test_ex.output is not None:
                        print_grid(test_ex.output, f"Test {i+1} Ground Truth:")
        
        accuracy = correct / len(task.test_examples)
        results['correct'] = correct
        results['total'] = len(task.test_examples)
        results['accuracy'] = accuracy
        
        print(f"\nAccuracy: {correct}/{len(task.test_examples)} ({accuracy:.2%})")
    else:
        print("No ground truth available for evaluation")
        
        if verbose:
            # Show predictions
            for i, pred in enumerate(solution.test_solutions):
                if pred is not None:
                    print_grid(pred, f"Test {i+1} Prediction:")
    
    return results


def solve_batch(agent: ARCAgent,
               tasks: List[ARCTask],
               evaluator: SolutionEvaluator,
               verbose: bool = False,
               max_tasks: Optional[int] = None) -> Dict:
    """
    Solve multiple tasks
    
    Args:
        agent: ARC agent instance
        tasks: List of tasks to solve
        evaluator: Solution evaluator
        verbose: Print detailed output
        max_tasks: Maximum number of tasks to solve
        
    Returns:
        Batch results
    """
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    print(f"\n{'='*70}")
    print(f"Solving {len(tasks)} tasks...")
    print(f"{'='*70}")
    
    all_results = []
    total_correct = 0
    total_examples = 0
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] ", end='')
        
        try:
            results = solve_single_task(agent, task, evaluator, verbose=False)
            all_results.append(results)
            
            if 'correct' in results:
                total_correct += results['correct']
                total_examples += results['total']
        
        except Exception as e:
            print(f"Error solving {task.task_id}: {e}")
            all_results.append({
                'task_id': task.task_id,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in all_results if 'error' not in r]
    print(f"Tasks attempted: {len(tasks)}")
    print(f"Tasks completed: {len(successful)}")
    
    if total_examples > 0:
        print(f"Total test examples: {total_examples}")
        print(f"Correct predictions: {total_correct}")
        print(f"Overall accuracy: {total_correct/total_examples:.2%}")
    
    # Time statistics
    if successful:
        avg_time = np.mean([r['time'] for r in successful])
        print(f"Average solve time: {avg_time:.2f}s")
    
    # Strategy breakdown
    strategies = {}
    for r in successful:
        strat = r.get('strategy', 'unknown')
        strategies[strat] = strategies.get(strat, 0) + 1
    
    if strategies:
        print("\nStrategy usage:")
        for strat, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strat}: {count}")
    
    return {
        'results': all_results,
        'summary': {
            'total_tasks': len(tasks),
            'completed': len(successful),
            'total_correct': total_correct,
            'total_examples': total_examples,
            'accuracy': total_correct/total_examples if total_examples > 0 else 0
        }
    }


def save_solutions(solutions: Dict[str, List[List[int]]], output_path: str):
    """Save solutions to JSON file"""
    
    # Convert numpy arrays to lists
    json_solutions = {}
    for task_id, outputs in solutions.items():
        json_solutions[task_id] = [
            out.tolist() if isinstance(out, np.ndarray) else out
            for out in outputs
        ]
    
    with open(output_path, 'w') as f:
        json.dump(json_solutions, f, indent=2)
    
    print(f"Solutions saved to {output_path}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="ARC Prize 2025 Solver - Command Line Interface"
    )
    
    parser.add_argument(
        'command',
        choices=['solve', 'evaluate', 'demo'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--challenges',
        type=str,
        help='Path to challenges JSON file or directory'
    )
    
    parser.add_argument(
        '--solutions',
        type=str,
        help='Path to solutions JSON file (for evaluation)'
    )
    
    parser.add_argument(
        '--task-id',
        type=str,
        help='Specific task ID to solve'
    )
    
    parser.add_argument(
        '--mode',
        choices=['fast', 'balanced', 'comprehensive', 'adaptive'],
        default='balanced',
        help='Agent solving mode (default: balanced)'
    )
    
    parser.add_argument(
        '--max-tasks',
        type=int,
        help='Maximum number of tasks to solve'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for solutions'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize components
    print("Initializing ARC Agent...")
    mode = AgentMode[args.mode.upper()]
    agent = ARCAgent(mode=mode)
    evaluator = SolutionEvaluator()
    loader = TaskLoader()
    
    if args.command == 'demo':
        # Run demo with built-in example
        print("\n" + "="*70)
        print("ARC SOLVER DEMONSTRATION")
        print("="*70)
        
        # Try to load from attached files
        challenges_file = "attached_assets/arc-agi_training_challenges (2)_1757573784094.txt"
        solutions_file = "attached_assets/arc-agi_training_solutions_1757573784094.txt"
        
        if os.path.exists(challenges_file):
            print(f"\nLoading tasks from: {challenges_file}")
            tasks = loader.load_from_json_file(challenges_file)
            
            if os.path.exists(solutions_file):
                print(f"Loading solutions from: {solutions_file}")
                # Load solutions and merge with tasks
                with open(solutions_file, 'r') as f:
                    solutions_json = json.load(f)
                
                # Merge solutions into tasks
                for task_id, outputs in solutions_json.items():
                    if task_id in tasks:
                        for i, output in enumerate(outputs):
                            if i < len(tasks[task_id].test_examples):
                                tasks[task_id].test_examples[i].output = np.array(output, dtype=np.int8)
            
            # Select a few tasks for demo
            demo_task_ids = ['00576224', '0692e18c', '007bbfb7']  # Simple to moderate
            demo_tasks = []
            
            for task_id in demo_task_ids:
                if task_id in tasks:
                    demo_tasks.append(tasks[task_id])
            
            if demo_tasks:
                print(f"\nRunning demo on {len(demo_tasks)} tasks...")
                results = solve_batch(agent, demo_tasks, evaluator, verbose=args.verbose)
            else:
                print("Demo tasks not found in loaded data")
        else:
            print("Demo files not found. Please ensure training data is available.")
            
            # Create a simple demo task
            print("\nCreating synthetic demo task...")
            from src.arc.task_loader import ARCTask, ARCExample
            
            # Simple scaling task
            demo_task = ARCTask(
                task_id="demo_scale",
                train_examples=[
                    ARCExample(
                        input=np.array([[1, 2], [3, 4]]),
                        output=np.array([[1, 1, 2, 2], 
                                        [1, 1, 2, 2],
                                        [3, 3, 4, 4],
                                        [3, 3, 4, 4]])
                    )
                ],
                test_examples=[
                    ARCExample(
                        input=np.array([[5, 6], [7, 8]]),
                        output=np.array([[5, 5, 6, 6],
                                        [5, 5, 6, 6],
                                        [7, 7, 8, 8],
                                        [7, 7, 8, 8]])
                    )
                ]
            )
            
            results = solve_single_task(agent, demo_task, evaluator, verbose=True)
    
    elif args.command == 'solve':
        if not args.challenges:
            print("Error: --challenges required for solve command")
            sys.exit(1)
        
        # Load tasks
        print(f"Loading tasks from: {args.challenges}")
        tasks = loader.load_from_json_file(args.challenges)
        
        if args.solutions:
            print(f"Loading ground truth from: {args.solutions}")
            # Load solutions and merge with tasks
            with open(args.solutions, 'r') as f:
                solutions_json = json.load(f)
            
            # Merge solutions into tasks
            for task_id, outputs in solutions_json.items():
                if task_id in tasks:
                    for i, output in enumerate(outputs):
                        if i < len(tasks[task_id].test_examples):
                            tasks[task_id].test_examples[i].output = np.array(output, dtype=np.int8)
        
        # Filter by task ID if specified
        if args.task_id:
            if args.task_id in tasks:
                task = tasks[args.task_id]
                results = solve_single_task(agent, task, evaluator, verbose=args.verbose)
                
                # Save solution if requested
                if args.output:
                    solution = agent.solutions_cache.get(args.task_id)
                    if solution:
                        save_solutions(
                            {args.task_id: solution.test_solutions},
                            args.output
                        )
            else:
                print(f"Task {args.task_id} not found")
                sys.exit(1)
        else:
            # Solve all or limited tasks
            task_list = list(tasks.values())
            results = solve_batch(
                agent, task_list, evaluator,
                verbose=args.verbose,
                max_tasks=args.max_tasks
            )
            
            # Save solutions if requested
            if args.output:
                all_solutions = {}
                for task_id, solution in agent.solutions_cache.items():
                    all_solutions[task_id] = solution.test_solutions
                save_solutions(all_solutions, args.output)
    
    elif args.command == 'evaluate':
        if not args.challenges or not args.solutions:
            print("Error: Both --challenges and --solutions required for evaluate command")
            sys.exit(1)
        
        print(f"Loading tasks from: {args.challenges}")
        tasks = loader.load_from_json_file(args.challenges)
        
        print(f"Loading solutions from: {args.solutions}")
        solutions = {}
        with open(args.solutions, 'r') as f:
            solutions = json.load(f)
        
        # Evaluate solutions
        print("\nEvaluating solutions...")
        results = []
        
        for task_id, task in tasks.items():
            if task_id not in solutions:
                continue
            
            task_solutions = solutions[task_id]
            for i, (test_ex, pred) in enumerate(zip(task.test_examples, task_solutions)):
                if test_ex.output is not None:
                    pred_array = np.array(pred)
                    metrics = evaluator.evaluate_solution(
                        pred_array, test_ex.output,
                        task_id=f"{task_id}_test_{i}"
                    )
                    results.append(metrics)
                    
                    if args.verbose:
                        print(evaluator.generate_report(metrics))
        
        # Summary
        summary = evaluator.get_summary_stats()
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
        print(f"Correct: {summary.get('correct', 0)}")
        print(f"Accuracy: {summary.get('accuracy', 0):.2%}")
        print(f"Average pixel accuracy: {summary.get('average_pixel_accuracy', 0):.2%}")
    
    # Print final statistics
    print(f"\n{'='*70}")
    print("Agent Performance Statistics:")
    stats = agent.get_performance_stats()
    for key, value in stats.items():
        if key != 'solver_accuracy':
            print(f"  {key}: {value}")
    
    print("\nThank you for using ARC Solver!")


if __name__ == "__main__":
    main()