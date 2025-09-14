from __future__ import annotations
import numpy as np
import json
import os
import logging
from arc_ultimate_system import RevolutionaryOrchestrator

# ---       ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def load_arc_task(task_path: str) -> dict:
    """  ARC   JSON."""
    try:
        with open(task_path, 'r', encoding='utf-8') as f:
            task = json.load(f)
        logger.info(f"Task loaded successfully from {task_path}")
        return task
    except FileNotFoundError:
        logger.error(f"Task file not found: {task_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {task_path}: {e}")
        return None

def main():
    logger.info("Starting ARC Ultimate Revolutionary System Main Execution.")

    #   
    orchestrator = RevolutionaryOrchestrator()

    #    ARC (    )
    #       .
    #          ARC.
    sample_task_1 = {
        "id": "sample_task_001",
        "train": [
            {"input": [[1,1,1],[0,1,0],[0,1,0]], "output": [[5,5,5],[0,5,0],[0,5,0]]},
            {"input": [[2,2],[2,0]], "output": [[6,6],[6,0]]}
        ],
        "test": [
            {"input": [[1,1,1],[0,1,0],[0,1,0]], "output": [[5,5,5],[0,5,0],[0,5,0]]}
        ]
    }

    sample_task_2 = {
        "id": "sample_task_002",
        "train": [
            {"input": [[1,2,3],[4,5,6]], "output": [[1,1,2,2,3,3],[4,4,5,5,6,6]]}
        ],
        "test": [
            {"input": [[7,8],[9,0]], "output": [[7,7,8,8],[9,9,0,0]]}
        ]
    }

    #  
    print("\n" + "="*80)
    print("🚀 Processing Sample Task 1")
    print("="*80)
    result_1 = orchestrator.process_task_holistically(sample_task_1)
    print(f"Task 1 Result: Score={result_1.get('validation_score', 0.0):.2f}, Time={result_1.get('total_time', 0.0):.2f}s")
    print(f"Predicted Output for Task 1 Test Case: {result_1.get(\'predicted_outputs\', \'N/A\')}")
    print(f"Solution Strategy Used: {result_1.get('solution_strategy', 'N/A')}")
    if "error" in result_1: logger.error(f"Error processing Task 1: {result_1.get("error", "Unknown Error")}")

    print("\n" + "="*80)
    print("🚀 Processing Sample Task 2")
    print("="*80)
    result_2 = orchestrator.process_task_holistically(sample_task_2)
    print(f"Task 2 Result: Score={result_2.get('validation_score', 0.0):.2f}, Time={result_2.get('total_time', 0.0):.2f}s")
  print(f"Predicted Output for Task 2 Test Case: {result_2.get(\'predicted_outputs\', \'N/A\')}")   print(f"Solution Strategy Used: {result_2.get('solution_strategy', 'N/A')}")
    if "error" in result_2: logger.error(f"Error processing Task 2: {result_2.get("error", "Unknown Error")}")

    print("\n" + "="*80)
    print("🎉 ARC Ultimate Revolutionary System Execution Completed!")
    print("="*80)

    #      
    # :    
    # arc_tasks_dir = "/path/to/your/arc_tasks"
    # for filename in os.listdir(arc_tasks_dir):
    #     if filename.endswith(".json"):
    #         task_path = os.path.join(arc_tasks_dir, filename)
    #         task = load_arc_task(task_path)
    #         if task:
    #             print(f"\nProcessing task from file: {filename}")
    #             orchestrator.process_task_holistically(task)

if __name__ == "__main__":
    main()



