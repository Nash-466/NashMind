"""
ARC Task Loader Module
Handles loading and parsing ARC tasks from JSON files
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ARCExample:
    """Represents a single input/output example"""
    input: np.ndarray
    output: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Convert lists to numpy arrays if needed"""
        if not isinstance(self.input, np.ndarray):
            self.input = np.array(self.input, dtype=np.int8)
        if self.output is not None and not isinstance(self.output, np.ndarray):
            self.output = np.array(self.output, dtype=np.int8)
    
    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.input.shape
    
    @property
    def output_shape(self) -> Optional[Tuple[int, int]]:
        return self.output.shape if self.output is not None else None


@dataclass
class ARCTask:
    """Represents a complete ARC task with train and test examples"""
    task_id: str
    train_examples: List[ARCExample]
    test_examples: List[ARCExample]
    
    @property
    def num_train(self) -> int:
        return len(self.train_examples)
    
    @property
    def num_test(self) -> int:
        return len(self.test_examples)
    
    def get_train_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a specific training input/output pair"""
        if index >= self.num_train:
            raise IndexError(f"Train index {index} out of range (max: {self.num_train - 1})")
        example = self.train_examples[index]
        return example.input, example.output
    
    def get_test_input(self, index: int) -> np.ndarray:
        """Get a specific test input"""
        if index >= self.num_test:
            raise IndexError(f"Test index {index} out of range (max: {self.num_test - 1})")
        return self.test_examples[index].input
    
    def validate(self) -> bool:
        """Validate task data integrity"""
        if not self.task_id:
            return False
        
        if not self.train_examples:
            return False
        
        # Check all train examples have outputs
        for example in self.train_examples:
            if example.output is None:
                return False
            
            # Check valid color range (0-9)
            if not np.all((example.input >= 0) & (example.input <= 9)):
                return False
            if not np.all((example.output >= 0) & (example.output <= 9)):
                return False
        
        # Check test examples have valid inputs
        for example in self.test_examples:
            if not np.all((example.input >= 0) & (example.input <= 9)):
                return False
        
        return True


class TaskLoader:
    """Loads and manages ARC tasks from various sources"""
    
    def __init__(self):
        self.tasks: Dict[str, ARCTask] = {}
        self.solutions: Dict[str, List[np.ndarray]] = {}
    
    def load_from_json_file(self, filepath: str, is_solution: bool = False) -> Dict[str, ARCTask]:
        """Load tasks from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if is_solution:
            return self._parse_solutions(data)
        else:
            return self._parse_tasks(data)
    
    def load_from_json_string(self, json_str: str, is_solution: bool = False) -> Dict[str, ARCTask]:
        """Load tasks from a JSON string"""
        data = json.loads(json_str)
        
        if is_solution:
            return self._parse_solutions(data)
        else:
            return self._parse_tasks(data)
    
    def _parse_tasks(self, data: Dict) -> Dict[str, ARCTask]:
        """Parse task data from JSON format"""
        tasks = {}
        
        for task_id, task_data in data.items():
            train_examples = []
            test_examples = []
            
            # Parse training examples
            if 'train' in task_data:
                for example in task_data['train']:
                    train_examples.append(ARCExample(
                        input=example['input'],
                        output=example.get('output')
                    ))
            
            # Parse test examples
            if 'test' in task_data:
                for example in task_data['test']:
                    test_examples.append(ARCExample(
                        input=example['input'],
                        output=example.get('output')
                    ))
            
            task = ARCTask(
                task_id=task_id,
                train_examples=train_examples,
                test_examples=test_examples
            )
            
            if task.validate():
                tasks[task_id] = task
                self.tasks[task_id] = task
        
        return tasks
    
    def _parse_solutions(self, data: Dict) -> Dict:
        """Parse solution data from JSON format"""
        for task_id, solutions in data.items():
            self.solutions[task_id] = [np.array(sol, dtype=np.int8) for sol in solutions]
        return self.solutions
    
    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Get a specific task by ID"""
        return self.tasks.get(task_id)
    
    def get_solution(self, task_id: str) -> Optional[List[np.ndarray]]:
        """Get solutions for a specific task"""
        return self.solutions.get(task_id)
    
    def list_tasks(self) -> List[str]:
        """List all available task IDs"""
        return list(self.tasks.keys())
    
    def load_directory(self, directory: str) -> int:
        """Load all JSON files from a directory"""
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                is_solution = 'solution' in filename.lower()
                self.load_from_json_file(filepath, is_solution)
                count += 1
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded tasks"""
        stats = {
            'total_tasks': len(self.tasks),
            'total_train_examples': sum(task.num_train for task in self.tasks.values()),
            'total_test_examples': sum(task.num_test for task in self.tasks.values()),
            'tasks_with_solutions': len(self.solutions),
            'avg_train_per_task': 0,
            'avg_test_per_task': 0,
            'grid_sizes': {}
        }
        
        if self.tasks:
            stats['avg_train_per_task'] = stats['total_train_examples'] / stats['total_tasks']
            stats['avg_test_per_task'] = stats['total_test_examples'] / stats['total_tasks']
            
            # Collect grid size statistics
            for task in self.tasks.values():
                for example in task.train_examples:
                    size_key = f"{example.input_shape[0]}x{example.input_shape[1]}"
                    stats['grid_sizes'][size_key] = stats['grid_sizes'].get(size_key, 0) + 1
        
        return stats
    
    def validate_all(self) -> Tuple[List[str], List[str]]:
        """Validate all loaded tasks and return valid and invalid task IDs"""
        valid = []
        invalid = []
        
        for task_id, task in self.tasks.items():
            if task.validate():
                valid.append(task_id)
            else:
                invalid.append(task_id)
        
        return valid, invalid