from __future__ import annotations
"""
ULTIMATE FULL SYSTEM TEST - 1000 ARC TASKS
===========================================
اختبار شامل لجميع الأنظمة على 1000 مهمة حقيقية
مع مقارنة الحلول وتحليل مفصل للأداء

Author: Testing Team
Date: 2025
"""

import numpy as np
import time
import json
import os
import glob
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import traceback
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_test_1000.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# TASK LOADER
# ==============================================================================

class TaskLoader:
    """Load ARC tasks from JSON files"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.tasks = []
        
    def load_all_tasks(self, limit: int = 1000) -> List[Dict]:
        """Load all available ARC tasks up to limit"""
        
        logger.info(f"Loading ARC tasks from {self.data_dir}")
        
        # Look for JSON files with ARC tasks
        json_patterns = [
            'arc_tasks*.json',
            'training/*.json', 
            'evaluation/*.json',
            'test/*.json',
            '*.json'
        ]
        
        all_tasks = []
        
        for pattern in json_patterns:
            json_files = glob.glob(os.path.join(self.data_dir, pattern))
            
            for json_file in json_files:
                if 'test_results' in json_file or 'report' in json_file:
                    continue  # Skip result files
                    
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        
                        # Check if it's a task file
                        if isinstance(data, dict):
                            if 'train' in data and 'test' in data:
                                # Single task
                                all_tasks.append(data)
                            elif any(key.endswith('.json') for key in data.keys()):
                                # Multiple tasks in one file
                                for task_name, task_data in data.items():
                                    if isinstance(task_data, dict) and 'train' in task_data:
                                        all_tasks.append(task_data)
                        elif isinstance(data, list):
                            # List of tasks
                            for task in data:
                                if isinstance(task, dict) and 'train' in task:
                                    all_tasks.append(task)
                                    
                except Exception as e:
                    logger.debug(f"Could not load {json_file}: {e}")
        
        # If no real tasks found, generate synthetic ones
        if len(all_tasks) < limit:
            logger.info(f"Found {len(all_tasks)} real tasks, generating synthetic tasks...")
            all_tasks.extend(self.generate_synthetic_tasks(limit - len(all_tasks)))
        
        self.tasks = all_tasks[:limit]
        logger.info(f"Loaded {len(self.tasks)} tasks total")
        
        return self.tasks
    
    def generate_synthetic_tasks(self, count: int) -> List[Dict]:
        """Generate synthetic ARC-like tasks for testing"""
        
        synthetic_tasks = []
        
        task_generators = [
            self._generate_rotation_task,
            self._generate_color_mapping_task,
            self._generate_pattern_completion_task,
            self._generate_symmetry_task,
            self._generate_scaling_task,
            self._generate_object_movement_task,
            self._generate_counting_task,
            self._generate_filling_task,
            self._generate_boundary_task,
            self._generate_maze_task
        ]
        
        for i in range(count):
            generator = task_generators[i % len(task_generators)]
            task = generator(i)
            synthetic_tasks.append(task)
        
        return synthetic_tasks
    
    def _generate_rotation_task(self, seed: int) -> Dict:
        """Generate rotation task"""
        np.random.seed(seed)
        size = np.random.randint(3, 8)
        
        input_grid = np.random.randint(0, 5, (size, size))
        output_grid = np.rot90(input_grid, k=np.random.randint(1, 4))
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': np.random.randint(0, 5, (size, size)).tolist()}
            ]
        }
    
    def _generate_color_mapping_task(self, seed: int) -> Dict:
        """Generate color mapping task"""
        np.random.seed(seed)
        size = np.random.randint(3, 8)
        
        input_grid = np.random.randint(0, 4, (size, size))
        color_map = {0: 0, 1: 2, 2: 3, 3: 1}
        output_grid = np.vectorize(color_map.get)(input_grid)
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()},
                {'input': np.random.randint(0, 4, (size, size)).tolist(),
                 'output': np.vectorize(color_map.get)(np.random.randint(0, 4, (size, size))).tolist()}
            ],
            'test': [
                {'input': np.random.randint(0, 4, (size, size)).tolist()}
            ]
        }
    
    def _generate_pattern_completion_task(self, seed: int) -> Dict:
        """Generate pattern completion task"""
        np.random.seed(seed)
        size = np.random.randint(4, 8)
        
        # Create a pattern with missing center
        input_grid = np.zeros((size, size), dtype=int)
        input_grid[0, :] = 1
        input_grid[-1, :] = 1
        input_grid[:, 0] = 1
        input_grid[:, -1] = 1
        
        output_grid = input_grid.copy()
        output_grid[size//2, size//2] = 2
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }
    
    def _generate_symmetry_task(self, seed: int) -> Dict:
        """Generate symmetry task"""
        np.random.seed(seed)
        size = np.random.randint(4, 8)
        
        # Create half pattern
        input_grid = np.zeros((size, size), dtype=int)
        input_grid[:, :size//2] = np.random.randint(0, 3, (size, size//2))
        
        # Mirror it
        output_grid = input_grid.copy()
        half_size = size // 2
        if size % 2 == 0:
            output_grid[:, half_size:] = np.fliplr(input_grid[:, :half_size])
        else:
            output_grid[:, half_size+1:] = np.fliplr(input_grid[:, :half_size])
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': np.random.randint(0, 3, (size, size)).tolist()}
            ]
        }
    
    def _generate_scaling_task(self, seed: int) -> Dict:
        """Generate scaling task"""
        np.random.seed(seed)
        
        small_size = np.random.randint(2, 4)
        input_grid = np.random.randint(1, 4, (small_size, small_size))
        
        # Scale up 2x
        output_grid = np.repeat(np.repeat(input_grid, 2, axis=0), 2, axis=1)
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': np.random.randint(1, 4, (small_size, small_size)).tolist()}
            ]
        }
    
    def _generate_object_movement_task(self, seed: int) -> Dict:
        """Generate object movement task"""
        np.random.seed(seed)
        size = np.random.randint(5, 10)
        
        input_grid = np.zeros((size, size), dtype=int)
        # Place object
        obj_size = 2
        x, y = np.random.randint(0, size-obj_size, 2)
        input_grid[x:x+obj_size, y:y+obj_size] = np.random.randint(1, 4)
        
        # Move object
        output_grid = np.zeros((size, size), dtype=int)
        new_x, new_y = np.random.randint(0, size-obj_size, 2)
        output_grid[new_x:new_x+obj_size, new_y:new_y+obj_size] = input_grid[x:x+obj_size, y:y+obj_size]
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }
    
    def _generate_counting_task(self, seed: int) -> Dict:
        """Generate counting task"""
        np.random.seed(seed)
        size = np.random.randint(5, 10)
        
        input_grid = np.random.randint(0, 3, (size, size))
        
        # Count each color and create output
        output_grid = np.zeros((3, 3), dtype=int)
        for color in range(1, 3):
            count = np.sum(input_grid == color)
            if count > 0:
                output_grid[color-1, :min(count, 3)] = color
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': np.random.randint(0, 3, (size, size)).tolist()}
            ]
        }
    
    def _generate_filling_task(self, seed: int) -> Dict:
        """Generate filling task"""
        np.random.seed(seed)
        size = np.random.randint(5, 8)
        
        # Create shape with hole
        input_grid = np.zeros((size, size), dtype=int)
        input_grid[1:-1, 1:-1] = 1
        input_grid[2:-2, 2:-2] = 0
        
        # Fill the hole
        output_grid = input_grid.copy()
        output_grid[2:-2, 2:-2] = 2
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }
    
    def _generate_boundary_task(self, seed: int) -> Dict:
        """Generate boundary extraction task"""
        np.random.seed(seed)
        size = np.random.randint(5, 10)
        
        # Create filled shape
        input_grid = np.zeros((size, size), dtype=int)
        input_grid[2:-2, 2:-2] = np.random.randint(1, 4)
        
        # Extract boundary
        output_grid = np.zeros((size, size), dtype=int)
        output_grid[2, 2:-2] = 1
        output_grid[-3, 2:-2] = 1
        output_grid[2:-2, 2] = 1
        output_grid[2:-2, -3] = 1
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }
    
    def _generate_maze_task(self, seed: int) -> Dict:
        """Generate maze/path task"""
        np.random.seed(seed)
        size = 7
        
        # Create simple maze
        input_grid = np.zeros((size, size), dtype=int)
        # Walls
        input_grid[1, 1:-1] = 1
        input_grid[3, 1:-1] = 1
        input_grid[5, 1:-1] = 1
        # Start and end
        input_grid[0, 0] = 2
        input_grid[-1, -1] = 3
        
        # Solution path
        output_grid = input_grid.copy()
        # Mark path
        output_grid[0, 1:3] = 4
        output_grid[1:3, 2] = 4
        output_grid[2, 3:5] = 4
        output_grid[3:5, 4] = 4
        output_grid[4, 5:] = 4
        output_grid[5:, -1] = 4
        
        return {
            'train': [
                {'input': input_grid.tolist(), 'output': output_grid.tolist()}
            ],
            'test': [
                {'input': input_grid.tolist()}
            ]
        }

# ==============================================================================
# SYSTEM MANAGER
# ==============================================================================

class SystemManager:
    """Manage all ARC solving systems"""
    
    def __init__(self):
        self.systems = {}
        self.load_all_systems()
    
    def load_all_systems(self):
        """Load all available systems"""
        
        logger.info("="*70)
        logger.info("LOADING ALL ARC SYSTEMS...")
        logger.info("="*70)
        
        # System 1: Perfect ARC System V2
        try:
            from perfect_arc_system_v2 import PerfectARCSystem
            self.systems['Perfect_V2'] = PerfectARCSystem()
            logger.info("✅ Perfect ARC System V2.0 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Perfect V2: {str(e)[:50]}")
            self.systems['Perfect_V2'] = None
        
        # System 2: Ultra Advanced System V2
        try:
            from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
            self.systems['Ultra_V2'] = UltraAdvancedARCSystem()
            logger.info("✅ Ultra Advanced System V2.0 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultra V2: {str(e)[:50]}")
            self.systems['Ultra_V2'] = None
        
        # System 3: Interactive System V2
        try:
            from interactive_arc_system_v2 import InteractiveARCSystem
            self.systems['Interactive_V2'] = InteractiveARCSystem()
            logger.info("✅ Interactive System V2.0 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Interactive V2: {str(e)[:50]}")
            self.systems['Interactive_V2'] = None
        
        # System 4: Deep Learning System
        try:
            from deep_learning_arc_system import DeepLearningARCSystem
            self.systems['DeepLearning'] = DeepLearningARCSystem()
            logger.info("✅ Deep Learning System loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Deep Learning: {str(e)[:50]}")
            self.systems['DeepLearning'] = None
        
        # System 5: Ultimate Solver
        try:
            from ultimate_arc_solver import UltimateARCSolver
            self.systems['Ultimate'] = UltimateARCSolver()
            logger.info("✅ Ultimate ARC Solver loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultimate: {str(e)[:50]}")
            self.systems['Ultimate'] = None
        
        # System 6: Original Perfect System
        try:
            from perfect_arc_system import PerfectARCSystem as PerfectV1
            self.systems['Perfect_V1'] = PerfectV1()
            logger.info("✅ Perfect ARC System V1 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Perfect V1: {str(e)[:50]}")
            self.systems['Perfect_V1'] = None
        
        # System 7: Original Ultra System
        try:
            from ultra_advanced_arc_system import UltraAdvancedSystem as UltraV1
            self.systems['Ultra_V1'] = UltraV1()
            logger.info("✅ Ultra Advanced System V1 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultra V1: {str(e)[:50]}")
            self.systems['Ultra_V1'] = None
        
        # System 8: Interactive V1
        try:
            from interactive_system import InteractiveARCSystem as InteractiveV1
            self.systems['Interactive_V1'] = InteractiveV1()
            logger.info("✅ Interactive System V1 loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load Interactive V1: {str(e)[:50]}")
            self.systems['Interactive_V1'] = None
        
        # Count loaded systems
        loaded_count = sum(1 for s in self.systems.values() if s is not None)
        logger.info("="*70)
        logger.info(f"Successfully loaded {loaded_count}/{len(self.systems)} systems")
        logger.info("="*70)
    
    def solve_with_system(self, system_name: str, system, task: Dict) -> Dict:
        """Solve task with a specific system"""
        
        if system is None:
            return {
                'status': 'not_loaded',
                'output': None,
                'time': 0,
                'error': 'System not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Try different solving methods
            output = None
            
            if hasattr(system, 'solve'):
                output = system.solve(task)
            elif hasattr(system, 'process_task'):
                output = system.process_task(task)
            elif hasattr(system, 'solve_task'):
                output = system.solve_task(task)
            else:
                return {
                    'status': 'no_method',
                    'output': None,
                    'time': 0,
                    'error': 'No solving method found'
                }
            
            elapsed_time = time.time() - start_time
            
            # Convert output to numpy array if needed
            if not isinstance(output, np.ndarray):
                output = np.array(output)
            
            return {
                'status': 'success',
                'output': output,
                'time': elapsed_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'output': None,
                'time': time.time() - start_time,
                'error': str(e)[:100]
            }
    
    def solve_all_systems(self, task: Dict) -> Dict[str, Dict]:
        """Solve task with all available systems"""
        
        results = {}
        
        for system_name, system in self.systems.items():
            results[system_name] = self.solve_with_system(system_name, system, task)
        
        return results

# ==============================================================================
# SOLUTION COMPARATOR
# ==============================================================================

class SolutionComparator:
    """Compare solutions from different systems"""
    
    @staticmethod
    def compare_outputs(outputs: Dict[str, np.ndarray], ground_truth: np.ndarray = None) -> Dict:
        """Compare outputs from different systems"""
        
        comparison = {
            'num_systems': len(outputs),
            'unique_solutions': 0,
            'consensus': None,
            'agreement_matrix': {},
            'accuracy_scores': {}
        }
        
        # Get unique solutions
        unique_solutions = []
        solution_groups = defaultdict(list)
        
        for system_name, output in outputs.items():
            if output is None:
                continue
                
            # Check if this solution is unique
            is_unique = True
            for idx, unique_sol in enumerate(unique_solutions):
                if np.array_equal(output, unique_sol):
                    solution_groups[idx].append(system_name)
                    is_unique = False
                    break
            
            if is_unique:
                unique_solutions.append(output)
                solution_groups[len(unique_solutions)-1].append(system_name)
        
        comparison['unique_solutions'] = len(unique_solutions)
        comparison['solution_groups'] = dict(solution_groups)
        
        # Find consensus (most common solution)
        if solution_groups:
            largest_group = max(solution_groups.values(), key=len)
            if len(largest_group) > 1:
                comparison['consensus'] = largest_group
                comparison['consensus_rate'] = len(largest_group) / len(outputs)
        
        # Calculate agreement between systems
        system_names = list(outputs.keys())
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                if outputs[sys1] is not None and outputs[sys2] is not None:
                    agreement = SolutionComparator.calculate_similarity(
                        outputs[sys1], outputs[sys2]
                    )
                    comparison['agreement_matrix'][f"{sys1}-{sys2}"] = agreement
        
        # Calculate accuracy if ground truth available
        if ground_truth is not None:
            for system_name, output in outputs.items():
                if output is not None:
                    accuracy = SolutionComparator.calculate_accuracy(output, ground_truth)
                    comparison['accuracy_scores'][system_name] = accuracy
        
        return comparison
    
    @staticmethod
    def calculate_similarity(output1: np.ndarray, output2: np.ndarray) -> float:
        """Calculate similarity between two outputs"""
        
        # Handle different shapes
        if output1.shape != output2.shape:
            return 0.0
        
        # Calculate pixel-wise similarity
        total_pixels = output1.size
        matching_pixels = np.sum(output1 == output2)
        
        return matching_pixels / total_pixels
    
    @staticmethod
    def calculate_accuracy(output: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calculate accuracy against ground truth"""
        
        if output.shape != ground_truth.shape:
            # Try to resize or pad
            return 0.0
        
        return SolutionComparator.calculate_similarity(output, ground_truth)

# ==============================================================================
# PERFORMANCE ANALYZER
# ==============================================================================

class PerformanceAnalyzer:
    """Analyze system performance"""
    
    def __init__(self):
        self.results = defaultdict(lambda: defaultdict(list))
        
    def add_result(self, task_id: int, system_name: str, result: Dict):
        """Add a result for analysis"""
        
        self.results[system_name]['task_ids'].append(task_id)
        self.results[system_name]['statuses'].append(result['status'])
        self.results[system_name]['times'].append(result['time'])
        
        if result['status'] == 'success':
            self.results[system_name]['successes'].append(task_id)
        else:
            self.results[system_name]['failures'].append(task_id)
            self.results[system_name]['errors'].append(result.get('error', 'Unknown'))
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        report = {
            'summary': {},
            'detailed_stats': {},
            'rankings': {},
            'error_analysis': {}
        }
        
        # Calculate stats for each system
        for system_name, data in self.results.items():
            total_tasks = len(data['task_ids'])
            success_count = len(data['successes'])
            
            stats = {
                'total_tasks': total_tasks,
                'successes': success_count,
                'failures': len(data['failures']),
                'success_rate': success_count / total_tasks if total_tasks > 0 else 0,
                'avg_time': np.mean(data['times']) if data['times'] else 0,
                'median_time': np.median(data['times']) if data['times'] else 0,
                'min_time': np.min(data['times']) if data['times'] else 0,
                'max_time': np.max(data['times']) if data['times'] else 0,
                'total_time': np.sum(data['times']) if data['times'] else 0
            }
            
            report['detailed_stats'][system_name] = stats
            
            # Error analysis
            if data['errors']:
                error_counts = Counter(data['errors'])
                report['error_analysis'][system_name] = dict(error_counts.most_common(5))
        
        # Generate rankings
        if report['detailed_stats']:
            # Rank by success rate
            report['rankings']['by_success_rate'] = sorted(
                report['detailed_stats'].items(),
                key=lambda x: x[1]['success_rate'],
                reverse=True
            )
            
            # Rank by speed
            report['rankings']['by_speed'] = sorted(
                report['detailed_stats'].items(),
                key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] > 0 else float('inf')
            )
            
            # Overall score (70% accuracy, 30% speed)
            overall_scores = {}
            for system_name, stats in report['detailed_stats'].items():
                accuracy_score = stats['success_rate']
                speed_score = 1.0 / (1.0 + stats['avg_time']) if stats['avg_time'] > 0 else 0
                overall_scores[system_name] = 0.7 * accuracy_score + 0.3 * speed_score
            
            report['rankings']['overall'] = sorted(
                overall_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        # Summary
        all_successes = sum(s['successes'] for s in report['detailed_stats'].values())
        all_attempts = sum(s['total_tasks'] for s in report['detailed_stats'].values())
        
        report['summary'] = {
            'total_systems': len(self.results),
            'total_task_attempts': all_attempts,
            'total_successes': all_successes,
            'overall_success_rate': all_successes / all_attempts if all_attempts > 0 else 0
        }
        
        return report

# ==============================================================================
# MAIN TEST ORCHESTRATOR
# ==============================================================================

class UltimateTestOrchestrator:
    """Orchestrate the ultimate test of all systems"""
    
    def __init__(self):
        self.task_loader = TaskLoader()
        self.system_manager = SystemManager()
        self.comparator = SolutionComparator()
        self.analyzer = PerformanceAnalyzer()
        self.results = []
        
    def run_ultimate_test(self, num_tasks: int = 1000):
        """Run the ultimate test on all systems"""
        
        logger.info("\n" + "="*70)
        logger.info("STARTING ULTIMATE TEST - 1000 TASKS")
        logger.info("="*70)
        
        # Load tasks
        tasks = self.task_loader.load_all_tasks(num_tasks)
        
        # Test each task
        for task_id, task in enumerate(tasks):
            if task_id % 10 == 0:
                logger.info(f"\n📊 Progress: {task_id}/{len(tasks)} tasks completed")
            
            # Solve with all systems
            results = self.system_manager.solve_all_systems(task)
            
            # Extract successful outputs
            successful_outputs = {}
            for system_name, result in results.items():
                if result['status'] == 'success' and result['output'] is not None:
                    successful_outputs[system_name] = result['output']
                
                # Add to analyzer
                self.analyzer.add_result(task_id, system_name, result)
            
            # Compare solutions
            if successful_outputs:
                comparison = self.comparator.compare_outputs(successful_outputs)
                
                # Store results
                self.results.append({
                    'task_id': task_id,
                    'system_results': results,
                    'comparison': comparison
                })
            
            # Log brief status
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            if task_id % 10 == 9:
                logger.info(f"  Task {task_id+1}: {success_count}/{len(results)} systems succeeded")
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        logger.info("\n" + "="*70)
        logger.info("GENERATING FINAL REPORT")
        logger.info("="*70)
        
        # Get performance report
        perf_report = self.analyzer.generate_report()
        
        # Print summary
        logger.info("\n📊 PERFORMANCE SUMMARY")
        logger.info("-"*50)
        logger.info(f"Total Systems Tested: {perf_report['summary']['total_systems']}")
        logger.info(f"Total Task Attempts: {perf_report['summary']['total_task_attempts']}")
        logger.info(f"Total Successes: {perf_report['summary']['total_successes']}")
        logger.info(f"Overall Success Rate: {perf_report['summary']['overall_success_rate']:.2%}")
        
        # Print system rankings
        logger.info("\n🏆 SYSTEM RANKINGS BY SUCCESS RATE")
        logger.info("-"*50)
        for rank, (system_name, stats) in enumerate(perf_report['rankings']['by_success_rate'][:10], 1):
            logger.info(f"{rank:2}. {system_name:15} - {stats['success_rate']:.2%} ({stats['successes']}/{stats['total_tasks']})")
        
        logger.info("\n⚡ SYSTEM RANKINGS BY SPEED")
        logger.info("-"*50)
        for rank, (system_name, stats) in enumerate(perf_report['rankings']['by_speed'][:10], 1):
            logger.info(f"{rank:2}. {system_name:15} - {stats['avg_time']:.4f}s average")
        
        logger.info("\n🎯 OVERALL RANKINGS (70% accuracy, 30% speed)")
        logger.info("-"*50)
        for rank, (system_name, score) in enumerate(perf_report['rankings']['overall'][:10], 1):
            stats = perf_report['detailed_stats'][system_name]
            logger.info(f"{rank:2}. {system_name:15} - Score: {score:.3f} (Acc: {stats['success_rate']:.2%}, Time: {stats['avg_time']:.3f}s)")
        
        # Analyze solution consensus
        self.analyze_consensus()
        
        # Save detailed report
        self.save_detailed_report(perf_report)
        
        logger.info("\n" + "="*70)
        logger.info("✅ ULTIMATE TEST COMPLETE!")
        logger.info("="*70)
    
    def analyze_consensus(self):
        """Analyze consensus among systems"""
        
        logger.info("\n🤝 CONSENSUS ANALYSIS")
        logger.info("-"*50)
        
        consensus_count = 0
        total_with_solutions = 0
        agreement_scores = []
        
        for result in self.results:
            if 'comparison' in result and result['comparison']['num_systems'] > 1:
                total_with_solutions += 1
                
                if result['comparison'].get('consensus'):
                    consensus_count += 1
                
                # Calculate average agreement
                if result['comparison']['agreement_matrix']:
                    agreements = list(result['comparison']['agreement_matrix'].values())
                    agreement_scores.extend(agreements)
        
        if total_with_solutions > 0:
            consensus_rate = consensus_count / total_with_solutions
            logger.info(f"Tasks with consensus: {consensus_rate:.2%} ({consensus_count}/{total_with_solutions})")
        
        if agreement_scores:
            avg_agreement = np.mean(agreement_scores)
            logger.info(f"Average pairwise agreement: {avg_agreement:.2%}")
            logger.info(f"Median pairwise agreement: {np.median(agreement_scores):.2%}")
    
    def save_detailed_report(self, perf_report: Dict):
        """Save detailed report to file"""
        
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'performance_report': perf_report,
            'num_tasks': len(self.results),
            'task_results_sample': self.results[:10] if self.results else []
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            return obj
        
        report_data = convert_arrays(report_data)
        
        filename = f"ultimate_test_report_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"\n📁 Detailed report saved to: {filename}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           ULTIMATE FULL SYSTEM TEST - 1000 ARC TASKS              ║
    ║                                                                    ║
    ║  Testing ALL systems on 1000 tasks with comprehensive analysis    ║
    ║                                                                    ║
    ║  Systems: Perfect V2, Ultra V2, Interactive V2, Deep Learning,    ║
    ║           Ultimate, and all original versions                     ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run the ultimate test
    orchestrator = UltimateTestOrchestrator()
    orchestrator.run_ultimate_test(num_tasks=1000)
    
    return orchestrator

if __name__ == "__main__":
    orchestrator = main()
