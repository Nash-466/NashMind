from __future__ import annotations
"""
COMPREHENSIVE TEST SUITE FOR ARC SYSTEMS
=========================================
نظام اختبار شامل لتقييم جميع أنظمة ARC وقياس التحسينات

Author: Enhanced Testing System
Version: 1.0
Date: 2025
"""

import numpy as np
import json
import time
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import os
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# TEST RESULT STRUCTURES
# ==============================================================================

@dataclass
class TestResult:
    """Result from a single test"""
    task_id: str
    system_name: str
    success: bool
    similarity_score: float
    execution_time: float
    error_message: Optional[str] = None
    output_shape: Optional[Tuple] = None
    expected_shape: Optional[Tuple] = None

@dataclass
class SystemPerformance:
    """Overall performance metrics for a system"""
    system_name: str
    total_tests: int = 0
    successful_tests: int = 0
    failed_tests: int = 0
    average_similarity: float = 0.0
    average_time: float = 0.0
    success_rate: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)

# ==============================================================================
# TEST RUNNER
# ==============================================================================

class ComprehensiveTestRunner:
    """Runs comprehensive tests on all ARC systems"""
    
    def __init__(self, data_path: str = "."):
        self.data_path = data_path
        self.systems = self._load_systems()
        self.test_data = self._load_test_data()
        self.results = {}
        
    def _load_systems(self) -> Dict[str, Any]:
        """Load all available ARC systems"""
        
        systems = {}
        
        # Load the new Ultimate Solver (HIGHEST PRIORITY)
        try:
            from ultimate_arc_solver import UltimateARCSolver
            systems['Ultimate ARC Solver (NEW)'] = UltimateARCSolver()
            logger.info("✅ Loaded Ultimate ARC Solver (NEW)")
        except Exception as e:
            logger.warning(f"Could not load Ultimate ARC Solver: {e}")
        
        # Load existing systems
        try:
            from ultra_advanced_arc_system import UltraAdvancedARCSystem
            systems['Ultra Advanced ARC System'] = UltraAdvancedARCSystem()
            logger.info("✅ Loaded Ultra Advanced ARC System")
        except Exception as e:
            logger.warning(f"Could not load Ultra Advanced ARC System: {e}")
            
        try:
            from arc_interactive_system import ARCInteractiveSystem
            systems['Interactive System'] = ARCInteractiveSystem()
            logger.info("✅ Loaded Interactive System")
        except Exception as e:
            logger.warning(f"Could not load Interactive System: {e}")
            
        try:
            from perfect_arc_system import PerfectARCSystem
            systems['Perfect ARC System'] = PerfectARCSystem()
            logger.info("✅ Loaded Perfect ARC System")
        except Exception as e:
            logger.warning(f"Could not load Perfect ARC System: {e}")
            
        try:
            from revolutionary_arc_system import RevolutionaryARCSystem
            systems['Revolutionary ARC System'] = RevolutionaryARCSystem()
            logger.info("✅ Loaded Revolutionary ARC System")
        except Exception as e:
            logger.warning(f"Could not load Revolutionary ARC System: {e}")
            
        try:
            from ultimate_arc_system import UltimateARCSystem
            systems['Ultimate ARC System'] = UltimateARCSystem()
            logger.info("✅ Loaded Ultimate ARC System")
        except Exception as e:
            logger.warning(f"Could not load Ultimate ARC System: {e}")
            
        logger.info(f"Total systems loaded: {len(systems)}")
        return systems
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data from files"""
        
        test_tasks = []
        
        # Try to load training data for testing
        training_file = os.path.join(self.data_path, "arc-agi_training_challenges.json")
        solutions_file = os.path.join(self.data_path, "arc-agi_training_solutions.json")
        
        if os.path.exists(training_file) and os.path.exists(solutions_file):
            try:
                with open(training_file, 'r') as f:
                    challenges = json.load(f)
                with open(solutions_file, 'r') as f:
                    solutions = json.load(f)
                
                # Take first 20 tasks for quick testing
                for task_id in list(challenges.keys())[:20]:
                    if task_id in solutions:
                        task = challenges[task_id]
                        task['id'] = task_id
                        task['expected_output'] = solutions[task_id]
                        test_tasks.append(task)
                
                logger.info(f"Loaded {len(test_tasks)} test tasks")
            except Exception as e:
                logger.error(f"Error loading test data: {e}")
        
        # If no file data, create sample tasks
        if not test_tasks:
            logger.info("Creating sample test tasks")
            test_tasks = self._create_sample_tasks()
        
        return test_tasks
    
    def _create_sample_tasks(self) -> List[Dict]:
        """Create sample test tasks"""
        
        return [
            {
                'id': 'sample_1',
                'train': [
                    {
                        'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        'output': [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
                    }
                ],
                'test': [
                    {
                        'input': [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
                    }
                ],
                'expected_output': [[[1, 1, 0], [1, 0, 0], [0, 0, 0]]]
            },
            {
                'id': 'sample_2',
                'train': [
                    {
                        'input': [[1, 0], [0, 1]],
                        'output': [[0, 1], [1, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[2, 0], [0, 2]]
                    }
                ],
                'expected_output': [[[0, 2], [2, 0]]]
            }
        ]
    
    def run_tests(self, max_tasks: int = None) -> Dict[str, SystemPerformance]:
        """Run comprehensive tests on all systems"""
        
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE TESTS")
        logger.info("=" * 60)
        
        results = {}
        tasks_to_test = self.test_data[:max_tasks] if max_tasks else self.test_data
        
        for system_name, system in self.systems.items():
            logger.info(f"\n🔄 Testing: {system_name}")
            performance = self._test_system(system, system_name, tasks_to_test)
            results[system_name] = performance
            
            # Print summary
            logger.info(f"✅ {system_name} Results:")
            logger.info(f"   Success Rate: {performance.success_rate:.1%}")
            logger.info(f"   Avg Similarity: {performance.average_similarity:.1%}")
            logger.info(f"   Avg Time: {performance.average_time:.3f}s")
        
        self.results = results
        return results
    
    def _test_system(self, system: Any, system_name: str, 
                    tasks: List[Dict]) -> SystemPerformance:
        """Test a single system on all tasks"""
        
        performance = SystemPerformance(system_name=system_name)
        
        for task in tasks:
            test_result = self._test_single_task(system, task, system_name)
            performance.test_results.append(test_result)
            
            # Update metrics
            performance.total_tests += 1
            if test_result.success:
                performance.successful_tests += 1
            else:
                performance.failed_tests += 1
        
        # Calculate averages
        if performance.test_results:
            performance.average_similarity = np.mean([r.similarity_score for r in performance.test_results])
            performance.average_time = np.mean([r.execution_time for r in performance.test_results])
            performance.success_rate = performance.successful_tests / performance.total_tests
        
        return performance
    
    def _test_single_task(self, system: Any, task: Dict, system_name: str) -> TestResult:
        """Test a system on a single task"""
        
        task_id = task.get('id', 'unknown')
        start_time = time.time()
        
        try:
            # Different systems have different interfaces
            if hasattr(system, 'solve'):
                # New Ultimate Solver interface
                output = system.solve(task)
            elif hasattr(system, 'solve_arc_challenge'):
                # Ultra Advanced system interface
                test_input = np.array(task['test'][0]['input'])
                solution = system.solve_arc_challenge(test_input, task)
                output = solution.solution_grid if hasattr(solution, 'solution_grid') else solution
            elif hasattr(system, 'process_task_interactive'):
                # Interactive system interface
                result = system.process_task_interactive(task, task_id)
                output = result.final_solution if hasattr(result, 'final_solution') else None
            elif hasattr(system, 'process_task'):
                # Generic interface
                result = system.process_task(task)
                output = result.get('solution') if isinstance(result, dict) else result
            else:
                # Unknown interface
                output = None
            
            execution_time = time.time() - start_time
            
            # Calculate similarity
            if output is not None and 'expected_output' in task:
                expected = np.array(task['expected_output'][0])
                output = np.array(output) if not isinstance(output, np.ndarray) else output
                
                similarity = self._calculate_similarity(output, expected)
                success = np.array_equal(output, expected)
                
                return TestResult(
                    task_id=task_id,
                    system_name=system_name,
                    success=success,
                    similarity_score=similarity,
                    execution_time=execution_time,
                    output_shape=output.shape,
                    expected_shape=expected.shape
                )
            else:
                return TestResult(
                    task_id=task_id,
                    system_name=system_name,
                    success=False,
                    similarity_score=0.0,
                    execution_time=execution_time,
                    error_message="No output generated"
                )
                
        except Exception as e:
            return TestResult(
                task_id=task_id,
                system_name=system_name,
                success=False,
                similarity_score=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _calculate_similarity(self, output: np.ndarray, expected: np.ndarray) -> float:
        """Calculate similarity between output and expected"""
        
        if output.shape != expected.shape:
            return 0.0
        
        matching = np.sum(output == expected)
        total = output.size
        
        return matching / total if total > 0 else 0.0
    
    def generate_report(self, output_file: str = "test_report.json"):
        """Generate comprehensive test report"""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_systems_tested': len(self.results),
            'total_tasks': len(self.test_data),
            'system_rankings': [],
            'detailed_results': {}
        }
        
        # Rank systems by performance
        rankings = []
        for system_name, performance in self.results.items():
            rankings.append({
                'rank': 0,  # Will be set after sorting
                'system': system_name,
                'success_rate': performance.success_rate,
                'avg_similarity': performance.average_similarity,
                'avg_time': performance.average_time,
                'successful_tests': performance.successful_tests,
                'total_tests': performance.total_tests
            })
        
        # Sort by success rate, then by similarity
        rankings.sort(key=lambda x: (x['success_rate'], x['avg_similarity']), reverse=True)
        
        # Set ranks
        for i, ranking in enumerate(rankings, 1):
            ranking['rank'] = i
        
        report['system_rankings'] = rankings
        
        # Add detailed results
        for system_name, performance in self.results.items():
            report['detailed_results'][system_name] = {
                'metrics': {
                    'success_rate': performance.success_rate,
                    'average_similarity': performance.average_similarity,
                    'average_time': performance.average_time,
                    'successful_tests': performance.successful_tests,
                    'failed_tests': performance.failed_tests
                },
                'task_results': [
                    {
                        'task_id': r.task_id,
                        'success': r.success,
                        'similarity': r.similarity_score,
                        'time': r.execution_time,
                        'error': r.error_message
                    }
                    for r in performance.test_results
                ]
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"\n📊 System Rankings:\n")
        
        for ranking in rankings:
            print(f"{ranking['rank']}. {ranking['system']}")
            print(f"   ✅ Success Rate: {ranking['success_rate']:.1%}")
            print(f"   📈 Avg Similarity: {ranking['avg_similarity']:.1%}")
            print(f"   ⏱️ Avg Time: {ranking['avg_time']:.3f}s")
            print(f"   📊 Tests: {ranking['successful_tests']}/{ranking['total_tests']}")
            print()
        
        print(f"📝 Full report saved to: {output_file}")
        
        return report

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_comprehensive_tests(max_tasks: int = 10):
    """Main function to run comprehensive tests"""
    
    print("🚀 Starting Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Run tests
    results = runner.run_tests(max_tasks=max_tasks)
    
    # Generate report
    report = runner.generate_report("comprehensive_test_report.json")
    
    # Return best performing system
    if report['system_rankings']:
        best_system = report['system_rankings'][0]
        print(f"\n🏆 BEST PERFORMING SYSTEM: {best_system['system']}")
        print(f"   Success Rate: {best_system['success_rate']:.1%}")
        
        return best_system
    
    return None

if __name__ == "__main__":
    # Run tests
    best_system = run_comprehensive_tests(max_tasks=10)
    
    if best_system:
        print(f"\n✨ Recommended System: {best_system['system']}")
        print(f"   Use this system for optimal ARC solving performance!")
