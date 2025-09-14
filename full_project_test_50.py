from __future__ import annotations
"""
FULL PROJECT TEST - 50 TASKS COMPREHENSIVE EVALUATION
======================================================
اختبار شامل لجميع أنظمة مشروع برهان على 50 مهمة
مع مقارنة النتائج بالحلول الصحيحة

Author: Enhanced Testing System
Version: 2.0
Date: 2025
"""

import numpy as np
import json
import time
import logging
import os
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import traceback
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class TaskResult:
    """Result for a single task"""
    task_id: str
    system_name: str
    success: bool
    similarity: float
    exact_match: bool
    execution_time: float
    output_shape: Optional[Tuple] = None
    expected_shape: Optional[Tuple] = None
    error: Optional[str] = None

@dataclass 
class SystemReport:
    """Complete report for a system"""
    system_name: str
    total_tasks: int = 0
    exact_matches: int = 0
    partial_matches: int = 0  # >80% similarity
    failed: int = 0
    success_rate: float = 0.0
    exact_match_rate: float = 0.0
    average_similarity: float = 0.0
    average_time: float = 0.0
    total_time: float = 0.0
    task_results: List[TaskResult] = field(default_factory=list)
    best_tasks: List[str] = field(default_factory=list)  # Tasks with 100% match
    worst_tasks: List[str] = field(default_factory=list)  # Tasks with 0% similarity

# ==============================================================================
# COMPREHENSIVE TEST SYSTEM
# ==============================================================================

class FullProjectTester:
    """Comprehensive testing system for all ARC solvers"""
    
    def __init__(self, num_tasks: int = 50):
        self.num_tasks = num_tasks
        self.systems = {}
        self.test_tasks = []
        self.solutions = {}
        self.reports = {}
        self.load_systems()
        self.load_test_data()
        
    def load_systems(self):
        """Load all available systems"""
        print("🔄 Loading all systems...")
        
        # 1. Ultimate ARC Solver (NEW - BEST)
        try:
            from ultimate_arc_solver import UltimateARCSolver
            self.systems['Ultimate ARC Solver (NEW)'] = {
                'instance': UltimateARCSolver(),
                'interface': 'solve'
            }
            print("   ✅ Ultimate ARC Solver loaded")
        except Exception as e:
            print(f"   ❌ Ultimate ARC Solver failed: {e}")
        
        # 2. Ultra Advanced ARC System
        try:
            from ultra_advanced_arc_system import UltraAdvancedARCSystem
            self.systems['Ultra Advanced ARC System'] = {
                'instance': UltraAdvancedARCSystem(),
                'interface': 'solve_arc_challenge'
            }
            print("   ✅ Ultra Advanced ARC System loaded")
        except Exception as e:
            print(f"   ❌ Ultra Advanced ARC System failed: {e}")
        
        # 3. Interactive System
        try:
            from arc_interactive_system import ARCInteractiveSystem
            self.systems['Interactive System'] = {
                'instance': ARCInteractiveSystem(),
                'interface': 'process_task_interactive'
            }
            print("   ✅ Interactive System loaded")
        except Exception as e:
            print(f"   ❌ Interactive System failed: {e}")
        
        # 4. Perfect ARC System
        try:
            from perfect_arc_system import PerfectARCSystem
            self.systems['Perfect ARC System'] = {
                'instance': PerfectARCSystem(),
                'interface': 'solve'
            }
            print("   ✅ Perfect ARC System loaded")
        except Exception as e:
            print(f"   ❌ Perfect ARC System failed: {e}")
        
        # 5. Revolutionary ARC System
        try:
            from revolutionary_arc_system import RevolutionaryARCSystem
            self.systems['Revolutionary ARC System'] = {
                'instance': RevolutionaryARCSystem(),
                'interface': 'solve'
            }
            print("   ✅ Revolutionary ARC System loaded")
        except Exception as e:
            print(f"   ❌ Revolutionary ARC System failed: {e}")
        
        # 6. Ultimate ARC System
        try:
            from ultimate_arc_system import UltimateARCSystem
            self.systems['Ultimate ARC System'] = {
                'instance': UltimateARCSystem(),
                'interface': 'solve'
            }
            print("   ✅ Ultimate ARC System loaded")
        except Exception as e:
            print(f"   ❌ Ultimate ARC System failed: {e}")
            
        print(f"\n📊 Total systems loaded: {len(self.systems)}")
        
    def load_test_data(self):
        """Load test tasks and solutions"""
        print(f"\n🔄 Loading {self.num_tasks} test tasks...")
        
        # Load training challenges and solutions
        challenges_file = "arc-agi_training_challenges.json"
        solutions_file = "arc-agi_training_solutions.json"
        
        if not os.path.exists(challenges_file) or not os.path.exists(solutions_file):
            print("   ❌ Data files not found, creating sample tasks...")
            self.create_sample_tasks()
            return
            
        try:
            with open(challenges_file, 'r') as f:
                all_challenges = json.load(f)
            with open(solutions_file, 'r') as f:
                all_solutions = json.load(f)
            
            # Take first N tasks
            task_ids = list(all_challenges.keys())[:self.num_tasks]
            
            for task_id in task_ids:
                if task_id in all_solutions:
                    task = all_challenges[task_id]
                    task['id'] = task_id
                    self.test_tasks.append(task)
                    self.solutions[task_id] = all_solutions[task_id]
            
            print(f"   ✅ Loaded {len(self.test_tasks)} tasks with solutions")
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            self.create_sample_tasks()
    
    def create_sample_tasks(self):
        """Create sample tasks if data files not available"""
        print("   📝 Creating 5 sample tasks for testing...")
        
        sample_tasks = [
            {
                'id': 'sample_001',
                'train': [
                    {'input': [[0,1,0],[1,1,1],[0,1,0]], 
                     'output': [[1,0,1],[0,0,0],[1,0,1]]}
                ],
                'test': [{'input': [[0,0,1],[0,1,1],[1,1,1]]}]
            },
            {
                'id': 'sample_002',
                'train': [
                    {'input': [[1,0],[0,1]], 
                     'output': [[0,1],[1,0]]}
                ],
                'test': [{'input': [[2,0],[0,2]]}]
            },
            {
                'id': 'sample_003',
                'train': [
                    {'input': [[1,1,1],[0,0,0],[2,2,2]], 
                     'output': [[2,2,2],[0,0,0],[1,1,1]]}
                ],
                'test': [{'input': [[3,3,3],[0,0,0],[4,4,4]]}]
            },
            {
                'id': 'sample_004',
                'train': [
                    {'input': [[1,2],[3,4]], 
                     'output': [[4,3],[2,1]]}
                ],
                'test': [{'input': [[5,6],[7,8]]}]
            },
            {
                'id': 'sample_005',
                'train': [
                    {'input': [[0,1,0],[0,1,0],[0,1,0]], 
                     'output': [[0,0,0],[1,1,1],[0,0,0]]}
                ],
                'test': [{'input': [[1,0,0],[1,0,0],[1,0,0]]}]
            }
        ]
        
        for task in sample_tasks[:self.num_tasks]:
            self.test_tasks.append(task)
            # Create expected output (for samples, use inverted input as solution)
            test_input = np.array(task['test'][0]['input'])
            expected = np.where(test_input > 0, 0, 1)  # Simple inversion
            self.solutions[task['id']] = [expected.tolist()]
        
        self.num_tasks = min(self.num_tasks, len(sample_tasks))
    
    def solve_with_system(self, system_info: Dict, task: Dict) -> Optional[np.ndarray]:
        """Solve a task with a specific system"""
        try:
            system = system_info['instance']
            interface = system_info['interface']
            
            if interface == 'solve':
                # Simple solve interface
                if hasattr(system, 'solve'):
                    return system.solve(task)
                    
            elif interface == 'solve_arc_challenge':
                # Ultra Advanced interface
                test_input = np.array(task['test'][0]['input'])
                result = system.solve_arc_challenge(test_input, task)
                if hasattr(result, 'solution_grid'):
                    return result.solution_grid
                return result
                
            elif interface == 'process_task_interactive':
                # Interactive interface
                result = system.process_task_interactive(task, task['id'])
                if hasattr(result, 'final_solution'):
                    return result.final_solution
                    
            return None
            
        except Exception as e:
            logger.debug(f"System solve error: {e}")
            return None
    
    def calculate_similarity(self, output: np.ndarray, expected: np.ndarray) -> float:
        """Calculate similarity between output and expected"""
        if output is None or expected is None:
            return 0.0
            
        # If shapes don't match, return 0
        if output.shape != expected.shape:
            return 0.0
        
        # Calculate pixel-wise similarity
        matching = np.sum(output == expected)
        total = output.size
        
        return (matching / total) if total > 0 else 0.0
    
    def test_single_system(self, system_name: str, system_info: Dict) -> SystemReport:
        """Test a single system on all tasks"""
        report = SystemReport(system_name=system_name)
        
        print(f"\n🔄 Testing {system_name}...")
        
        for i, task in enumerate(self.test_tasks, 1):
            task_id = task['id']
            
            # Get expected output
            expected = np.array(self.solutions[task_id][0])
            
            # Solve with system
            start_time = time.time()
            try:
                output = self.solve_with_system(system_info, task)
                execution_time = time.time() - start_time
                
                if output is not None:
                    output = np.array(output) if not isinstance(output, np.ndarray) else output
                    similarity = self.calculate_similarity(output, expected)
                    exact_match = np.array_equal(output, expected)
                    
                    result = TaskResult(
                        task_id=task_id,
                        system_name=system_name,
                        success=True,
                        similarity=similarity,
                        exact_match=exact_match,
                        execution_time=execution_time,
                        output_shape=output.shape,
                        expected_shape=expected.shape
                    )
                else:
                    result = TaskResult(
                        task_id=task_id,
                        system_name=system_name,
                        success=False,
                        similarity=0.0,
                        exact_match=False,
                        execution_time=execution_time,
                        error="No output generated"
                    )
                    
            except Exception as e:
                result = TaskResult(
                    task_id=task_id,
                    system_name=system_name,
                    success=False,
                    similarity=0.0,
                    exact_match=False,
                    execution_time=time.time() - start_time,
                    error=str(e)
                )
            
            report.task_results.append(result)
            
            # Update statistics
            report.total_tasks += 1
            if result.exact_match:
                report.exact_matches += 1
                report.best_tasks.append(task_id)
            elif result.similarity > 0.8:
                report.partial_matches += 1
            elif result.similarity == 0.0:
                report.failed += 1
                report.worst_tasks.append(task_id)
            
            # Progress indicator
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(self.test_tasks)} tasks")
        
        # Calculate final statistics
        if report.total_tasks > 0:
            report.success_rate = (report.exact_matches + report.partial_matches) / report.total_tasks * 100
            report.exact_match_rate = report.exact_matches / report.total_tasks * 100
            similarities = [r.similarity for r in report.task_results]
            report.average_similarity = np.mean(similarities) * 100
            times = [r.execution_time for r in report.task_results]
            report.average_time = np.mean(times)
            report.total_time = sum(times)
        
        print(f"   ✅ Complete - Exact: {report.exact_matches}/{report.total_tasks}, Avg Similarity: {report.average_similarity:.1f}%")
        
        return report
    
    def run_all_tests(self):
        """Run tests on all systems"""
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE TEST ON {} TASKS".format(self.num_tasks))
        print("="*80)
        
        start_time = time.time()
        
        for system_name, system_info in self.systems.items():
            report = self.test_single_system(system_name, system_info)
            self.reports[system_name] = report
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️ Total testing time: {total_time:.2f} seconds")
        
        return self.reports
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS - {} TASKS".format(self.num_tasks))
        print("="*80)
        
        # Sort systems by exact match rate, then by average similarity
        sorted_systems = sorted(
            self.reports.items(),
            key=lambda x: (x[1].exact_match_rate, x[1].average_similarity),
            reverse=True
        )
        
        print("\n🏆 SYSTEM RANKINGS:")
        print("-"*80)
        
        for rank, (system_name, report) in enumerate(sorted_systems, 1):
            print(f"\n{rank}. {system_name}")
            print(f"   📈 Exact Matches: {report.exact_matches}/{report.total_tasks} ({report.exact_match_rate:.1f}%)")
            print(f"   📊 Partial Matches (>80%): {report.partial_matches}")
            print(f"   🎯 Average Similarity: {report.average_similarity:.1f}%")
            print(f"   ⚡ Average Time: {report.average_time:.3f}s")
            print(f"   ⏱️ Total Time: {report.total_time:.2f}s")
            
            if report.best_tasks:
                print(f"   ✅ Perfect Tasks: {len(report.best_tasks)} tasks")
            if report.worst_tasks:
                print(f"   ❌ Failed Tasks: {len(report.worst_tasks)} tasks")
        
        # Best overall system
        if sorted_systems:
            best_system = sorted_systems[0]
            print("\n" + "="*80)
            print(f"🥇 BEST SYSTEM: {best_system[0]}")
            print(f"   Exact Match Rate: {best_system[1].exact_match_rate:.1f}%")
            print(f"   Average Similarity: {best_system[1].average_similarity:.1f}%")
            print("="*80)
        
        # Save detailed report
        self.save_detailed_report()
        
        return sorted_systems
    
    def save_detailed_report(self):
        """Save detailed report to JSON file"""
        report_data = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_tasks': self.num_tasks,
            'num_systems': len(self.systems),
            'systems': {}
        }
        
        for system_name, report in self.reports.items():
            report_data['systems'][system_name] = {
                'summary': {
                    'total_tasks': report.total_tasks,
                    'exact_matches': report.exact_matches,
                    'partial_matches': report.partial_matches,
                    'failed': report.failed,
                    'exact_match_rate': report.exact_match_rate,
                    'average_similarity': report.average_similarity,
                    'average_time': report.average_time,
                    'total_time': report.total_time
                },
                'best_tasks': report.best_tasks[:10],  # Top 10
                'worst_tasks': report.worst_tasks[:10],  # Bottom 10
                'task_details': [
                    {
                        'task_id': r.task_id,
                        'exact_match': r.exact_match,
                        'similarity': r.similarity,
                        'time': r.execution_time
                    }
                    for r in report.task_results[:20]  # First 20 for brevity
                ]
            }
        
        filename = f'full_test_report_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📝 Detailed report saved to: {filename}")
        
        return filename

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    print("🎯 COMPREHENSIVE PROJECT TEST - 50 TASKS")
    print("="*80)
    print("Testing all systems in مشروع برهان on real ARC tasks")
    print("Comparing results with correct solutions")
    print("="*80)
    
    # Create tester
    tester = FullProjectTester(num_tasks=50)
    
    # Run all tests
    reports = tester.run_all_tests()
    
    # Generate comparison report
    rankings = tester.generate_comparison_report()
    
    # Summary
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    if rankings:
        print(f"\n🏆 Winner: {rankings[0][0]}")
        print(f"   Performance: {rankings[0][1].exact_match_rate:.1f}% exact matches")
        print(f"   Similarity: {rankings[0][1].average_similarity:.1f}% average")
    
    return rankings

if __name__ == "__main__":
    rankings = main()
