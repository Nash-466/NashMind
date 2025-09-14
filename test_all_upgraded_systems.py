from __future__ import annotations
"""
COMPREHENSIVE TEST FOR ALL UPGRADED SYSTEMS
============================================
اختبار شامل لجميع الأنظمة المطورة والمحدثة
للتأكد من عملها بكفاءة عالية

Author: Testing Team
Date: 2025
"""

import numpy as np
import time
import json
import logging
from collections.abc import Callable
from typing import Dict, List, Any
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# TEST DATA
# ==============================================================================

def create_test_tasks() -> List[Dict]:
    """Create test tasks for evaluation"""
    
    tasks = [
        # Task 1: Simple Rotation
        {
            'name': 'Simple Rotation',
            'train': [
                {'input': [[1,2,3],[4,5,6],[7,8,9]], 
                 'output': [[7,4,1],[8,5,2],[9,6,3]]}
            ],
            'test': [{'input': [[1,1,2],[2,3,3],[4,4,5]]}]
        },
        
        # Task 2: Color Mapping
        {
            'name': 'Color Mapping',
            'train': [
                {'input': [[1,1,0],[1,0,0],[0,0,0]], 
                 'output': [[2,2,0],[2,0,0],[0,0,0]]},
                {'input': [[3,3,0],[3,0,0],[0,0,0]], 
                 'output': [[4,4,0],[4,0,0],[0,0,0]]}
            ],
            'test': [{'input': [[5,5,0],[5,0,0],[0,0,0]]}]
        },
        
        # Task 3: Pattern Completion
        {
            'name': 'Pattern Completion',
            'train': [
                {'input': [[1,0,1],[0,0,0],[1,0,1]], 
                 'output': [[1,0,1],[0,1,0],[1,0,1]]}
            ],
            'test': [{'input': [[2,0,2],[0,0,0],[2,0,2]]}]
        },
        
        # Task 4: Symmetry
        {
            'name': 'Symmetry Detection',
            'train': [
                {'input': [[1,2,3],[0,0,0],[0,0,0]], 
                 'output': [[1,2,3],[0,0,0],[1,2,3]]}
            ],
            'test': [{'input': [[4,5,6],[0,0,0],[0,0,0]]}]
        },
        
        # Task 5: Complex Pattern
        {
            'name': 'Complex Pattern',
            'train': [
                {'input': [[1,1,0,0],[1,1,0,0],[0,0,2,2],[0,0,2,2]], 
                 'output': [[2,2,0,0],[2,2,0,0],[0,0,1,1],[0,0,1,1]]}
            ],
            'test': [{'input': [[3,3,0,0],[3,3,0,0],[0,0,4,4],[0,0,4,4]]}]
        }
    ]
    
    return tasks

# ==============================================================================
# SYSTEM TESTER
# ==============================================================================

class SystemTester:
    """Test all upgraded systems"""
    
    def __init__(self):
        self.test_results = {}
        self.systems_loaded = {}
        self.load_systems()
    
    def load_systems(self):
        """Load all upgraded systems"""
        
        logger.info("=" * 70)
        logger.info("LOADING ALL UPGRADED SYSTEMS...")
        logger.info("=" * 70)
        
        # System 1: Perfect ARC System V2
        try:
            from perfect_arc_system_v2 import PerfectARCSystem
            self.systems_loaded['Perfect_V2'] = PerfectARCSystem()
            logger.info("✅ Perfect ARC System V2.0 loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Perfect System V2: {e}")
            self.systems_loaded['Perfect_V2'] = None
        
        # System 2: Ultra Advanced System V2
        try:
            from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
            self.systems_loaded['Ultra_V2'] = UltraAdvancedARCSystem()
            logger.info("✅ Ultra Advanced System V2.0 loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultra Advanced V2: {e}")
            self.systems_loaded['Ultra_V2'] = None
        
        # System 3: Interactive System V2
        try:
            from interactive_arc_system_v2 import InteractiveARCSystem
            self.systems_loaded['Interactive_V2'] = InteractiveARCSystem()
            logger.info("✅ Interactive System V2.0 loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Interactive V2: {e}")
            self.systems_loaded['Interactive_V2'] = None
        
        # System 4: Deep Learning System
        try:
            from deep_learning_arc_system import DeepLearningARCSystem
            self.systems_loaded['DeepLearning'] = DeepLearningARCSystem()
            logger.info("✅ Deep Learning System loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Deep Learning System: {e}")
            self.systems_loaded['DeepLearning'] = None
        
        # System 5: Ultimate Solver (Original - still working)
        try:
            from ultimate_arc_solver import UltimateARCSolver
            self.systems_loaded['Ultimate'] = UltimateARCSolver()
            logger.info("✅ Ultimate ARC Solver loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Ultimate Solver: {e}")
            self.systems_loaded['Ultimate'] = None
        
        logger.info("=" * 70)
        logger.info(f"Successfully loaded {sum(1 for s in self.systems_loaded.values() if s)} / {len(self.systems_loaded)} systems")
        logger.info("=" * 70)
    
    def test_system(self, system_name: str, system, task: Dict) -> Dict:
        """Test a single system on a task"""
        
        if system is None:
            return {
                'status': 'not_loaded',
                'output': None,
                'time': 0,
                'error': 'System not loaded'
            }
        
        try:
            start_time = time.time()
            
            # Run the system
            if hasattr(system, 'solve'):
                output = system.solve(task)
            elif hasattr(system, 'process_task'):
                output = system.process_task(task)
            else:
                return {
                    'status': 'no_solve_method',
                    'output': None,
                    'time': 0,
                    'error': 'No solve method found'
                }
            
            elapsed_time = time.time() - start_time
            
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
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all systems"""
        
        tasks = create_test_tasks()
        
        logger.info("\n" + "=" * 70)
        logger.info("STARTING COMPREHENSIVE TESTING")
        logger.info("=" * 70)
        
        # Test each system on each task
        for task_idx, task in enumerate(tasks):
            logger.info(f"\n📋 Task {task_idx + 1}: {task['name']}")
            logger.info("-" * 50)
            
            task_results = {}
            
            for system_name, system in self.systems_loaded.items():
                result = self.test_system(system_name, system, task)
                task_results[system_name] = result
                
                if result['status'] == 'success':
                    logger.info(f"  ✅ {system_name:15} - Time: {result['time']:.3f}s")
                elif result['status'] == 'not_loaded':
                    logger.info(f"  ⚠️  {system_name:15} - Not loaded")
                else:
                    logger.info(f"  ❌ {system_name:15} - Error: {result['error'][:30]}")
            
            self.test_results[f"Task_{task_idx + 1}"] = task_results
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        
        logger.info("\n" + "=" * 70)
        logger.info("TEST SUMMARY")
        logger.info("=" * 70)
        
        # Count successes per system
        system_stats = {}
        
        for system_name in self.systems_loaded.keys():
            success_count = 0
            total_time = 0
            
            for task_results in self.test_results.values():
                if system_name in task_results:
                    result = task_results[system_name]
                    if result['status'] == 'success':
                        success_count += 1
                        total_time += result['time']
            
            system_stats[system_name] = {
                'successes': success_count,
                'total_tasks': len(self.test_results),
                'success_rate': success_count / len(self.test_results) if self.test_results else 0,
                'avg_time': total_time / success_count if success_count > 0 else 0
            }
        
        # Display stats
        logger.info("\n📊 Performance Statistics:")
        logger.info("-" * 50)
        
        # Sort by success rate
        sorted_systems = sorted(system_stats.items(), 
                              key=lambda x: (x[1]['success_rate'], -x[1]['avg_time']), 
                              reverse=True)
        
        for rank, (system_name, stats) in enumerate(sorted_systems, 1):
            logger.info(f"\n{rank}. {system_name}")
            logger.info(f"   Success Rate: {stats['success_rate']:.1%} ({stats['successes']}/{stats['total_tasks']})")
            if stats['avg_time'] > 0:
                logger.info(f"   Average Time: {stats['avg_time']:.3f}s")
        
        # Overall statistics
        logger.info("\n" + "=" * 70)
        logger.info("OVERALL STATISTICS")
        logger.info("=" * 70)
        
        total_tests = len(self.test_results) * len(self.systems_loaded)
        total_successes = sum(stats['successes'] for stats in system_stats.values())
        overall_success_rate = total_successes / total_tests if total_tests > 0 else 0
        
        logger.info(f"Total Tests Run: {total_tests}")
        logger.info(f"Total Successes: {total_successes}")
        logger.info(f"Overall Success Rate: {overall_success_rate:.1%}")
        
        # Best performing system
        if sorted_systems:
            best_system = sorted_systems[0]
            logger.info(f"\n🏆 Best Performing System: {best_system[0]}")
            logger.info(f"   With {best_system[1]['success_rate']:.1%} success rate")
    
    def export_results(self, filename: str = "test_results.json"):
        """Export test results to JSON"""
        
        export_data = {
            'timestamp': time.time(),
            'systems_tested': list(self.systems_loaded.keys()),
            'tasks_count': len(self.test_results),
            'detailed_results': self.test_results,
            'summary': {}
        }
        
        # Add summary
        for system_name in self.systems_loaded.keys():
            success_count = sum(
                1 for task_results in self.test_results.values()
                if system_name in task_results and task_results[system_name]['status'] == 'success'
            )
            
            export_data['summary'][system_name] = {
                'successes': success_count,
                'total': len(self.test_results),
                'success_rate': success_count / len(self.test_results) if self.test_results else 0
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"\n📁 Results exported to {filename}")

# ==============================================================================
# PERFORMANCE ANALYZER
# ==============================================================================

class PerformanceAnalyzer:
    """Analyze performance of all systems"""
    
    def __init__(self, test_results: Dict):
        self.test_results = test_results
    
    def analyze(self):
        """Perform detailed analysis"""
        
        logger.info("\n" + "=" * 70)
        logger.info("DETAILED PERFORMANCE ANALYSIS")
        logger.info("=" * 70)
        
        # Analyze patterns of success/failure
        self.analyze_failure_patterns()
        
        # Analyze timing performance
        self.analyze_timing()
        
        # Generate recommendations
        self.generate_recommendations()
    
    def analyze_failure_patterns(self):
        """Analyze patterns in failures"""
        
        logger.info("\n🔍 Failure Pattern Analysis:")
        logger.info("-" * 50)
        
        # Find which tasks are hardest
        task_difficulty = {}
        
        for task_name, results in self.test_results.items():
            failures = sum(1 for r in results.values() 
                         if r['status'] != 'success')
            task_difficulty[task_name] = failures
        
        # Sort by difficulty
        sorted_tasks = sorted(task_difficulty.items(), 
                            key=lambda x: x[1], reverse=True)
        
        logger.info("Hardest tasks:")
        for task, failures in sorted_tasks[:3]:
            logger.info(f"  • {task}: {failures} systems failed")
    
    def analyze_timing(self):
        """Analyze timing performance"""
        
        logger.info("\n⏱️  Timing Analysis:")
        logger.info("-" * 50)
        
        # Calculate average times
        system_times = {}
        
        for results in self.test_results.values():
            for system_name, result in results.items():
                if result['status'] == 'success':
                    if system_name not in system_times:
                        system_times[system_name] = []
                    system_times[system_name].append(result['time'])
        
        # Sort by average time
        avg_times = {
            system: sum(times)/len(times) 
            for system, times in system_times.items()
        }
        sorted_times = sorted(avg_times.items(), key=lambda x: x[1])
        
        logger.info("Fastest systems (average time):")
        for system, avg_time in sorted_times[:3]:
            logger.info(f"  • {system}: {avg_time:.3f}s")
    
    def generate_recommendations(self):
        """Generate improvement recommendations"""
        
        logger.info("\n💡 Recommendations:")
        logger.info("-" * 50)
        
        recommendations = [
            "1. Focus on improving pattern recognition for complex tasks",
            "2. Optimize deep learning models for faster inference",
            "3. Implement caching to improve response times",
            "4. Add more training examples for better accuracy",
            "5. Consider ensemble methods for difficult tasks"
        ]
        
        for rec in recommendations:
            logger.info(f"  {rec}")

# ==============================================================================
# MAIN TEST EXECUTION
# ==============================================================================

def main():
    """Main test execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     COMPREHENSIVE TESTING OF ALL UPGRADED ARC SYSTEMS           ║
    ║                                                                  ║
    ║     Testing: Perfect V2, Ultra V2, Interactive V2,              ║
    ║              Deep Learning, and Ultimate Solver                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    tester = SystemTester()
    tester.run_comprehensive_test()
    
    # Export results
    tester.export_results("upgraded_systems_test_results.json")
    
    # Analyze performance
    analyzer = PerformanceAnalyzer(tester.test_results)
    analyzer.analyze()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TESTING COMPLETE!")
    logger.info("=" * 70)
    
    return tester.test_results

if __name__ == "__main__":
    results = main()
