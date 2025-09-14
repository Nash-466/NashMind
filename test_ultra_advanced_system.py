from __future__ import annotations
#!/usr/bin/env python3
"""
Test script for Ultra Advanced ARC System
Tests all advanced capabilities and generates comprehensive report
"""

import numpy as np
import time
import json
import logging
from collections.abc import Callable
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from ultra_advanced_arc_system import UltraAdvancedARCSystem, solve_arc_problem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.error(f"Ultra Advanced ARC System not available: {e}")
    SYSTEM_AVAILABLE = False

def create_test_problems() -> List[Dict[str, Any]]:
    """Create a variety of test problems for comprehensive testing"""
    
    test_problems = []
    
    # Test 1: Simple symmetry problem
    test_problems.append({
        'name': 'Simple Symmetry',
        'description': 'Test basic symmetry detection and transformation',
        'input_grid': np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0]
        ]),
        'expected_patterns': ['symmetry'],
        'difficulty': 'easy'
    })
    
    # Test 2: Color mapping problem
    test_problems.append({
        'name': 'Color Mapping',
        'description': 'Test color transformation capabilities',
        'input_grid': np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ]),
        'expected_patterns': ['color_mapping', 'repetition'],
        'difficulty': 'medium'
    })
    
    # Test 3: Complex pattern problem
    test_problems.append({
        'name': 'Complex Pattern',
        'description': 'Test advanced pattern recognition',
        'input_grid': np.array([
            [0, 1, 0, 1, 0],
            [1, 2, 3, 2, 1],
            [0, 3, 4, 3, 0],
            [1, 2, 3, 2, 1],
            [0, 1, 0, 1, 0]
        ]),
        'expected_patterns': ['nested_symmetry', 'concentric'],
        'difficulty': 'hard'
    })
    
    # Test 4: Sparse pattern problem
    test_problems.append({
        'name': 'Sparse Pattern',
        'description': 'Test handling of sparse patterns',
        'input_grid': np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 4, 0],
            [0, 0, 0, 0, 0, 0]
        ]),
        'expected_patterns': ['sparse', 'positional'],
        'difficulty': 'medium'
    })
    
    # Test 5: Large grid problem
    test_problems.append({
        'name': 'Large Grid',
        'description': 'Test scalability with larger grids',
        'input_grid': np.random.randint(0, 5, size=(10, 10)),
        'expected_patterns': ['random', 'statistical'],
        'difficulty': 'hard'
    })
    
    return test_problems

def run_comprehensive_test(arc_system: UltraAdvancedARCSystem, 
                         test_problem: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive test on a single problem"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_problem['name']}")
    print(f"Description: {test_problem['description']}")
    print(f"Difficulty: {test_problem['difficulty']}")
    print(f"Input shape: {test_problem['input_grid'].shape}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Solve the problem
        solution = arc_system.solve_arc_challenge(
            test_problem['input_grid'],
            context={'test_name': test_problem['name']}
        )
        
        test_time = time.time() - start_time
        
        # Analyze results
        results = {
            'test_name': test_problem['name'],
            'success': True,
            'solution_confidence': solution.confidence,
            'generation_time': solution.generation_time,
            'test_time': test_time,
            'approach_used': solution.metadata.get('approach_used', 'unknown'),
            'patterns_detected': len(solution.patterns_used),
            'transformations_applied': len(solution.transformations_applied),
            'verification_passed': solution.verification_results.get('success', False),
            'quality_score': solution.quality_assessment.get('overall_quality', 0.0),
            'reasoning_steps': len(solution.reasoning_chain),
            'enhanced': solution.metadata.get('enhanced', False),
            'creatively_enhanced': solution.metadata.get('creatively_enhanced', False)
        }
        
        # Print detailed results
        print(f"✅ Solution generated successfully!")
        print(f"   Confidence: {solution.confidence:.3f}")
        print(f"   Generation time: {solution.generation_time:.3f}s")
        print(f"   Approach: {results['approach_used']}")
        print(f"   Patterns detected: {results['patterns_detected']}")
        print(f"   Transformations: {results['transformations_applied']}")
        print(f"   Enhanced: {results['enhanced']}")
        print(f"   Creative enhancement: {results['creatively_enhanced']}")
        
        if solution.reasoning_chain:
            print(f"   Reasoning chain:")
            for i, step in enumerate(solution.reasoning_chain[:3], 1):
                print(f"     {i}. {step}")
        
        return results
        
    except Exception as e:
        test_time = time.time() - start_time
        print(f"❌ Test failed: {str(e)}")
        
        return {
            'test_name': test_problem['name'],
            'success': False,
            'error': str(e),
            'test_time': test_time
        }

def analyze_system_capabilities(arc_system: UltraAdvancedARCSystem) -> Dict[str, Any]:
    """Analyze system capabilities and status"""
    
    print(f"\n{'='*60}")
    print("SYSTEM CAPABILITY ANALYSIS")
    print(f"{'='*60}")
    
    # Get system status
    status = arc_system.get_system_status()
    
    capabilities = {
        'system_info': {
            'name': status.get('system_name', 'Unknown'),
            'version': status.get('version', 'Unknown'),
            'advanced_systems': status.get('advanced_systems_available', False)
        },
        'memory_capacity': {
            'memory_nodes': status.get('memory_nodes', 0),
            'learned_patterns': status.get('learned_patterns', 0),
            'creative_ideas': status.get('creative_ideas', 0),
            'solution_history': status.get('solutions_in_history', 0)
        },
        'performance_metrics': status.get('performance_stats', {})
    }
    
    # Print capability analysis
    print(f"System: {capabilities['system_info']['name']} v{capabilities['system_info']['version']}")
    print(f"Advanced Systems Available: {capabilities['system_info']['advanced_systems']}")
    print(f"Memory Nodes: {capabilities['memory_capacity']['memory_nodes']}")
    print(f"Learned Patterns: {capabilities['memory_capacity']['learned_patterns']}")
    print(f"Creative Ideas: {capabilities['memory_capacity']['creative_ideas']}")
    print(f"Solutions in History: {capabilities['memory_capacity']['solution_history']}")
    
    performance = capabilities['performance_metrics']
    if performance:
        print(f"Success Rate: {performance.get('success_rate', 0.0):.1%}")
        print(f"Average Confidence: {performance.get('average_confidence', 0.0):.3f}")
        print(f"Average Solution Time: {performance.get('average_solution_time', 0.0):.3f}s")
    
    return capabilities

def generate_comprehensive_report(test_results: List[Dict[str, Any]], 
                                capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    
    successful_tests = [r for r in test_results if r.get('success', False)]
    failed_tests = [r for r in test_results if not r.get('success', False)]
    
    if successful_tests:
        avg_confidence = np.mean([r['solution_confidence'] for r in successful_tests])
        avg_generation_time = np.mean([r['generation_time'] for r in successful_tests])
        avg_quality = np.mean([r.get('quality_score', 0.0) for r in successful_tests])
    else:
        avg_confidence = 0.0
        avg_generation_time = 0.0
        avg_quality = 0.0
    
    report = {
        'test_summary': {
            'total_tests': len(test_results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'success_rate': len(successful_tests) / len(test_results) if test_results else 0.0
        },
        'performance_metrics': {
            'average_confidence': avg_confidence,
            'average_generation_time': avg_generation_time,
            'average_quality_score': avg_quality
        },
        'capability_analysis': capabilities,
        'detailed_results': test_results,
        'recommendations': []
    }
    
    # Generate recommendations
    if report['test_summary']['success_rate'] < 0.8:
        report['recommendations'].append("Consider tuning system parameters for better success rate")
    
    if avg_confidence < 0.7:
        report['recommendations'].append("Improve confidence estimation mechanisms")
    
    if avg_generation_time > 5.0:
        report['recommendations'].append("Optimize performance for faster solution generation")
    
    return report

def main():
    """Main test execution function"""
    
    print("🚀 ULTRA ADVANCED ARC SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    if not SYSTEM_AVAILABLE:
        print("❌ Ultra Advanced ARC System not available. Please check installation.")
        return
    
    # Initialize system
    print("Initializing Ultra Advanced ARC System...")
    arc_system = UltraAdvancedARCSystem()
    
    # Analyze system capabilities
    capabilities = analyze_system_capabilities(arc_system)
    
    # Create test problems
    print("\nCreating test problems...")
    test_problems = create_test_problems()
    print(f"Created {len(test_problems)} test problems")
    
    # Run comprehensive tests
    print("\nRunning comprehensive tests...")
    test_results = []
    
    for problem in test_problems:
        result = run_comprehensive_test(arc_system, problem)
        test_results.append(result)
    
    # Generate comprehensive report
    print("\nGenerating comprehensive report...")
    report = generate_comprehensive_report(test_results, capabilities)
    
    # Print final report
    print(f"\n{'='*80}")
    print("FINAL TEST REPORT")
    print(f"{'='*80}")
    
    summary = report['test_summary']
    metrics = report['performance_metrics']
    
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Confidence: {metrics['average_confidence']:.3f}")
    print(f"Average Generation Time: {metrics['average_generation_time']:.3f}s")
    print(f"Average Quality Score: {metrics['average_quality_score']:.3f}")
    
    if report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save detailed report
    report_filename = f"arc_test_report_{int(time.time())}.json"
    try:
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_filename}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
    
    print(f"\n🎉 Test suite completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
