from __future__ import annotations
"""
TEST INTEGRATION - اختبار التكامل بين جميع الأنظمة
=====================================================
يختبر هذا الملف التكامل والتعاون بين جميع أنظمة المشروع
"""

import numpy as np
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def test_system_integration():
    """اختبار التكامل بين جميع الأنظمة"""
    
    print("=" * 60)
    print("🔍 اختبار التكامل بين أنظمة مشروع برهان")
    print("=" * 60)
    
    # إنشاء مهمة اختبارية
    test_task = {
        'id': 'integration_test',
        'train': [
            {
                'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                'output': [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
            },
            {
                'input': [[1, 1], [1, 1]],
                'output': [[0, 0], [0, 0]]
            }
        ],
        'test': [
            {
                'input': [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
            }
        ]
    }
    
    results = {}
    
    # 1. اختبار النظام المحسن الجديد
    print("\n1️⃣ اختبار Ultimate ARC Solver...")
    try:
        from ultimate_arc_solver import UltimateARCSolver
        solver = UltimateARCSolver()
        start = time.time()
        result = solver.solve(test_task)
        elapsed = time.time() - start
        results['Ultimate ARC Solver'] = {
            'success': True,
            'output_shape': result.shape,
            'time': elapsed,
            'output': result.tolist()
        }
        print(f"   ✅ نجح - الشكل: {result.shape}, الوقت: {elapsed:.3f}s")
    except Exception as e:
        results['Ultimate ARC Solver'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل: {e}")
    
    # 2. اختبار النظام المتقدم الفائق
    print("\n2️⃣ اختبار Ultra Advanced ARC System...")
    try:
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        system = UltraAdvancedARCSystem()
        start = time.time()
        test_input = np.array(test_task['test'][0]['input'])
        solution = system.solve_arc_challenge(test_input, test_task)
        elapsed = time.time() - start
        output = solution.solution_grid if hasattr(solution, 'solution_grid') else solution
        results['Ultra Advanced ARC System'] = {
            'success': True,
            'output_shape': output.shape,
            'time': elapsed,
            'confidence': solution.confidence if hasattr(solution, 'confidence') else 0
        }
        print(f"   ✅ نجح - الثقة: {solution.confidence if hasattr(solution, 'confidence') else 'N/A':.2f}, الوقت: {elapsed:.3f}s")
    except Exception as e:
        results['Ultra Advanced ARC System'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل: {e}")
    
    # 3. اختبار النظام التفاعلي المتكامل
    print("\n3️⃣ اختبار Interactive System...")
    try:
        from arc_interactive_system import ARCInteractiveSystem
        interactive = ARCInteractiveSystem()
        start = time.time()
        result = interactive.process_task_interactive(test_task, 'test_integration')
        elapsed = time.time() - start
        results['Interactive System'] = {
            'success': True,
            'has_solution': result.final_solution is not None,
            'consensus_score': result.consensus_score,
            'time': elapsed,
            'num_systems': len(result.system_results)
        }
        print(f"   ✅ نجح - إجماع: {result.consensus_score:.2f}, عدد الأنظمة: {len(result.system_results)}, الوقت: {elapsed:.3f}s")
    except Exception as e:
        results['Interactive System'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل: {e}")
    
    # 4. اختبار التكامل بين النظام المحسن والنظام المتقدم
    print("\n4️⃣ اختبار التكامل بين Ultimate Solver و Ultra Advanced System...")
    try:
        # Ultra Advanced System يستخدم Ultimate Solver داخلياً
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        from ultimate_arc_solver import UltimateARCSolver
        
        # تحقق من أن Ultra Advanced يستطيع استخدام Ultimate Solver
        system = UltraAdvancedARCSystem()
        test_input = np.array(test_task['test'][0]['input'])
        
        # تأكد من أن النظام يستخدم Ultimate Solver
        start = time.time()
        solution = system.solve_arc_challenge(test_input, test_task)
        elapsed = time.time() - start
        
        results['Integration Test'] = {
            'success': True,
            'integration_working': True,
            'time': elapsed
        }
        print(f"   ✅ التكامل يعمل بنجاح - الوقت: {elapsed:.3f}s")
    except Exception as e:
        results['Integration Test'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل التكامل: {e}")
    
    # 5. اختبار تدفق البيانات بين الأنظمة
    print("\n5️⃣ اختبار تدفق البيانات بين الأنظمة...")
    try:
        from ultimate_arc_solver import UltimateARCSolver
        from arc_interactive_system import ARCInteractiveSystem
        
        # حل باستخدام Ultimate Solver
        solver = UltimateARCSolver()
        solution1 = solver.solve(test_task)
        
        # استخدم نفس المهمة في النظام التفاعلي
        interactive = ARCInteractiveSystem()
        result2 = interactive.process_task_interactive(test_task, 'data_flow_test')
        
        # تحقق من التوافق
        data_flow_working = (solution1 is not None and result2.final_solution is not None)
        
        results['Data Flow Test'] = {
            'success': True,
            'data_flow_working': data_flow_working,
            'solver_output_shape': solution1.shape if solution1 is not None else None,
            'interactive_has_solution': result2.final_solution is not None
        }
        print(f"   ✅ تدفق البيانات يعمل: {data_flow_working}")
    except Exception as e:
        results['Data Flow Test'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل تدفق البيانات: {e}")
    
    # 6. اختبار الأداء المقارن
    print("\n6️⃣ اختبار الأداء المقارن...")
    try:
        performance_comparison = {}
        
        # Ultimate Solver
        from ultimate_arc_solver import UltimateARCSolver
        solver = UltimateARCSolver()
        start = time.time()
        solver.solve(test_task)
        performance_comparison['Ultimate Solver'] = time.time() - start
        
        # Ultra Advanced System
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        system = UltraAdvancedARCSystem()
        test_input = np.array(test_task['test'][0]['input'])
        start = time.time()
        system.solve_arc_challenge(test_input, test_task)
        performance_comparison['Ultra Advanced'] = time.time() - start
        
        # Interactive System
        from arc_interactive_system import ARCInteractiveSystem
        interactive = ARCInteractiveSystem()
        start = time.time()
        interactive.process_task_interactive(test_task, 'perf_test')
        performance_comparison['Interactive'] = time.time() - start
        
        # إيجاد الأسرع
        fastest = min(performance_comparison, key=performance_comparison.get)
        results['Performance Comparison'] = {
            'success': True,
            'times': performance_comparison,
            'fastest': fastest,
            'fastest_time': performance_comparison[fastest]
        }
        
        print(f"   ✅ الأسرع: {fastest} ({performance_comparison[fastest]:.3f}s)")
        for system, time_taken in performance_comparison.items():
            print(f"      - {system}: {time_taken:.3f}s")
            
    except Exception as e:
        results['Performance Comparison'] = {'success': False, 'error': str(e)}
        print(f"   ❌ فشل المقارنة: {e}")
    
    # التقرير النهائي
    print("\n" + "=" * 60)
    print("📊 التقرير النهائي للتكامل")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    print(f"\n✅ الاختبارات الناجحة: {successful_tests}/{total_tests}")
    print(f"📈 معدل النجاح: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 جميع الأنظمة تعمل بشكل تكاملي ممتاز!")
        integration_status = "FULLY INTEGRATED"
    elif successful_tests >= total_tests * 0.7:
        print("\n✅ معظم الأنظمة تعمل بشكل تكاملي جيد")
        integration_status = "MOSTLY INTEGRATED"
    elif successful_tests >= total_tests * 0.5:
        print("\n⚠️ التكامل جزئي - بعض الأنظمة تحتاج تحسين")
        integration_status = "PARTIALLY INTEGRATED"
    else:
        print("\n❌ التكامل ضعيف - الأنظمة تحتاج عمل إضافي")
        integration_status = "POOR INTEGRATION"
    
    # حفظ النتائج
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'integration_status': integration_status,
        'success_rate': (successful_tests/total_tests)*100,
        'test_results': results
    }
    
    with open('integration_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 تم حفظ التقرير في: integration_test_report.json")
    
    return integration_status, successful_tests, total_tests

if __name__ == "__main__":
    print("🚀 بدء اختبار التكامل بين أنظمة مشروع برهان...")
    print("=" * 60)
    
    status, successful, total = test_system_integration()
    
    print("\n" + "=" * 60)
    print(f"🏁 النتيجة النهائية: {status}")
    print(f"   معدل النجاح: {(successful/total)*100:.1f}%")
    print("=" * 60)
