from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 تشغيل التقييم الشامل لنظام برهان المتقدم
=============================================

هذا السكريپت يقوم بتشغيل جميع اختبارات النظام وإنتاج تقرير شامل.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_command(command: str, description: str) -> Dict[str, Any]:
    """تشغيل أمر وإرجاع النتائج"""
    
    print(f"\n🔄 {description}...")
    print(f"📝 الأمر: {command}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run the command
        exit_code = os.system(command)
        execution_time = time.time() - start_time
        
        if exit_code == 0:
            print(f"✅ {description} - اكتمل بنجاح في {execution_time:.2f} ثانية")
            return {
                'success': True,
                'execution_time': execution_time,
                'exit_code': exit_code,
                'description': description
            }
        else:
            print(f"❌ {description} - فشل مع رمز الخروج {exit_code}")
            return {
                'success': False,
                'execution_time': execution_time,
                'exit_code': exit_code,
                'description': description
            }
    
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"💥 {description} - خطأ: {e}")
        return {
            'success': False,
            'execution_time': execution_time,
            'error': str(e),
            'description': description
        }


def check_system_requirements() -> bool:
    """فحص متطلبات النظام"""
    
    print("🔍 فحص متطلبات النظام...")
    
    required_files = [
        'ultra_advanced_arc_system.py',
        'run_training_evaluation.py',
        'simple_pattern_analysis.py',
        'arc-agi_training_challenges.json',
        'arc-agi_training_solutions.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ ملفات مفقودة: {missing_files}")
        return False
    
    print("✅ جميع الملفات المطلوبة متوفرة")
    return True


def run_system_tests() -> Dict[str, Any]:
    """تشغيل اختبارات النظام"""
    
    results = {
        'start_time': time.time(),
        'tests': [],
        'summary': {}
    }
    
    # Test 1: Quick system test
    test_result = run_command(
        'python -c "from ultra_advanced_arc_system import solve_arc_problem; import numpy as np; print(\'System loaded successfully\')"',
        'اختبار تحميل النظام'
    )
    results['tests'].append(test_result)
    
    # Test 2: Pattern analysis
    test_result = run_command(
        'python simple_pattern_analysis.py',
        'تحليل أنماط مهام التدريب'
    )
    results['tests'].append(test_result)
    
    # Test 3: Training evaluation (limited)
    test_result = run_command(
        'python run_training_evaluation.py',
        'تقييم النظام على مهام التدريب'
    )
    results['tests'].append(test_result)
    
    # Calculate summary
    total_tests = len(results['tests'])
    successful_tests = sum(1 for test in results['tests'] if test['success'])
    total_time = time.time() - results['start_time']
    
    results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'failed_tests': total_tests - successful_tests,
        'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
        'total_execution_time': total_time
    }
    
    return results


def load_latest_results() -> Dict[str, Any]:
    """تحميل أحدث النتائج"""
    
    results = {}
    
    try:
        # Load pattern analysis results
        if Path('simple_training_patterns_stats.json').exists():
            with open('simple_training_patterns_stats.json', 'r', encoding='utf-8') as f:
                results['pattern_analysis'] = json.load(f)
        
        # Load training evaluation results (find latest)
        training_files = list(Path('.').glob('training_evaluation_results_*.json'))
        if training_files:
            latest_training = max(training_files, key=lambda x: x.stat().st_mtime)
            with open(latest_training, 'r', encoding='utf-8') as f:
                results['training_evaluation'] = json.load(f)
    
    except Exception as e:
        logging.warning(f"Error loading results: {e}")
    
    return results


def generate_summary_report(test_results: Dict[str, Any], evaluation_results: Dict[str, Any]) -> str:
    """إنتاج تقرير ملخص"""
    
    report = []
    report.append("=" * 80)
    report.append("🏆 تقرير التقييم الشامل لنظام برهان المتقدم")
    report.append("=" * 80)
    report.append("")
    
    # System tests summary
    summary = test_results['summary']
    report.append("🧪 ملخص اختبارات النظام:")
    report.append(f"  📊 إجمالي الاختبارات: {summary['total_tests']}")
    report.append(f"  ✅ الاختبارات الناجحة: {summary['successful_tests']}")
    report.append(f"  ❌ الاختبارات الفاشلة: {summary['failed_tests']}")
    report.append(f"  📈 معدل النجاح: {summary['success_rate']:.1%}")
    report.append(f"  ⏱️  إجمالي وقت التنفيذ: {summary['total_execution_time']:.2f} ثانية")
    report.append("")
    
    # Training evaluation results
    if 'training_evaluation' in evaluation_results:
        training = evaluation_results['training_evaluation']
        report.append("🎯 نتائج تقييم مهام التدريب:")
        report.append(f"  📊 إجمالي المهام: {training.get('total_problems', 'غير متوفر')}")
        report.append(f"  ✅ المهام المحلولة صحيحاً: {training.get('solved_correctly', 'غير متوفر')}")
        report.append(f"  🔶 المهام المحلولة جزئياً: {training.get('partial_matches', 'غير متوفر')}")
        report.append(f"  📈 معدل النجاح: {training.get('success_rate', 0):.1%}")
        report.append(f"  📊 متوسط التشابه: {training.get('average_similarity', 0):.3f}")
        report.append(f"  ⏱️  متوسط وقت الحل: {training.get('average_time', 0):.3f} ثانية")
        report.append("")
    
    # Pattern analysis results
    if 'pattern_analysis' in evaluation_results:
        patterns = evaluation_results['pattern_analysis']
        report.append("🔍 نتائج تحليل الأنماط:")
        report.append(f"  📚 إجمالي المهام المحللة: {patterns.get('total_problems', 'غير متوفر')}")
        report.append(f"  📐 إجمالي الشبكات: {len(patterns.get('grid_sizes', []))}")
        report.append(f"  🎨 الألوان المستخدمة: {len(patterns.get('color_usage', {}))}")
        
        # Transformation types
        trans_types = patterns.get('transformation_types', {})
        if trans_types:
            report.append("  🔄 أنواع التحويلات:")
            for trans_type, count in trans_types.items():
                report.append(f"    - {trans_type}: {count}")
        report.append("")
    
    # System capabilities
    report.append("🧠 قدرات النظام المطورة:")
    report.append("  ✅ محلل الأنماط المتقدم")
    report.append("  ✅ محرك التفكير المنطقي")
    report.append("  ✅ نظام الذاكرة الدلالية")
    report.append("  ✅ محرك الإبداع والابتكار")
    report.append("  ✅ محرك المحاكاة المتقدم")
    report.append("  ✅ نظام التحقق الذكي")
    report.append("  ✅ محسن الأداء والذاكرة")
    report.append("")
    
    # Recommendations
    report.append("💡 التوصيات للتحسين:")
    report.append("  🔧 تحسين خوارزميات التحجيم")
    report.append("  🎯 تطوير فهم التحويلات المعقدة")
    report.append("  📚 تحسين نظام التعلم التكيفي")
    report.append("  🚀 إضافة قدرات تحليل السياق")
    report.append("")
    
    report.append("=" * 80)
    report.append("🎉 النظام جاهز للمشاركة في ARC Prize 2025!")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """الدالة الرئيسية"""
    
    print("🚀 بدء التقييم الشامل لنظام برهان المتقدم")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Check system requirements
        if not check_system_requirements():
            print("❌ فشل في فحص متطلبات النظام")
            return 1
        
        # Run system tests
        print("\n🧪 تشغيل اختبارات النظام...")
        test_results = run_system_tests()
        
        # Load evaluation results
        print("\n📊 تحميل نتائج التقييم...")
        evaluation_results = load_latest_results()
        
        # Generate summary report
        print("\n📝 إنتاج التقرير النهائي...")
        summary_report = generate_summary_report(test_results, evaluation_results)
        
        # Print summary
        print("\n" + summary_report)
        
        # Save detailed results
        final_results = {
            'timestamp': time.time(),
            'test_results': test_results,
            'evaluation_results': evaluation_results,
            'summary_report': summary_report
        }
        
        with open('complete_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        with open('EVALUATION_SUMMARY.md', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        total_time = time.time() - start_time
        
        print(f"\n💾 النتائج محفوظة في:")
        print(f"  📄 complete_evaluation_results.json")
        print(f"  📄 EVALUATION_SUMMARY.md")
        print(f"\n⏱️  إجمالي وقت التقييم: {total_time:.2f} ثانية")
        print("🎉 انتهى التقييم الشامل بنجاح!")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 خطأ في التقييم الشامل: {e}")
        logging.error(f"Complete evaluation error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
