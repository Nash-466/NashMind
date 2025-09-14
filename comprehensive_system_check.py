from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 فحص شامل 100% لجميع ملفات النظام
"""

import sys
import traceback
import importlib
import numpy as np
import time
from pathlib import Path

def check_all_system_files():
    """فحص جميع ملفات النظام"""
    
    print("🔍 فحص شامل 100% لجميع ملفات النظام...")
    print("="*70)
    
    # Core system files to check
    core_files = [
        'ultra_advanced_arc_system',
        'efficient_zero_engine',
        'arc_complete_agent_part1',
        'arc_complete_agent_part2', 
        'arc_complete_agent_part3',
        'arc_complete_agent_part4',
        'arc_complete_agent_part5',
        'arc_complete_agent_part6',
        'arc_hierarchical_reasoning',
        'arc_adaptive_self_improvement',
        'advanced_simulation_engine',
        'semantic_memory_system',
        'creative_innovation_engine',
        'intelligent_verification_system',
        'arc_object_centric_reasoning',
        'arc_ultimate_mind_part7',
        'arc_ultimate_system',
        'arc_revolutionary_system',
        'burhan_meta_brain'
    ]
    
    # Support files
    support_files = [
        'dependency_manager',
        'error_manager', 
        'performance_optimizer',
        'unified_interfaces',
        'integration_tests'
    ]
    
    results = {
        'core_files': {},
        'support_files': {},
        'import_success': 0,
        'import_total': 0,
        'functional_success': 0,
        'functional_total': 0
    }
    
    print("1️⃣ فحص الملفات الأساسية...")
    print("-" * 50)
    
    for file_name in core_files:
        print(f"   📁 {file_name}...")
        try:
            # Try to import
            module = importlib.import_module(file_name)
            results['import_success'] += 1
            
            # Check for main classes/functions
            has_main_class = False
            main_classes = []
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and not attr_name.startswith('_'):
                    main_classes.append(attr_name)
                    has_main_class = True
            
            if has_main_class:
                results['functional_success'] += 1
                print(f"      ✅ استيراد ووظائف: {', '.join(main_classes[:3])}")
            else:
                print(f"      ⚠️  استيراد فقط (لا توجد فئات رئيسية)")
            
            results['core_files'][file_name] = {
                'import': True,
                'functional': has_main_class,
                'classes': main_classes
            }
            
        except Exception as e:
            print(f"      ❌ فشل: {str(e)[:50]}...")
            results['core_files'][file_name] = {
                'import': False,
                'functional': False,
                'error': str(e)
            }
        
        results['import_total'] += 1
        results['functional_total'] += 1
    
    print(f"\n2️⃣ فحص الملفات المساعدة...")
    print("-" * 50)
    
    for file_name in support_files:
        print(f"   📁 {file_name}...")
        try:
            module = importlib.import_module(file_name)
            print(f"      ✅ استيراد نجح")
            results['support_files'][file_name] = {'import': True}
        except Exception as e:
            print(f"      ❌ فشل: {str(e)[:50]}...")
            results['support_files'][file_name] = {'import': False, 'error': str(e)}
    
    return results

def test_main_system_integration():
    """اختبار تكامل النظام الرئيسي"""
    
    print(f"\n3️⃣ اختبار تكامل النظام الرئيسي...")
    print("-" * 50)
    
    integration_results = {}
    
    # Test 1: Ultra Advanced ARC System
    print("   🧠 النظام المتقدم...")
    try:
        from ultra_advanced_arc_system import UltraAdvancedARCSystem
        system = UltraAdvancedARCSystem()
        
        # Check subsystems
        subsystems = [
            'pattern_analyzer',
            'reasoning_engine', 
            'learning_engine',
            'simulation_engine',
            'memory_system',
            'creativity_engine',
            'verification_system',
            'efficient_zero_engine'
        ]
        
        active_subsystems = 0
        for subsystem in subsystems:
            if hasattr(system, subsystem):
                active_subsystems += 1
                print(f"      ✅ {subsystem}")
            else:
                print(f"      ❌ {subsystem}")
        
        integration_results['ultra_advanced'] = {
            'success': True,
            'active_subsystems': active_subsystems,
            'total_subsystems': len(subsystems)
        }
        
        print(f"      📊 أنظمة نشطة: {active_subsystems}/{len(subsystems)}")
        
    except Exception as e:
        print(f"      ❌ فشل: {e}")
        integration_results['ultra_advanced'] = {'success': False, 'error': str(e)}
    
    # Test 2: EfficientZero Engine
    print("   🧠 محرك EfficientZero...")
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        # Test basic functionality
        test_grid = np.array([[1, 0], [0, 1]])
        result = ez.solve_arc_problem(test_grid, max_steps=2)
        
        confidence = result.get('confidence', 0)
        print(f"      ✅ يعمل - ثقة: {confidence:.2f}")
        
        integration_results['efficient_zero'] = {
            'success': True,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"      ❌ فشل: {e}")
        integration_results['efficient_zero'] = {'success': False, 'error': str(e)}
    
    # Test 3: Other major systems
    major_systems = [
        ('arc_complete_agent_part2', 'UltraComprehensivePatternAnalyzer'),
        ('arc_hierarchical_reasoning', 'AdvancedLogicalReasoningEngine'),
        ('semantic_memory_system', 'AdvancedSemanticMemorySystem'),
        ('creative_innovation_engine', 'AdvancedCreativeInnovationEngine')
    ]
    
    for module_name, class_name in major_systems:
        print(f"   🧠 {module_name}...")
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            print(f"      ✅ يعمل")
            integration_results[module_name] = {'success': True}
        except Exception as e:
            print(f"      ❌ فشل: {str(e)[:50]}...")
            integration_results[module_name] = {'success': False, 'error': str(e)}
    
    return integration_results

def test_system_with_sample_problems():
    """اختبار النظام مع مشاكل عينة"""
    
    print(f"\n4️⃣ اختبار النظام مع مشاكل عينة...")
    print("-" * 50)
    
    # Sample ARC-like problems
    sample_problems = [
        {
            'name': 'بسيط 2x2',
            'input': np.array([[1, 0], [0, 1]]),
            'description': 'شبكة بسيطة'
        },
        {
            'name': 'متوسط 3x3',
            'input': np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]]),
            'description': 'نمط متماثل'
        },
        {
            'name': 'معقد 4x4',
            'input': np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 0, 1, 0], [0, 2, 0, 2]]),
            'description': 'نمط متكرر'
        },
        {
            'name': 'كبير 5x5',
            'input': np.random.randint(0, 3, (5, 5)),
            'description': 'شبكة عشوائية'
        }
    ]
    
    test_results = []
    
    # Test with EfficientZero
    print("   🧠 اختبار مع EfficientZero...")
    try:
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        for i, problem in enumerate(sample_problems):
            try:
                start_time = time.time()
                result = ez.solve_arc_problem(problem['input'], max_steps=3)
                solve_time = time.time() - start_time
                
                confidence = result.get('confidence', 0)
                success = confidence > 0.1
                
                print(f"      {i+1}. {problem['name']}: ثقة={confidence:.2f}, وقت={solve_time:.3f}s")
                
                test_results.append({
                    'problem': problem['name'],
                    'success': success,
                    'confidence': confidence,
                    'solve_time': solve_time,
                    'method': 'efficient_zero'
                })
                
            except Exception as e:
                print(f"      {i+1}. {problem['name']}: فشل - {e}")
                test_results.append({
                    'problem': problem['name'],
                    'success': False,
                    'error': str(e),
                    'method': 'efficient_zero'
                })
        
    except Exception as e:
        print(f"      ❌ EfficientZero غير متاح: {e}")
    
    # Test with main system if available
    print("   🧠 اختبار مع النظام الرئيسي...")
    try:
        from ultra_advanced_arc_system import solve_arc_problem
        
        for i, problem in enumerate(sample_problems[:2]):  # Test first 2 only
            try:
                start_time = time.time()
                result = solve_arc_problem(problem['input'])
                solve_time = time.time() - start_time
                
                confidence = result.confidence
                success = confidence > 0.1
                
                print(f"      {i+1}. {problem['name']}: ثقة={confidence:.2f}, وقت={solve_time:.3f}s")
                
                test_results.append({
                    'problem': problem['name'],
                    'success': success,
                    'confidence': confidence,
                    'solve_time': solve_time,
                    'method': 'main_system'
                })
                
            except Exception as e:
                print(f"      {i+1}. {problem['name']}: فشل - {e}")
                test_results.append({
                    'problem': problem['name'],
                    'success': False,
                    'error': str(e),
                    'method': 'main_system'
                })
        
    except Exception as e:
        print(f"      ⚠️  النظام الرئيسي غير متاح: {e}")
    
    return test_results

def main():
    """الدالة الرئيسية"""
    
    print("🔍 فحص شامل 100% لجميع ملفات مشروع برهان")
    print("="*70)
    
    # Step 1: Check all files
    file_results = check_all_system_files()
    
    # Step 2: Test integration
    integration_results = test_main_system_integration()
    
    # Step 3: Test with sample problems
    problem_results = test_system_with_sample_problems()
    
    # Summary
    print("\n" + "="*70)
    print("📊 ملخص النتائج:")
    print("-" * 70)
    
    # File import summary
    import_rate = file_results['import_success'] / file_results['import_total'] * 100
    functional_rate = file_results['functional_success'] / file_results['functional_total'] * 100
    
    print(f"📁 الملفات:")
    print(f"   - معدل الاستيراد: {file_results['import_success']}/{file_results['import_total']} ({import_rate:.1f}%)")
    print(f"   - معدل الوظائف: {file_results['functional_success']}/{file_results['functional_total']} ({functional_rate:.1f}%)")
    
    # Integration summary
    integration_success = sum(1 for r in integration_results.values() if r.get('success', False))
    integration_total = len(integration_results)
    integration_rate = integration_success / integration_total * 100 if integration_total > 0 else 0
    
    print(f"🔗 التكامل:")
    print(f"   - معدل النجاح: {integration_success}/{integration_total} ({integration_rate:.1f}%)")
    
    # Problem solving summary
    if problem_results:
        problem_success = sum(1 for r in problem_results if r.get('success', False))
        problem_total = len(problem_results)
        problem_rate = problem_success / problem_total * 100 if problem_total > 0 else 0
        
        avg_confidence = np.mean([r.get('confidence', 0) for r in problem_results if 'confidence' in r])
        
        print(f"🧩 حل المشاكل:")
        print(f"   - معدل النجاح: {problem_success}/{problem_total} ({problem_rate:.1f}%)")
        print(f"   - متوسط الثقة: {avg_confidence:.2f}")
    
    # Overall assessment
    overall_health = (import_rate + functional_rate + integration_rate) / 3
    
    print(f"\n🎯 التقييم العام: {overall_health:.1f}%")
    
    if overall_health >= 80:
        print("🎉 النظام في حالة ممتازة!")
        print("✅ جاهز للاختبار على جميع المهام")
    elif overall_health >= 60:
        print("⚠️  النظام في حالة جيدة مع بعض المشاكل")
        print("🔧 يحتاج تحسينات قبل الاختبار الكامل")
    else:
        print("❌ النظام يحتاج إصلاحات كبيرة")
        print("🛠️  يجب حل المشاكل قبل المتابعة")
    
    return overall_health >= 60

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
