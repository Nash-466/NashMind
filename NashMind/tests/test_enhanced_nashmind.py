#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار NashMind المحسن مع التعلم الحقيقي وحل ARC
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aces_system import ACES
import time

def test_enhanced_nashmind():
    """اختبار شامل للنظام المحسن"""
    
    print("🚀 اختبار NashMind المحسن مع التعلم الحقيقي وحل ARC")
    print("="*80)
    
    # تهيئة النظام
    aces = ACES()
    
    print("\n📊 حالة النظام الأولية:")
    initial_stats = aces.get_enhanced_system_stats()
    print(f"   مستوى التعلم: {initial_stats['true_learning']['overall_learning_level']:.3f}")
    print(f"   مستوى التكامل: {initial_stats['integration_level']:.3f}")
    print(f"   الذكاء الإجمالي: {initial_stats['overall_intelligence']:.3f}")
    
    # اختبار 1: التعلم الحقيقي
    print("\n🧠 اختبار 1: التعلم الحقيقي")
    print("-" * 50)
    
    experiences = [
        "الرياضيات تساعد في حل المشاكل المعقدة",
        "الأنماط المتكررة يمكن التنبؤ بها",
        "التعلم من الأخطاء يحسن الأداء",
        "الإبداع يأتي من ربط المفاهيم المختلفة"
    ]
    
    for i, experience in enumerate(experiences, 1):
        print(f"\n   تجربة {i}: {experience}")
        result = aces.real_learning_from_experience(experience, f"test_experience_{i}")
        print(f"   ✅ تم التعلم - أنماط جديدة: {result['patterns_discovered']}")
        print(f"   📈 نمو التعلم: {result['learning_growth']:.3f}")
    
    # اختبار 2: حل مسائل ARC
    print("\n🧩 اختبار 2: حل مسائل ARC")
    print("-" * 50)
    
    # مثال ARC بسيط
    training_examples = [
        {
            "input": [[1, 0], [0, 1]],
            "output": [[0, 1], [1, 0]]
        },
        {
            "input": [[2, 0], [0, 2]],
            "output": [[0, 2], [2, 0]]
        }
    ]
    
    test_input = [[3, 0], [0, 3]]
    
    print(f"   أمثلة التدريب: {len(training_examples)}")
    print(f"   اختبار الدخل: {test_input}")
    
    arc_solution = aces.solve_arc_problem(training_examples, test_input)
    
    print(f"   ✅ الحل المتوقع: {arc_solution['predicted_output']}")
    print(f"   🎯 الثقة: {arc_solution['confidence']:.3f}")
    print(f"   🔧 الاستراتيجية: {arc_solution['strategy_used']}")
    
    # اختبار 3: حل المشاكل المحسن
    print("\n🎯 اختبار 3: حل المشاكل المحسن")
    print("-" * 50)
    
    problems = [
        "كيف يمكن تحسين كفاءة الطاقة في المباني؟",
        "ما هي أفضل طريقة لتعلم لغة جديدة؟",
        "كيف نحل مشكلة التلوث البيئي؟"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n   مشكلة {i}: {problem}")
        solution = aces.enhanced_problem_solving(problem)
        print(f"   ✅ الثقة في الحل: {solution['confidence']:.3f}")
        print(f"   🔍 أنماط قابلة للتطبيق: {len(solution['applicable_patterns'])}")
        print(f"   🧠 نهج الحل: {solution['solution_approach']}")
    
    # اختبار 4: التفاعل مع المستخدم المحسن
    print("\n💬 اختبار 4: التفاعل مع المستخدم المحسن")
    print("-" * 50)
    
    user_commands = [
        "حالة النظام",
        "إحصائيات محسنة",
        "تعلم من: الذكاء الاصطناعي يتطور بسرعة",
        "حل مسألة ARC",
        "كيف يمكنني تحسين مهاراتي في البرمجة؟"
    ]
    
    for i, command in enumerate(user_commands, 1):
        print(f"\n   أمر {i}: {command}")
        user_input = {"type": "text", "content": command}
        response = aces.process_user_input(user_input)
        print(f"   📝 الاستجابة: {response['content'][:100]}...")
        
        # معالجة الرسائل الداخلية
        aces.communication_manager.process_internal_messages()
        time.sleep(0.1)  # وقت قصير للمعالجة
    
    # النتائج النهائية
    print("\n📊 النتائج النهائية:")
    print("="*80)
    
    final_stats = aces.get_enhanced_system_stats()
    
    print(f"🧠 التعلم الحقيقي:")
    print(f"   📚 إجمالي التجارب: {final_stats['true_learning']['total_experiences']}")
    print(f"   🔍 الأنماط المكتشفة: {final_stats['true_learning']['total_patterns']}")
    print(f"   🔗 الروابط المفاهيمية: {final_stats['true_learning']['concept_connections']}")
    print(f"   📈 مستوى التعلم: {final_stats['true_learning']['overall_learning_level']:.3f}")
    
    print(f"\n🧩 حل ARC:")
    print(f"   🎯 المسائل المحاولة: {final_stats['arc_solving']['performance_stats']['problems_attempted']}")
    print(f"   ✅ المسائل المحلولة: {final_stats['arc_solving']['performance_stats']['problems_solved']}")
    print(f"   📊 معدل النجاح: {final_stats['arc_solving']['success_rate']:.3f}")
    print(f"   🔧 الاستراتيجيات المتعلمة: {len(final_stats['arc_solving']['learned_strategies'])}")
    
    print(f"\n🎯 الأداء الإجمالي:")
    print(f"   🔗 مستوى التكامل: {final_stats['integration_level']:.3f}")
    print(f"   🧠 الذكاء الإجمالي: {final_stats['overall_intelligence']:.3f}")
    
    # مقارنة التحسن
    learning_improvement = (final_stats['true_learning']['overall_learning_level'] - 
                          initial_stats['true_learning']['overall_learning_level'])
    intelligence_improvement = (final_stats['overall_intelligence'] - 
                              initial_stats['overall_intelligence'])
    
    print(f"\n📈 التحسن المحقق:")
    print(f"   🧠 تحسن التعلم: +{learning_improvement:.3f}")
    print(f"   🎯 تحسن الذكاء: +{intelligence_improvement:.3f}")
    
    if learning_improvement > 0 or intelligence_improvement > 0:
        print("   ✅ النظام يتعلم ويتحسن فعلاً!")
    else:
        print("   ⚠️ لم يتم رصد تحسن واضح")
    
    print("\n🎊 انتهى اختبار NashMind المحسن!")
    print("="*80)
    
    return {
        "initial_stats": initial_stats,
        "final_stats": final_stats,
        "learning_improvement": learning_improvement,
        "intelligence_improvement": intelligence_improvement,
        "test_passed": learning_improvement > 0 or intelligence_improvement > 0
    }

def test_specific_arc_problems():
    """اختبار مسائل ARC محددة"""
    
    print("\n🧩 اختبار مسائل ARC محددة")
    print("="*50)
    
    aces = ACES()
    
    # مسألة ARC: نمط الدوران
    print("\n🔄 مسألة 1: نمط الدوران")
    training_rotation = [
        {
            "input": [[1, 0], [0, 0]],
            "output": [[0, 0], [1, 0]]
        },
        {
            "input": [[0, 1], [0, 0]],
            "output": [[0, 0], [0, 1]]
        }
    ]
    
    test_rotation = [[1, 1], [0, 0]]
    solution_rotation = aces.solve_arc_problem(training_rotation, test_rotation)
    
    print(f"   الدخل: {test_rotation}")
    print(f"   الحل: {solution_rotation['predicted_output']}")
    print(f"   الثقة: {solution_rotation['confidence']:.3f}")
    
    # مسألة ARC: تغيير الألوان
    print("\n🎨 مسألة 2: تغيير الألوان")
    training_colors = [
        {
            "input": [[1, 2], [2, 1]],
            "output": [[2, 1], [1, 2]]
        },
        {
            "input": [[3, 4], [4, 3]],
            "output": [[4, 3], [3, 4]]
        }
    ]
    
    test_colors = [[5, 6], [6, 5]]
    solution_colors = aces.solve_arc_problem(training_colors, test_colors)
    
    print(f"   الدخل: {test_colors}")
    print(f"   الحل: {solution_colors['predicted_output']}")
    print(f"   الثقة: {solution_colors['confidence']:.3f}")
    
    return {
        "rotation_solution": solution_rotation,
        "colors_solution": solution_colors
    }

def demonstrate_real_learning():
    """عرض التعلم الحقيقي"""
    
    print("\n🧠 عرض التعلم الحقيقي")
    print("="*50)
    
    aces = ACES()
    
    # سلسلة من التجارب المترابطة
    learning_sequence = [
        "الماء يغلي عند 100 درجة مئوية",
        "الحرارة تؤثر على حالة المادة",
        "البخار هو الماء في الحالة الغازية",
        "التبريد يحول البخار إلى ماء",
        "دورة الماء في الطبيعة تعتمد على التبخر والتكثف"
    ]
    
    print("📚 سلسلة التعلم:")
    
    for i, experience in enumerate(learning_sequence, 1):
        print(f"\n   خطوة {i}: {experience}")
        result = aces.real_learning_from_experience(experience, "physics_learning")
        
        # عرض الروابط المفاهيمية الجديدة
        learning_stats = aces.true_learning_engine.get_learning_stats()
        print(f"   🔗 الروابط المفاهيمية: {learning_stats['concept_connections']}")
        print(f"   📈 مستوى التعلم: {learning_stats['overall_learning_level']:.3f}")
    
    # اختبار تطبيق المعرفة المكتسبة
    print(f"\n🎯 اختبار تطبيق المعرفة:")
    test_question = "ماذا يحدث عندما نسخن الجليد؟"
    solution = aces.enhanced_problem_solving(test_question)
    
    print(f"   السؤال: {test_question}")
    print(f"   الثقة في الإجابة: {solution['confidence']:.3f}")
    print(f"   الأنماط المطبقة: {len(solution['applicable_patterns'])}")
    
    return solution

if __name__ == "__main__":
    # تشغيل جميع الاختبارات
    print("🎯 بدء اختبارات NashMind المحسن")
    print("="*80)
    
    # الاختبار الشامل
    main_results = test_enhanced_nashmind()
    
    # اختبار ARC محدد
    arc_results = test_specific_arc_problems()
    
    # عرض التعلم الحقيقي
    learning_demo = demonstrate_real_learning()
    
    # النتيجة النهائية
    print(f"\n🏆 النتيجة النهائية:")
    print(f"   ✅ الاختبار الرئيسي: {'نجح' if main_results['test_passed'] else 'فشل'}")
    print(f"   🧩 حل ARC: {arc_results['rotation_solution']['confidence']:.2f} ثقة")
    print(f"   🧠 التعلم الحقيقي: {learning_demo['confidence']:.2f} ثقة")
    
    if main_results['test_passed']:
        print("\n🎊 NashMind المحسن يعمل بنجاح مع التعلم الحقيقي وحل ARC!")
    else:
        print("\n⚠️ يحتاج NashMind المحسن إلى مزيد من التطوير")
    
    print("="*80)
