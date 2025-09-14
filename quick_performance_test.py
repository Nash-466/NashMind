from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ اختبار الأداء السريع - تقييم سريع للنظام الحالي
"""

import json
import numpy as np
import time
from collections.abc import Callable
from typing import Dict, List

def load_data():
    """تحميل البيانات"""
    try:
        with open('arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open('arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        return challenges, solutions
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        return {}, {}

def solve_task(task_id: str, challenges: Dict, solutions: Dict):
    """حل مهمة واحدة"""

    challenge = challenges[task_id]
    solution = solutions[task_id]

    test_case = challenge['test'][0]
    input_grid = np.array(test_case['input'])
    expected_output = np.array(solution[0])

    # 🧠 حل باستخدام النظام العبقري المتكامل
    start_time = time.time()

    try:
        from genius_breakthrough_system import GeniusBreakthroughSystem

        # إنشاء النظام العبقري (مثيل جديد لكل مهمة للاختبار السريع)
        genius = GeniusBreakthroughSystem(verbose=False)  # إخفاء رسائل التهيئة

        task_dict = {
            'train': [{'input': ex['input'], 'output': ex['output']} for ex in challenge.get('train', [])],
            'test': [{'input': test_case['input']}]
        }

        # الحل العبقري المتكامل
        genius_result = genius.solve_with_genius(task_dict)
        solve_time = time.time() - start_time

        if genius_result.get('success', False) and genius_result.get('output') is not None:
            output_grid = np.array(genius_result['output'])
            used_engine = f"Genius-{genius_result.get('engine', 'unknown')}"
            confidence = genius_result.get('confidence', 0.5)

            # حساب التشابه
            if output_grid.shape == expected_output.shape:
                similarity = np.sum(output_grid == expected_output) / output_grid.size
            else:
                similarity = 0.0
        else:
            # fallback إلى EfficientZero إذا فشل النظام العبقري
            from efficient_zero_engine import EfficientZeroEngine
            ez = EfficientZeroEngine()
            result = ez.solve_arc_problem(input_grid, max_steps=5)

            if result.get('success', True):
                output_grid = np.array(result.get('solution_grid', input_grid))
                used_engine = 'EfficientZero-Fallback'
                confidence = result.get('confidence', 0.3)

                # حساب التشابه
                if output_grid.shape == expected_output.shape:
                    similarity = np.sum(output_grid == expected_output) / output_grid.size
                else:
                    similarity = 0.0
            else:
                return {
                    'task_id': task_id,
                    'similarity': 0.0,
                    'confidence': 0.0,
                    'solve_time': solve_time,
                    'solved_correctly': False,
                    'success': False,
                    'error': result.get('error', 'فشل في الحل'),
                    'used_engine': 'Failed'
                }

    except Exception as e:
        print(f"⚠️ خطأ في النظام العبقري: {e}")
        # fallback طوارئ إلى EfficientZero
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        result = ez.solve_arc_problem(input_grid, max_steps=5)
        solve_time = time.time() - start_time

        if result.get('success', True):
            output_grid = np.array(result.get('solution_grid', input_grid))
            used_engine = 'EfficientZero-Emergency'
            confidence = result.get('confidence', 0.2)
            # حساب التشابه
            if output_grid.shape == expected_output.shape:
                similarity = np.sum(output_grid == expected_output) / output_grid.size
            else:
                similarity = 0.0
        else:
            return {
                'task_id': task_id,
                'similarity': 0.0,
                'confidence': 0.0,
                'solve_time': solve_time,
                'solved_correctly': False,
                'success': False,
                'error': result.get('error', 'فشل في الحل'),
                'used_engine': 'Failed-Emergency'
            }

    return {
        'task_id': task_id,
        'similarity': similarity,
        'confidence': confidence,
        'solve_time': solve_time,
        'solved_correctly': similarity >= 0.99,
        'success': True,
        'used_engine': used_engine
    }

def quick_test(num_tasks: int = 30):
    """اختبار سريع"""

    print("⚡ اختبار الأداء السريع")
    print("="*40)

    # تحميل البيانات
    challenges, solutions = load_data()

    if not challenges:
        print("❌ لا توجد بيانات للاختبار")
        return

    print(f"📊 اختبار {num_tasks} مهمة...")

    # اختيار مهام للاختبار
    task_ids = list(challenges.keys())[:num_tasks]

    results = []
    solved_count = 0
    high_similarity_count = 0
    total_similarity = 0
    total_time = 0

    for i, task_id in enumerate(task_ids):
        try:
            result = solve_task(task_id, challenges, solutions)
            results.append(result)

            if result['solved_correctly']:
                solved_count += 1
                status = "✅"
            elif result['similarity'] >= 0.9:
                high_similarity_count += 1
                status = f"🎯 {result['similarity']:.3f}"
            elif result['similarity'] >= 0.7:
                status = f"📊 {result['similarity']:.3f}"
            elif result['similarity'] >= 0.3:
                status = f"📉 {result['similarity']:.3f}"
            else:
                status = f"❌ {result['similarity']:.3f}"

            total_similarity += result['similarity']
            total_time += result['solve_time']

            print(f"   {i+1:2d}. {task_id[:8]}: {status}")

            # طباعة تقدم كل 10 مهام
            if (i + 1) % 10 == 0:
                current_rate = solved_count / (i + 1)
                current_avg_sim = total_similarity / (i + 1)
                print(f"       📈 التقدم: {solved_count}/{i+1} ({current_rate:.1%}) | متوسط التشابه: {current_avg_sim:.3f}")

        except Exception as e:
            print(f"   {i+1:2d}. {task_id[:8]}: ❌ خطأ: {str(e)[:30]}...")
            results.append({
                'task_id': task_id,
                'similarity': 0.0,
                'solved_correctly': False,
                'success': False,
                'error': str(e)
            })

    # حساب الإحصائيات
    success_rate = solved_count / len(results)
    high_similarity_rate = high_similarity_count / len(results)
    avg_similarity = total_similarity / len(results)
    avg_time = total_time / len(results)

    # تحليل التوزيع
    similarity_ranges = {
        '90%+': sum(1 for r in results if r['similarity'] >= 0.9),
        '70-89%': sum(1 for r in results if 0.7 <= r['similarity'] < 0.9),
        '50-69%': sum(1 for r in results if 0.5 <= r['similarity'] < 0.7),
        '30-49%': sum(1 for r in results if 0.3 <= r['similarity'] < 0.5),
        '<30%': sum(1 for r in results if r['similarity'] < 0.3)
    }

    print(f"\n📊 النتائج النهائية:")
    print(f"✅ مهام محلولة بشكل صحيح: {solved_count}/{len(results)} ({success_rate:.1%})")
    print(f"🎯 مهام عالية التشابه (90%+): {high_similarity_count}/{len(results)} ({high_similarity_rate:.1%})")
    print(f"📈 متوسط التشابه: {avg_similarity:.3f}")
    print(f"⏱️ متوسط وقت الحل: {avg_time:.2f} ثانية")

    print(f"\n📊 توزيع التشابه:")
    for range_name, count in similarity_ranges.items():
        percentage = count / len(results) * 100
        print(f"   {range_name}: {count} مهمة ({percentage:.1f}%)")

    # أفضل النتائج
    best_results = sorted([r for r in results if r['similarity'] > 0],
                         key=lambda x: x['similarity'], reverse=True)[:10]

    print(f"\n🏆 أفضل 10 نتائج:")
    for i, result in enumerate(best_results):
        status = "✅ حُلت" if result['solved_correctly'] else f"🎯 {result['similarity']:.3f}"
        print(f"   {i+1:2d}. {result['task_id'][:8]}: {status}")

    # تحليل الأداء
    print(f"\n🔍 تحليل الأداء:")

    if success_rate >= 0.1:
        print("🎉 ممتاز! النظام يحل المهام بنجاح!")
    elif high_similarity_rate >= 0.3:
        print("🎯 جيد جداً! النظام قريب جداً من الحلول الصحيحة!")
    elif avg_similarity >= 0.5:
        print("📈 جيد! النظام يفهم الأنماط ويحتاج تحسينات دقيقة!")
    elif avg_similarity >= 0.3:
        print("📊 متوسط! النظام يحتاج تطوير في فهم الأنماط!")
    else:
        print("⚠️ ضعيف! النظام يحتاج تطوير أساسي!")

    # اقتراحات التحسين
    print(f"\n💡 اقتراحات التحسين:")

    if similarity_ranges['90%+'] > 0:
        print("   🎯 ركز على تحويل المهام عالية التشابه إلى حلول صحيحة")

    if similarity_ranges['70-89%'] > 0:
        print("   📊 حسن دقة الخوارزميات للمهام متوسطة التشابه")

    if similarity_ranges['<30%'] > similarity_ranges['90%+']:
        print("   🔧 طور الخوارزميات الأساسية لفهم الأنماط")

    if avg_time > 1.0:
        print("   ⚡ حسن سرعة الحل")

    return {
        'success_rate': success_rate,
        'high_similarity_rate': high_similarity_rate,
        'avg_similarity': avg_similarity,
        'solved_count': solved_count,
        'total_tested': len(results),
        'best_results': best_results[:5]
    }

def main():
    """الدالة الرئيسية"""

    print("🚀 بدء اختبار الأداء السريع...")

    # اختبار سريع
    results = quick_test(30)

    if results:
        print(f"\n🏁 انتهى الاختبار!")
        print(f"📊 النتيجة النهائية: {results['success_rate']:.1%} نجاح")

        if results['success_rate'] > 0:
            print("🎉 النظام يعمل ويحل المهام!")
        elif results['high_similarity_rate'] > 0.2:
            print("🎯 النظام قريب جداً من النجاح!")
        else:
            print("⚠️ النظام يحتاج مزيد من التطوير")

if __name__ == "__main__":
    main()
