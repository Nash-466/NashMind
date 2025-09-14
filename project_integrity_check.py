from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 فحص تكامل المشروع الشامل
- يتحقق من: قراءة JSON, تطابق المفاتيح, استيراد الوحدات الأساسية, فحص سريع لبيئة التشغيل
"""

import importlib
import json
import sys
from pathlib import Path

CORE_MODULES = [
    'ultra_advanced_arc_system',
    'efficient_zero_engine',
    'advanced_simulation_engine',
    'arc_adaptive_self_improvement',
    'creative_innovation_engine',
    'semantic_memory_system',
    'intelligent_verification_system',
    'performance_optimizer',
    'error_manager',
    'unified_interfaces'
]

JSON_CHALLENGES = Path('arc-agi_training_challenges.json')
JSON_SOLUTIONS = Path('arc-agi_training_solutions.json')


def check_json():
    report = {}
    try:
        with JSON_CHALLENGES.open('r', encoding='utf-8') as f:
            challenges = json.load(f)
        with JSON_SOLUTIONS.open('r', encoding='utf-8') as f:
            solutions = json.load(f)
    except Exception as e:
        return {'ok': False, 'error': f'JSON load error: {e}'}

    keys_c = set(challenges.keys())
    keys_s = set(solutions.keys())

    missing_in_s = sorted(list(keys_c - keys_s))[:20]
    missing_in_c = sorted(list(keys_s - keys_c))[:20]
    common = keys_c & keys_s

    report['ok'] = True
    report['counts'] = {
        'challenges': len(keys_c),
        'solutions': len(keys_s),
        'intersection': len(common)
    }
    report['mismatch'] = {
        'in_challenges_not_solutions_sample': missing_in_s,
        'in_solutions_not_challenges_sample': missing_in_c
    }

    # فحص بنية أول مهمة مشتركة
    sample_ok = True
    sample_issue = None
    for tid in list(common)[:5]:
        try:
            ch = challenges[tid]
            sol = solutions[tid]
            if not isinstance(ch.get('test'), list) or len(ch['test']) == 0:
                sample_ok = False
                sample_issue = f"Task {tid}: no test cases"
                break
            if not isinstance(sol, list) or len(sol) == 0:
                sample_ok = False
                sample_issue = f"Task {tid}: empty solutions list"
                break
        except Exception as e:
            sample_ok = False
            sample_issue = f"Task {tid} structure error: {e}"
            break
    report['sample_structure_ok'] = sample_ok
    report['sample_issue'] = sample_issue
    return report


def check_imports():
    results = []
    for mod in CORE_MODULES:
        try:
            importlib.invalidate_caches()
            importlib.import_module(mod)
            results.append({'module': mod, 'ok': True})
        except Exception as e:
            results.append({'module': mod, 'ok': False, 'error': str(e)})
    all_ok = all(r['ok'] for r in results)
    return {'ok': all_ok, 'details': results}


def main():
    print('🔍 بدء فحص التكامل الشامل...')
    print('='*60)

    # 1) JSON
    print('\n📦 فحص ملفات JSON...')
    json_report = check_json()
    print('   ✅' if json_report.get('ok') else '   ❌', json_report)

    # 2) Imports
    print('\n📚 فحص استيراد الوحدات الأساسية...')
    imports_report = check_imports()
    for r in imports_report['details']:
        status = '✅' if r['ok'] else '❌'
        print(f"   {status} {r['module']}{' - ' + r.get('error','') if not r['ok'] else ''}")
    print('   النتيجة:', '✅ جميع الوحدات تعمل' if imports_report['ok'] else '❌ توجد مشاكل في الاستيراد')

    # 3) بيئة بايثون
    print('\n🐍 معلومات بيئة بايثون:')
    print('   ', sys.version)

    ok = json_report.get('ok') and imports_report.get('ok')
    print('\n🎯 الخلاصة:', '✅ التكامل سليم' if ok else '⚠️ يحتاج إصلاحات')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

