from __future__ import annotations
#!/usr/bin/env python3
"""
تشغيل مباشر للحلقة التلقائية
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
import logging

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*80)
    print("بدء التدريب التلقائي المباشر")
    print("="*80 + "\n")
    
    # استيراد النظام
    from automated_training_loop import AutomatedTrainingLoop
    
    print("إنشاء نظام التدريب...")
    loop = AutomatedTrainingLoop()
    
    print(f"البيانات المحملة:")
    print(f"  - مهام التدريب: {len(loop.training_data)}")
    print(f"  - مهام التقييم: {len(loop.evaluation_data)}")
    print(f"  - الأنظمة المحملة: {len(loop.orchestrator.systems)}")
    
    if not loop.training_data:
        print("✗ لا توجد بيانات تدريب!")
        return
        
    print("\nبدء الحلقة التلقائية...")
    print("-" * 60)
    
    max_iterations = 10  # نبدأ بـ 10 دورات فقط
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 الدورة {iteration}/{max_iterations}")
        print("=" * 40)
        
        try:
            # دورة التدريب
            print("📚 بدء التدريب...")
            start_time = time.time()
            
            train_accuracy, train_results = loop.train_iteration()
            
            train_time = time.time() - start_time
            print(f"✓ انتهى التدريب في {train_time:.1f} ثانية")
            print(f"  الدقة: {train_accuracy:.2%}")
            
            # التقييم
            if loop.evaluation_data:
                print("\n📊 بدء التقييم...")
                eval_start = time.time()
                
                eval_accuracy, eval_results = loop.evaluate()
                
                eval_time = time.time() - eval_start
                print(f"✓ انتهى التقييم في {eval_time:.1f} ثانية")
                print(f"  الدقة: {eval_accuracy:.2%}")
            else:
                print("⚠ لا توجد بيانات تقييم")
                eval_accuracy = train_accuracy
            
            # تحليل الأخطاء
            print("\n🔍 تحليل الأخطاء...")
            loop.analyze_failures(eval_results if loop.evaluation_data else train_results)
            
            # تطبيق التحسينات
            print("🔧 تطبيق التحسينات...")
            loop.apply_improvements()
            
            # ملخص الدورة
            print(f"\n📈 ملخص الدورة {iteration}:")
            print(f"  - دقة التدريب: {train_accuracy:.2%}")
            print(f"  - دقة التقييم: {eval_accuracy:.2%}")
            print(f"  - طول DSL: {loop.orchestrator.dsl_generator.max_length}")
            
            # التحقق من الهدف
            if eval_accuracy >= 0.98:  # 98% أو أكثر
                print(f"\n🎉 تم تحقيق دقة عالية: {eval_accuracy:.2%}!")
                
                # نستمر لدورة أخرى للتأكد من الاستقرار
                if iteration > 3:
                    print("✅ الأداء مستقر، انتهى التدريب بنجاح!")
                    break
            
            # حفظ التقدم
            progress = {
                'iteration': iteration,
                'train_accuracy': train_accuracy,
                'eval_accuracy': eval_accuracy,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f'progress_iter_{iteration}.json', 'w') as f:
                json.dump(progress, f, indent=2)
            
        except KeyboardInterrupt:
            print("\n⚠ تم إيقاف التدريب من قبل المستخدم")
            break
        except Exception as e:
            print(f"\n✗ خطأ في الدورة {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # التقرير النهائي
    print("\n" + "="*80)
    print("التقرير النهائي")
    print("="*80)
    print(f"عدد الدورات المنفذة: {iteration}")
    print(f"الدقة النهائية: {eval_accuracy:.2%}")
    
    # إحصائيات الأنظمة
    print("\nأداء الأنظمة:")
    for system_name, perf in loop.orchestrator.system_performance.items():
        if perf['total'] > 0:
            success_rate = perf['success'] / perf['total']
            print(f"  - {system_name}: {success_rate:.2%} ({perf['success']}/{perf['total']})")
    
    print("\n✅ انتهى التدريب التلقائي")

if __name__ == "__main__":
    main()
