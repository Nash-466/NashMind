from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار مباشر لنظام NashMind ACES
نسأله سؤال جديد ونراقب كيف يتعامل معه
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_aces_with_new_question():
    """اختبار النظام بسؤال جديد"""
    
    print("🧠 بدء اختبار نظام NashMind ACES")
    print("=" * 50)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام...")
    aces = ACES()
    
    print("\n" + "=" * 50)
    print("📊 حالة النظام قبل السؤال:")
    initial_state = aces.get_system_status()
    print(f"📈 درجة الأداء: {initial_state['overall_performance_score']:.3f}")
    print(f"🧠 عدد النماذج العقلية: {len(aces.mentality_simulator.mental_models_library)}")
    print(f"🏗️ عدد البنى المعرفية: {len(aces.architecture_developer.developed_architectures)}")
    print(f"🌟 عدد نماذج الذات: {len(aces.existential_learner.existential_memory.get_self_models())}")
    print(f"💡 عدد الفهوم الحدسية: {len(aces.intuitive_generator.integrated_understandings)}")
    
    # السؤال الجديد للاختبار
    test_question = "كيف يمكن للذكاء الاصطناعي أن يساعد في اكتشاف علاج للسرطان؟"
    
    print("\n" + "=" * 50)
    print(f"❓ السؤال الاختباري: {test_question}")
    print("=" * 50)
    
    # قياس الوقت
    start_time = time.time()
    
    # إرسال السؤال للنظام
    user_input = {"type": "text", "content": test_question}
    
    print("🚀 إرسال السؤال للنظام...")
    response = aces.process_user_input(user_input)
    
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ وقت المعالجة: {processing_time:.2f} ثانية")
    
    # عرض الاستجابة
    print("\n" + "=" * 50)
    print("🤖 استجابة النظام:")
    print("=" * 50)
    print(response.get("content", "لا توجد استجابة"))
    
    # معالجة الرسائل الداخلية
    print("\n🔄 معالجة الرسائل الداخلية...")
    aces.communication_manager.process_internal_messages()
    aces.communication_manager.process_internal_messages()  # معالجة إضافية للردود
    
    # حالة النظام بعد السؤال
    print("\n" + "=" * 50)
    print("📊 حالة النظام بعد السؤال:")
    final_state = aces.get_system_status()
    print(f"📈 درجة الأداء: {final_state['overall_performance_score']:.3f}")
    print(f"🧠 عدد النماذج العقلية: {len(aces.mentality_simulator.mental_models_library)}")
    print(f"🏗️ عدد البنى المعرفية: {len(aces.architecture_developer.developed_architectures)}")
    print(f"🌟 عدد نماذج الذات: {len(aces.existential_learner.existential_memory.get_self_models())}")
    print(f"💡 عدد الفهوم الحدسية: {len(aces.intuitive_generator.integrated_understandings)}")
    
    # حساب التطور
    print("\n" + "=" * 50)
    print("📈 التطور الحاصل:")
    performance_change = final_state['overall_performance_score'] - initial_state['overall_performance_score']
    models_added = len(aces.mentality_simulator.mental_models_library)
    architectures_added = len(aces.architecture_developer.developed_architectures)
    
    print(f"🎯 تحسن الأداء: {performance_change:+.3f}")
    print(f"🧠 نماذج عقلية جديدة: +{models_added}")
    print(f"🏗️ بنى معرفية جديدة: +{architectures_added}")
    
    # اختبار سؤال متابعة
    print("\n" + "=" * 50)
    print("🔄 اختبار سؤال متابعة...")
    
    follow_up_question = "ما هي أكبر التحديات في هذا المجال؟"
    follow_up_input = {"type": "text", "content": follow_up_question}
    
    start_time2 = time.time()
    response2 = aces.process_user_input(follow_up_input)
    processing_time2 = time.time() - start_time2
    
    print(f"❓ السؤال المتابع: {follow_up_question}")
    print(f"⏱️ وقت المعالجة: {processing_time2:.2f} ثانية")
    print("🤖 الاستجابة:")
    print(response2.get("content", "لا توجد استجابة"))
    
    # معالجة الرسائل مرة أخرى
    aces.communication_manager.process_internal_messages()
    aces.communication_manager.process_internal_messages()
    
    print("\n" + "=" * 50)
    print("✅ انتهى الاختبار!")
    print("=" * 50)
    
    return {
        "initial_performance": initial_state['overall_performance_score'],
        "final_performance": final_state['overall_performance_score'],
        "processing_time_1": processing_time,
        "processing_time_2": processing_time2,
        "models_count": len(aces.mentality_simulator.mental_models_library),
        "architectures_count": len(aces.architecture_developer.developed_architectures),
        "response_1": response.get("content", ""),
        "response_2": response2.get("content", "")
    }

if __name__ == "__main__":
    try:
        results = test_aces_with_new_question()
        print(f"\n🎉 النتائج النهائية:")
        print(f"📊 تحسن الأداء: {results['final_performance'] - results['initial_performance']:+.3f}")
        print(f"⏱️ متوسط وقت الاستجابة: {(results['processing_time_1'] + results['processing_time_2'])/2:.2f} ثانية")
        print(f"🧠 إجمالي النماذج العقلية: {results['models_count']}")
        print(f"🏗️ إجمالي البنى المعرفية: {results['architectures_count']}")
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
