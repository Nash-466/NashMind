from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار كامل لنظام NashMind ACES
نسأله سؤال ونحصل على الإجابة الكاملة
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_complete_question():
    """اختبار كامل للحصول على إجابة شاملة"""
    
    print("🧠 اختبار كامل لنظام NashMind ACES")
    print("=" * 60)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام...")
    aces = ACES()
    
    # السؤال الاختباري
    test_question = "ما هو أفضل طريقة لتعلم البرمجة؟"
    
    print(f"\n❓ السؤال: {test_question}")
    print("=" * 60)
    
    # قياس الوقت
    start_time = time.time()
    
    # إرسال السؤال للنظام
    user_input = {"type": "text", "content": test_question}
    
    print("🚀 إرسال السؤال للنظام...")
    response = aces.process_user_input(user_input)
    
    print(f"📝 الاستجابة الأولية: {response.get('content', 'لا توجد استجابة')}")
    
    # انتظار المعالجة الكاملة
    print("\n⏳ انتظار المعالجة الكاملة...")
    time.sleep(3)  # انتظار 3 ثواني للمعالجة
    
    # معالجة الرسائل الداخلية للحصول على النتيجة النهائية
    print("🔄 معالجة الرسائل الداخلية...")
    for i in range(5):
        aces.communication_manager.process_internal_messages()
        time.sleep(1)
        print(f"   معالجة دورة {i+1}/5...")
    
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ إجمالي وقت المعالجة: {processing_time:.2f} ثانية")
    
    # محاولة الحصول على حالة النظام
    print("\n📊 حالة النظام بعد المعالجة:")
    try:
        system_status = aces.get_system_status()
        print(f"📈 درجة الأداء: {system_status.get('overall_performance_score', 'غير متاح'):.3f}")
        print(f"🧠 عدد النماذج العقلية: {len(aces.mentality_simulator.mental_models_library)}")
        print(f"🏗️ عدد البنى المعرفية: {len(aces.architecture_developer.developed_architectures)}")
    except Exception as e:
        print(f"⚠️ لا يمكن الحصول على حالة النظام: {e}")
    
    # محاولة الحصول على إجابة أكثر تفصيلاً
    print("\n" + "=" * 60)
    print("🎯 محاولة الحصول على إجابة مفصلة...")
    
    # إرسال طلب للحصول على إجابة مفصلة
    detailed_request = {"type": "text", "content": "أعطني إجابة مفصلة عن السؤال السابق"}
    detailed_response = aces.process_user_input(detailed_request)
    
    print("🤖 الاستجابة المفصلة:")
    print("=" * 60)
    print(detailed_response.get("content", "لا توجد استجابة مفصلة"))
    
    # معالجة إضافية
    time.sleep(2)
    for i in range(3):
        aces.communication_manager.process_internal_messages()
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("✅ انتهى الاختبار الكامل!")
    print("=" * 60)
    
    return {
        "initial_response": response,
        "detailed_response": detailed_response,
        "processing_time": processing_time,
        "system_working": True
    }

if __name__ == "__main__":
    try:
        result = test_complete_question()
        print(f"\n🎉 تم الاختبار الكامل بنجاح!")
        print(f"⏱️ وقت المعالجة: {result['processing_time']:.2f} ثانية")
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
