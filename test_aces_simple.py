from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار مبسط لنظام NashMind ACES
نسأله سؤال واحد ونرى الاستجابة
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_simple_question():
    """اختبار بسيط بسؤال واحد"""
    
    print("🧠 اختبار مبسط لنظام NashMind ACES")
    print("=" * 50)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام...")
    aces = ACES()
    
    # السؤال الاختباري
    test_question = "ما هو أفضل طريقة لتعلم البرمجة؟"
    
    print(f"\n❓ السؤال: {test_question}")
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
    
    print("\n" + "=" * 50)
    print("✅ انتهى الاختبار!")
    print("=" * 50)
    
    return response

if __name__ == "__main__":
    try:
        result = test_simple_question()
        print(f"\n🎉 تم الاختبار بنجاح!")
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
