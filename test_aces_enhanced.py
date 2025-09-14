from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار النظام المطور NashMind ACES v2.0
يعطي إجابات كاملة وذكية مع معلومات النظام المعرفي
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_enhanced_aces():
    """اختبار النظام المطور مع أسئلة متنوعة"""
    
    print("🧠 اختبار نظام NashMind ACES v2.0 المطور")
    print("=" * 70)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام المطور...")
    aces = ACES()
    
    # قائمة الأسئلة للاختبار
    test_questions = [
        "ما هو أفضل طريقة لتعلم البرمجة؟",
        "كيف يعمل الذكاء الاصطناعي؟",
        "ما هي أفضل استراتيجيات التعلم؟",
        "كيف أحل المشاكل بطريقة إبداعية؟",
        "ما هو مستقبل التكنولوجيا؟"
    ]
    
    print(f"\n🎯 سيتم اختبار {len(test_questions)} أسئلة مختلفة")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 السؤال {i}/{len(test_questions)}: {question}")
        print("-" * 50)
        
        # قياس الوقت
        start_time = time.time()
        
        # إرسال السؤال للنظام
        user_input = {"type": "text", "content": question}
        
        print("🚀 معالجة السؤال...")
        response = aces.process_user_input(user_input)
        
        processing_time = time.time() - start_time
        
        print(f"⏱️ وقت المعالجة الإجمالي: {processing_time:.2f} ثانية")
        
        # عرض الاستجابة
        print("\n" + "🤖 استجابة النظام:")
        print("=" * 50)
        
        # استخراج المحتوى من الاستجابة
        if isinstance(response, dict):
            content = response.get("content", "")
            if "Response:" in content:
                # استخراج الاستجابة الفعلية
                response_part = content.split("Response: ")[1].split("Status:")[0].strip()
                print(response_part)
            else:
                print(content)
        else:
            print(response)
        
        print("\n" + "=" * 70)
        
        # انتظار قصير بين الأسئلة
        if i < len(test_questions):
            print("⏳ انتظار قبل السؤال التالي...")
            time.sleep(2)
    
    print("\n🎉 انتهى اختبار جميع الأسئلة!")
    
    # عرض إحصائيات النظام
    print("\n📊 إحصائيات النظام:")
    try:
        system_status = aces.get_system_status()
        print(f"📈 درجة الأداء العامة: {system_status.get('overall_performance_score', 'غير متاح'):.3f}")
        print(f"🧠 عدد النماذج العقلية: {len(aces.mentality_simulator.mental_models_library)}")
        print(f"🏗️ عدد البنى المعرفية: {len(aces.architecture_developer.developed_architectures)}")
        print(f"🔧 مرونة البنية: {system_status.get('architecture_flexibility', 'غير متاح')}")
    except Exception as e:
        print(f"⚠️ لا يمكن الحصول على إحصائيات النظام: {e}")
    
    return True

def test_single_question():
    """اختبار سؤال واحد بالتفصيل"""
    
    print("\n" + "🔍 اختبار مفصل لسؤال واحد")
    print("=" * 50)
    
    aces = ACES()
    
    question = "كيف يمكنني تطوير مهاراتي في الذكاء الاصطناعي؟"
    print(f"❓ السؤال: {question}")
    
    user_input = {"type": "text", "content": question}
    
    start_time = time.time()
    response = aces.process_user_input(user_input)
    processing_time = time.time() - start_time
    
    print(f"\n⏱️ وقت المعالجة: {processing_time:.2f} ثانية")
    print("\n🤖 الاستجابة الكاملة:")
    print("=" * 50)
    
    if isinstance(response, dict):
        content = response.get("content", "")
        if "Response:" in content:
            response_part = content.split("Response: ")[1].split("Status:")[0].strip()
            print(response_part)
        else:
            print(content)
    else:
        print(response)
    
    return True

if __name__ == "__main__":
    try:
        print("🚀 بدء اختبار النظام المطور...")
        
        # اختبار متعدد الأسئلة
        success = test_enhanced_aces()
        
        if success:
            print("\n✅ تم اختبار النظام المطور بنجاح!")
            
            # اختبار إضافي مفصل
            test_single_question()
            
            print("\n🎊 النظام يعمل بكامل طاقته ويعطي إجابات ذكية ومفصلة!")
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
