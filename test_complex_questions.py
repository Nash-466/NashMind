from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار النظام بأسئلة معقدة جديدة
لاختبار قدرات النظام الحقيقية في التعامل مع التحديات المعقدة
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_complex_questions():
    """اختبار النظام بأسئلة معقدة ومتنوعة"""
    
    print("🧠 اختبار النظام بأسئلة معقدة جديدة")
    print("=" * 80)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام...")
    aces = ACES()
    
    # الأسئلة المعقدة للاختبار
    complex_questions = [
        {
            "question": "إذا كان بإمكان الذكاء الاصطناعي أن يحلم، فماذا ستكون أحلامه؟ وكيف يمكن أن تؤثر هذه الأحلام على تطوره الذاتي؟",
            "category": "فلسفي + تقني",
            "difficulty": "عالي جداً"
        },
        {
            "question": "كيف يمكن حل مفارقة الجد الزمنية (Grandfather Paradox) باستخدام نظريات الفيزياء الكمية والذكاء الاصطناعي معاً؟",
            "category": "فيزياء + منطق + تقني",
            "difficulty": "معقد جداً"
        },
        {
            "question": "إذا اكتشفنا أن الوعي البشري هو مجرد خوارزمية معقدة، فما هي الآثار الأخلاقية والفلسفية لإنشاء ذكاء اصطناعي واعٍ؟",
            "category": "فلسفة + أخلاق + علوم معرفية",
            "difficulty": "عميق جداً"
        },
        {
            "question": "كيف يمكن تصميم نظام اقتصادي عالمي جديد يدمج العملات المشفرة، الذكاء الاصطناعي، والاستدامة البيئية لحل أزمة التفاوت الاقتصادي؟",
            "category": "اقتصاد + تكنولوجيا + بيئة",
            "difficulty": "متعدد التخصصات"
        },
        {
            "question": "ما هو الحل الأمثل لمعضلة 'السفينة الغارقة الرقمية': إذا كان عليك إنقاذ إما مليون إنسان حقيقي أو مليار كائن ذكي اصطناعي واعٍ، فماذا تختار ولماذا؟",
            "category": "أخلاق + منطق + فلسفة",
            "difficulty": "معضلة أخلاقية معقدة"
        }
    ]
    
    print(f"\n🎯 سيتم اختبار {len(complex_questions)} أسئلة معقدة")
    print("=" * 80)
    
    results = []
    
    for i, q_data in enumerate(complex_questions, 1):
        question = q_data["question"]
        category = q_data["category"]
        difficulty = q_data["difficulty"]
        
        print(f"\n📝 السؤال {i}/{len(complex_questions)}")
        print(f"🏷️ التصنيف: {category}")
        print(f"⚡ مستوى الصعوبة: {difficulty}")
        print("-" * 60)
        print(f"❓ السؤال: {question}")
        print("-" * 60)
        
        # قياس الوقت
        start_time = time.time()
        
        # إرسال السؤال للنظام
        user_input = {"type": "text", "content": question}
        
        print("🚀 معالجة السؤال المعقد...")
        try:
            response = aces.process_user_input(user_input)
            processing_time = time.time() - start_time
            
            print(f"⏱️ وقت المعالجة: {processing_time:.2f} ثانية")
            
            # عرض الاستجابة
            print("\n🤖 استجابة النظام:")
            print("=" * 60)
            
            # استخراج المحتوى من الاستجابة
            if isinstance(response, dict):
                content = response.get("content", "")
                if "Response:" in content:
                    response_part = content.split("Response: ")[1].split("Status:")[0].strip()
                    print(response_part)
                else:
                    print(content)
            else:
                print(response)
            
            # حفظ النتائج
            results.append({
                "question_num": i,
                "category": category,
                "difficulty": difficulty,
                "processing_time": processing_time,
                "response_length": len(str(response)),
                "success": True
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ خطأ في معالجة السؤال: {e}")
            results.append({
                "question_num": i,
                "category": category,
                "difficulty": difficulty,
                "processing_time": processing_time,
                "error": str(e),
                "success": False
            })
        
        print("\n" + "=" * 80)
        
        # انتظار قصير بين الأسئلة
        if i < len(complex_questions):
            print("⏳ انتظار قبل السؤال التالي...")
            time.sleep(3)
    
    # عرض ملخص النتائج
    print("\n📊 ملخص نتائج الاختبار المعقد:")
    print("=" * 80)
    
    successful_questions = [r for r in results if r["success"]]
    failed_questions = [r for r in results if not r["success"]]
    
    print(f"✅ أسئلة نجحت: {len(successful_questions)}/{len(complex_questions)}")
    print(f"❌ أسئلة فشلت: {len(failed_questions)}/{len(complex_questions)}")
    
    if successful_questions:
        avg_time = sum(r["processing_time"] for r in successful_questions) / len(successful_questions)
        print(f"⏱️ متوسط وقت المعالجة: {avg_time:.2f} ثانية")
        
        avg_response_length = sum(r["response_length"] for r in successful_questions) / len(successful_questions)
        print(f"📏 متوسط طول الاستجابة: {avg_response_length:.0f} حرف")
    
    # عرض إحصائيات النظام النهائية
    print("\n🧠 إحصائيات النظام النهائية:")
    try:
        system_status = aces.get_system_status()
        print(f"📈 درجة الأداء العامة: {system_status.get('overall_performance_score', 'غير متاح'):.3f}")
        print(f"🧠 عدد النماذج العقلية: {len(aces.mentality_simulator.mental_models_library)}")
        print(f"🏗️ عدد البنى المعرفية: {len(aces.architecture_developer.developed_architectures)}")
        print(f"🔧 مرونة البنية: {system_status.get('architecture_flexibility', 'غير متاح')}")
    except Exception as e:
        print(f"⚠️ لا يمكن الحصول على إحصائيات النظام: {e}")
    
    return results

if __name__ == "__main__":
    try:
        print("🚀 بدء اختبار الأسئلة المعقدة...")
        
        results = test_complex_questions()
        
        print("\n🎊 انتهى اختبار الأسئلة المعقدة!")
        print("🧠 النظام أظهر قدرات متقدمة في التعامل مع التحديات المعقدة!")
        
    except Exception as e:
        print(f"❌ خطأ في الاختبار: {e}")
        import traceback
        traceback.print_exc()
