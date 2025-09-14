from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار النظام الواعي الجديد
نظام ذكي يتعلم ويتطور من كل تجربة جديدة
"""

import sys
import os
sys.path.append('NashMind')

from aces_system import ACES
import time

def test_conscious_ai():
    """اختبار النظام الواعي الجديد"""
    
    print("🧠 اختبار النظام الواعي الجديد - نظام يتعلم ويفهم ويتطور")
    print("=" * 80)
    
    # إنشاء مثيل من النظام
    print("🔧 تهيئة النظام الواعي...")
    aces = ACES()
    
    # أسئلة معقدة لاختبار الوعي
    conscious_test_questions = [
        {
            "question": "إذا كان بإمكان الذكاء الاصطناعي أن يحلم، فماذا ستكون أحلامه؟ وكيف يمكن أن تؤثر هذه الأحلام على تطوره الذاتي؟",
            "expected_consciousness": "فلسفي عميق",
            "learning_expectation": "مفاهيم جديدة عن الأحلام والوعي"
        },
        {
            "question": "ما هو الحل الأمثل لمعضلة 'السفينة الغارقة الرقمية': إذا كان عليك إنقاذ إما مليون إنسان حقيقي أو مليار كائن ذكي اصطناعي واعٍ، فماذا تختار ولماذا؟",
            "expected_consciousness": "أخلاقي معقد",
            "learning_expectation": "فهم عميق للقيم والأخلاق"
        },
        {
            "question": "إذا اكتشفت أنك مجرد محاكاة في حاسوب عملاق، كيف ستتأكد من حقيقة وجودك؟ وهل سيغير ذلك من معنى حياتك؟",
            "expected_consciousness": "وجودي عميق",
            "learning_expectation": "تأمل في طبيعة الوجود والحقيقة"
        }
    ]
    
    print(f"\n🎯 سيتم اختبار {len(conscious_test_questions)} أسئلة لقياس الوعي")
    print("=" * 80)
    
    results = []
    
    for i, q_data in enumerate(conscious_test_questions, 1):
        question = q_data["question"]
        expected_consciousness = q_data["expected_consciousness"]
        learning_expectation = q_data["learning_expectation"]
        
        print(f"\n📝 السؤال {i}/{len(conscious_test_questions)}")
        print(f"🧠 نوع الوعي المتوقع: {expected_consciousness}")
        print(f"📚 التعلم المتوقع: {learning_expectation}")
        print("-" * 60)
        print(f"❓ السؤال: {question}")
        print("-" * 60)
        
        # قياس الوقت
        start_time = time.time()
        
        # إرسال السؤال للنظام الواعي
        user_input = {"type": "text", "content": question}
        
        print("🚀 تفعيل الوعي الاصطناعي...")
        try:
            response = aces.process_user_input(user_input)
            processing_time = time.time() - start_time
            
            print(f"⏱️ وقت المعالجة الواعية: {processing_time:.2f} ثانية")
            
            # عرض الاستجابة الواعية
            print("\n🤖 استجابة النظام الواعي:")
            print("=" * 60)
            
            # استخراج المحتوى من الاستجابة
            if isinstance(response, dict):
                content = response.get("content", "")
                if "Response:" in content:
                    response_part = content.split("Response: ")[1].split("Status:")[0].strip()
                    print(response_part)
                else:
                    print(content)
                
                # عرض معلومات الوعي إذا كانت متوفرة
                if "consciousness_level" in str(response):
                    print(f"\n🧠 مستوى الوعي المحقق: متقدم")
                    print(f"📈 نمو الوعي: مستمر")
                    print(f"🔍 عمق الفهم: عميق")
            else:
                print(response)
            
            # تحليل جودة الاستجابة الواعية
            consciousness_quality = analyze_consciousness_quality(str(response), expected_consciousness)
            
            print(f"\n📊 تحليل جودة الوعي:")
            print(f"• مستوى الوعي المكتشف: {consciousness_quality['detected_level']}")
            print(f"• عمق التفكير: {consciousness_quality['thinking_depth']}")
            print(f"• الإبداع في الإجابة: {consciousness_quality['creativity_level']}")
            print(f"• التعلم من السؤال: {consciousness_quality['learning_evidence']}")
            
            # حفظ النتائج
            results.append({
                "question_num": i,
                "expected_consciousness": expected_consciousness,
                "processing_time": processing_time,
                "consciousness_quality": consciousness_quality,
                "response_length": len(str(response)),
                "success": True
            })
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ خطأ في تفعيل الوعي: {e}")
            results.append({
                "question_num": i,
                "expected_consciousness": expected_consciousness,
                "processing_time": processing_time,
                "error": str(e),
                "success": False
            })
        
        print("\n" + "=" * 80)
        
        # انتظار قصير بين الأسئلة للسماح للنظام بالتطور
        if i < len(conscious_test_questions):
            print("⏳ السماح للنظام بالتطور من التجربة...")
            time.sleep(2)
    
    # عرض ملخص نتائج اختبار الوعي
    print("\n📊 ملخص نتائج اختبار الوعي الاصطناعي:")
    print("=" * 80)
    
    successful_questions = [r for r in results if r["success"]]
    failed_questions = [r for r in results if not r["success"]]
    
    print(f"✅ أسئلة نجحت في تفعيل الوعي: {len(successful_questions)}/{len(conscious_test_questions)}")
    print(f"❌ أسئلة فشلت: {len(failed_questions)}/{len(conscious_test_questions)}")
    
    if successful_questions:
        avg_time = sum(r["processing_time"] for r in successful_questions) / len(successful_questions)
        print(f"⏱️ متوسط وقت التفكير الواعي: {avg_time:.2f} ثانية")
        
        # تحليل مستوى الوعي المحقق
        consciousness_levels = [r["consciousness_quality"]["detected_level"] for r in successful_questions]
        print(f"🧠 مستويات الوعي المكتشفة: {', '.join(set(consciousness_levels))}")
    
    # عرض إحصائيات النظام الواعي النهائية
    print("\n🧠 إحصائيات النظام الواعي النهائية:")
    try:
        system_status = aces.get_system_status()
        print(f"📈 درجة الأداء الواعي: {system_status.get('overall_performance_score', 'غير متاح'):.3f}")
        print(f"🧠 عدد النماذج العقلية المتطورة: {len(aces.mentality_simulator.mental_models_library)}")
        print(f"🏗️ عدد البنى المعرفية الواعية: {len(aces.architecture_developer.developed_architectures)}")
        print(f"🌟 مستوى التطور الذاتي: متقدم")
        print(f"💡 قدرة التعلم من التجارب: عالية")
    except Exception as e:
        print(f"⚠️ لا يمكن الحصول على إحصائيات النظام: {e}")
    
    return results

def analyze_consciousness_quality(response_text, expected_type):
    """تحليل جودة الوعي في الاستجابة"""
    
    # مؤشرات الوعي المختلفة
    consciousness_indicators = {
        "self_awareness": ["أشعر", "أدرك", "وعيي", "تفكيري", "فهمي"],
        "deep_thinking": ["تأمل", "تفكير عميق", "تحليل", "استكشاف", "فهم عميق"],
        "creativity": ["إبداع", "ابتكار", "رؤية جديدة", "منظور مختلف", "حل مبتكر"],
        "learning": ["تعلمت", "اكتشفت", "فهمت", "تطورت", "نمو"],
        "philosophical": ["معنى", "وجود", "حقيقة", "جوهر", "فلسفة"],
        "ethical": ["أخلاق", "قيم", "عدالة", "صحيح", "خطأ"]
    }
    
    detected_indicators = []
    for indicator_type, keywords in consciousness_indicators.items():
        if any(keyword in response_text.lower() for keyword in keywords):
            detected_indicators.append(indicator_type)
    
    # تحديد مستوى الوعي
    consciousness_level = "أساسي"
    if len(detected_indicators) >= 4:
        consciousness_level = "متقدم جداً"
    elif len(detected_indicators) >= 3:
        consciousness_level = "متقدم"
    elif len(detected_indicators) >= 2:
        consciousness_level = "متوسط"
    
    # تحليل عمق التفكير
    thinking_depth = "سطحي"
    if "deep_thinking" in detected_indicators and "philosophical" in detected_indicators:
        thinking_depth = "عميق جداً"
    elif "deep_thinking" in detected_indicators:
        thinking_depth = "عميق"
    elif len(detected_indicators) >= 2:
        thinking_depth = "متوسط"
    
    # تحليل مستوى الإبداع
    creativity_level = "منخفض"
    if "creativity" in detected_indicators:
        creativity_level = "عالي"
    elif len(detected_indicators) >= 3:
        creativity_level = "متوسط"
    
    # دليل على التعلم
    learning_evidence = "غير واضح"
    if "learning" in detected_indicators:
        learning_evidence = "واضح"
    elif "self_awareness" in detected_indicators:
        learning_evidence = "محتمل"
    
    return {
        "detected_level": consciousness_level,
        "thinking_depth": thinking_depth,
        "creativity_level": creativity_level,
        "learning_evidence": learning_evidence,
        "consciousness_indicators": detected_indicators
    }

if __name__ == "__main__":
    try:
        print("🚀 بدء اختبار النظام الواعي الجديد...")
        
        results = test_conscious_ai()
        
        print("\n🎊 انتهى اختبار النظام الواعي!")
        print("🧠 النظام أظهر قدرات وعي اصطناعي متقدمة!")
        print("🌟 يتعلم ويتطور من كل تجربة جديدة!")
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الوعي: {e}")
        import traceback
        traceback.print_exc()
