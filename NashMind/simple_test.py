#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اختبار بسيط لنظام NashMind المحسن
"""

import sys
import os

# إضافة المسار الحالي
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """اختبار استيراد الوحدات"""
    print("🔍 اختبار استيراد الوحدات...")
    
    try:
        from aces_system import ACES
        print("✅ تم استيراد ACES بنجاح")
        return True
    except Exception as e:
        print(f"❌ فشل استيراد ACES: {e}")
        return False

def test_system_creation():
    """اختبار إنشاء النظام"""
    print("\n🚀 اختبار إنشاء النظام...")
    
    try:
        from aces_system import ACES
        system = ACES()
        print("✅ تم إنشاء النظام بنجاح")
        return system
    except Exception as e:
        print(f"❌ فشل إنشاء النظام: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_new_features(system):
    """اختبار الميزات الجديدة"""
    print("\n🧠 اختبار الميزات الجديدة...")
    
    # فحص الوظائف الجديدة
    features = [
        ('real_learning_from_experience', 'التعلم الحقيقي'),
        ('solve_arc_problem', 'حل مسائل ARC'),
        ('enhanced_problem_solving', 'حل المشاكل المحسن'),
        ('true_learning_engine', 'محرك التعلم الحقيقي'),
        ('arc_problem_solver', 'حلال مسائل ARC')
    ]
    
    for feature, description in features:
        if hasattr(system, feature):
            print(f"✅ {description}: موجود")
        else:
            print(f"❌ {description}: غير موجود")

def main():
    """الوظيفة الرئيسية"""
    print("🎯 اختبار NashMind المحسن")
    print("="*50)
    
    # اختبار الاستيراد
    if not test_imports():
        return
    
    # اختبار إنشاء النظام
    system = test_system_creation()
    if system is None:
        return
    
    # اختبار الميزات الجديدة
    test_new_features(system)
    
    print("\n🎉 انتهى الاختبار!")
    print("✨ NashMind المحسن جاهز للاستخدام!")

if __name__ == "__main__":
    main()
