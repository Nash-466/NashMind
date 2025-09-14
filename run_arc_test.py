from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تشغيل الاختبار الكامل لنظام ARC
"""

import sys
import os
import subprocess
import time

def check_requirements():
    """فحص المتطلبات"""
    
    print("🔍 فحص المتطلبات...")
    
    required_files = [
        'arc-agi_training_challenges.json',
        'arc-agi_training_solutions.json',
        'ultimate_ai_system.py',
        'arc_learning_solver.py',
        'arc_ultimate_test.py',
        'arc_results_analyzer.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ ملفات مفقودة:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ جميع الملفات متوفرة")
    return True

def install_requirements():
    """تثبيت المتطلبات"""
    
    print("📦 تثبيت المتطلبات...")
    
    requirements = [
        'numpy',
        'matplotlib',
        'scikit-learn'
    ]
    
    for package in requirements:
        try:
            __import__(package)
            print(f"✅ {package} متوفر")
        except ImportError:
            print(f"📦 تثبيت {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ تم تثبيت {package}")
            except subprocess.CalledProcessError:
                print(f"❌ فشل في تثبيت {package}")
                return False
    
    return True

def run_test():
    """تشغيل الاختبار"""
    
    print("\n🚀 بدء اختبار النظام النهائي على ARC")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # تشغيل الاختبار
        print("🧪 تشغيل الاختبار...")
        result = subprocess.run([sys.executable, 'arc_ultimate_test.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ انتهى الاختبار بنجاح!")
            print(result.stdout)
        else:
            print("❌ فشل الاختبار:")
            print(result.stderr)
            return False
        
        # تشغيل التحليل
        print("\n🔬 تشغيل تحليل النتائج...")
        result = subprocess.run([sys.executable, 'arc_results_analyzer.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ انتهى التحليل بنجاح!")
            print(result.stdout)
        else:
            print("❌ فشل التحليل:")
            print(result.stderr)
            return False
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️ إجمالي وقت التشغيل: {total_time:.2f} ثانية")
        
        return True
        
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {e}")
        return False

def show_results():
    """عرض النتائج"""
    
    print("\n📊 ملفات النتائج المتوفرة:")
    
    result_files = [
        'arc_test_results.json',
        'arc_performance_comparison.png'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size} بايت)")
        else:
            print(f"   ❌ {file} غير متوفر")

def main():
    """الدالة الرئيسية"""
    
    print("🎯 نظام اختبار ARC الشامل")
    print("="*80)
    
    # فحص المتطلبات
    if not check_requirements():
        print("\n❌ يرجى التأكد من وجود جميع الملفات المطلوبة")
        return
    
    # تثبيت المتطلبات
    if not install_requirements():
        print("\n❌ فشل في تثبيت المتطلبات")
        return
    
    # تشغيل الاختبار
    if not run_test():
        print("\n❌ فشل في تشغيل الاختبار")
        return
    
    # عرض النتائج
    show_results()
    
    print("\n🎊 انتهى الاختبار الشامل بنجاح!")
    print("="*80)
    
    print("\n📋 الخطوات التالية:")
    print("   1. راجع ملف arc_test_results.json للنتائج المفصلة")
    print("   2. اعرض الرسم البياني arc_performance_comparison.png")
    print("   3. حلل نقاط القوة والضعف في كل نظام")
    print("   4. طور تحسينات بناءً على النتائج")

if __name__ == "__main__":
    main()
