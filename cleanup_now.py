from __future__ import annotations
#!/usr/bin/env python3
"""
تنظيف فوري للمشروع
"""

import os
import shutil
from pathlib import Path

def main():
    print("🧹 بدء تنظيف المشروع...")
    
    # إنشاء مجلدات منظمة
    folders = {
        'core': 'الأنظمة الأساسية',
        'data': 'البيانات', 
        'tests': 'الاختبارات',
        'backup': 'النسخ الاحتياطية'
    }
    
    for folder, desc in folders.items():
        Path(folder).mkdir(exist_ok=True)
        print(f"📁 {folder}/ - {desc}")
    
    # نقل الملفات المهمة
    important_files = {
        'core': [
            'arc_ultimate_perfect_system.py',
            'arc_core_redesign.py', 
            'advanced_pattern_engine.py',
            'main.py'
        ],
        'tests': [
            'quick_system_test.py',
            'simple_test.py'
        ]
    }
    
    print("\n📦 نقل الملفات المهمة:")
    for dest, files in important_files.items():
        for file in files:
            if os.path.exists(file):
                dest_path = f"{dest}/{file}"
                if not os.path.exists(dest_path):
                    shutil.copy2(file, dest_path)
                    print(f"  ✅ {file} → {dest}/")
    
    # نقل مجلد البيانات
    data_folder = 'ملفات المسابقةarc-prize-2025'
    if os.path.exists(data_folder):
        dest_data = 'data/arc-tasks'
        if not os.path.exists(dest_data):
            shutil.copytree(data_folder, dest_data)
            print(f"  ✅ {data_folder} → data/arc-tasks/")
    
    # إحصائيات
    print("\n📊 النتائج:")
    for folder in folders.keys():
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
            print(f"  {folder}/: {count} ملف")
    
    print("\n✅ تم التنظيف بنجاح!")

if __name__ == "__main__":
    main()
