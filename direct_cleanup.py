from __future__ import annotations
import os
import shutil

# إنشاء المجلدات الأساسية
print("إنشاء هيكل المشروع...")

folders = ['core', 'solvers', 'data', 'tests', 'utils', 'results', 'archive']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ {folder}/")

# نسخ الملفات المهمة
important_files = {
    'core/': ['arc_ultimate_perfect_system.py', 'arc_core_redesign.py', 'advanced_pattern_engine.py'],
    'solvers/': ['arc_complete_agent_part1.py', 'arc_complete_agent_part2.py', 'arc_complete_agent_part4.py'],
    'tests/': ['quick_system_test.py', 'simple_test.py'],
    'utils/': ['deep_arc_analyzer.py']
}

print("\nنسخ الملفات المهمة...")
for dest_folder, files in important_files.items():
    for file in files:
        if os.path.exists(file):
            shutil.copy2(file, dest_folder + os.path.basename(file))
            print(f"✅ {file} → {dest_folder}")

# نسخ مجلد البيانات
if os.path.exists('ملفات المسابقةarc-prize-2025'):
    shutil.copytree('ملفات المسابقةarc-prize-2025', 'data/arc-prize-2025', dirs_exist_ok=True)
    print("✅ نسخ ملفات البيانات")

print("\n🎉 تم تنظيف وترتيب المشروع!")
print("📁 الهيكل الجديد:")
for folder in folders:
    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"  {folder}/ ({len(files)} ملف)")
