#!/usr/bin/env python3
"""
حذف ملفات JSON غير المستخدمة
"""

import os
import json

def cleanup_json_files():
    """حذف ملفات JSON غير المستخدمة"""
    
    # الملفات المهمة التي يجب الاحتفاظ بها
    important_files = {
        'arc-agi_evaluation_challenges.json',
        'ملفات المسابقةarc-prize-2025/arc-agi_evaluation_challenges.json',
        'ملفات المسابقةarc-prize-2025/arc-agi_evaluation_solutions.json',
        'ملفات المسابقةarc-prize-2025/arc-agi_test_challenges.json',
        'ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json',
        'arc_memory.json',
        'ai_memory.json',
        'meta_kb.json',
        'nashmind_memory.json',
        'ultimate_memory.json'
    }
    
    # البحث عن جميع ملفات JSON
    json_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.json'):
                full_path = os.path.join(root, file)
                json_files.append(full_path)
    
    print(f"🔍 تم العثور على {len(json_files)} ملف JSON")
    
    # تصنيف الملفات
    to_delete = []
    to_keep = []
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        relative_path = file_path.replace('.\\', '').replace('./', '')
        
        # فحص إذا كان الملف مهم
        is_important = False
        for important in important_files:
            if important in relative_path or important == file_name:
                is_important = True
                break
        
        if is_important:
            to_keep.append(file_path)
        else:
            # فحص إضافي للملفات التي قد تكون مهمة
            if any(keyword in file_name.lower() for keyword in ['training', 'evaluation', 'test', 'challenge', 'solution']):
                if 'iter_' not in file_name and 'results_' not in file_name:
                    to_keep.append(file_path)
                else:
                    to_delete.append(file_path)
            else:
                to_delete.append(file_path)
    
    print(f"📁 ملفات للاحتفاظ: {len(to_keep)}")
    print(f"🗑️ ملفات للحذف: {len(to_delete)}")
    
    # عرض الملفات التي سيتم حذفها
    if to_delete:
        print("\\n🗑️ الملفات التي سيتم حذفها:")
        for file_path in to_delete[:20]:  # عرض أول 20 فقط
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"   - {os.path.basename(file_path)} ({file_size:,} bytes)")
        
        if len(to_delete) > 20:
            print(f"   ... و {len(to_delete) - 20} ملف آخر")
    
    # حذف الملفات
    deleted_count = 0
    total_size_deleted = 0
    
    for file_path in to_delete:
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                os.remove(file_path)
                deleted_count += 1
                total_size_deleted += file_size
        except Exception as e:
            print(f"❌ فشل حذف {file_path}: {e}")
    
    print(f"\\n✅ تم حذف {deleted_count} ملف")
    print(f"💾 تم توفير {total_size_deleted:,} bytes ({total_size_deleted/1024/1024:.1f} MB)")
    
    # عرض الملفات المحتفظ بها
    print(f"\\n📁 الملفات المحتفظ بها ({len(to_keep)}):")
    for file_path in to_keep:
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"   ✅ {os.path.basename(file_path)} ({file_size:,} bytes)")

if __name__ == "__main__":
    cleanup_json_files()
