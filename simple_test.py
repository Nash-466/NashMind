from __future__ import annotations
print("🧪 اختبار الأنظمة")
print("=" * 30)

import os
import sys

# فحص الملفات الأساسية
files = [
    "main.py",
    "arc_complete_agent_part1.py", 
    "arc_complete_agent_part2.py",
    "arc_ultimate_perfect_system.py"
]

print("📁 فحص الملفات:")
for f in files:
    status = "✅ موجود" if os.path.exists(f) else "❌ مفقود"
    print(f"  {f}: {status}")

# فحص ملفات البيانات
data_path = "ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json"
if os.path.exists(data_path):
    print(f"\n📊 ملفات البيانات: ✅ متاحة")
else:
    print(f"\n📊 ملفات البيانات: ❌ مفقودة")

# اختبار استيراد بسيط
print("\n🔧 اختبار الاستيراد:")
try:
    import numpy as np
    print("  numpy: ✅")
except:
    print("  numpy: ❌")

try:
    import json
    print("  json: ✅")
except:
    print("  json: ❌")

print("\n✅ انتهى الاختبار")
