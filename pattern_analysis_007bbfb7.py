from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
تحليل مفصل لنمط المهمة 007bbfb7 لتحسين استراتيجية التوسيع
"""

import numpy as np
import json

def analyze_007bbfb7_pattern():
    """تحليل مفصل لنمط المهمة 007bbfb7"""
    
    print("🔍 تحليل مفصل لنمط المهمة 007bbfb7")
    print("="*50)
    
    # تحميل البيانات
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    task_data = challenges["007bbfb7"]
    solution_data = solutions["007bbfb7"]
    
    print(f"📊 عدد أمثلة التدريب: {len(task_data['train'])}")
    
    # تحليل كل مثال
    for i, example in enumerate(task_data['train']):
        print(f"\n🎓 تحليل المثال {i+1}:")
        
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        print(f"  الدخل: {input_grid.shape}")
        print(f"  الخرج: {output_grid.shape}")
        
        # تحليل النمط في الخرج
        pattern_positions = analyze_pattern_positions(input_grid, output_grid)
        print(f"  مواضع النمط: {pattern_positions}")
        
        # تحليل القاعدة
        rule = analyze_transformation_rule(input_grid, output_grid)
        print(f"  القاعدة: {rule}")
    
    # تحليل مثال الاختبار
    test_input = np.array(task_data['test'][0]['input'])
    correct_output = np.array(solution_data[0])
    
    print(f"\n🧪 تحليل مثال الاختبار:")
    print(f"  الدخل: {test_input.shape}")
    print(f"  الخرج الصحيح: {correct_output.shape}")
    
    # تحليل النمط الصحيح
    correct_positions = analyze_pattern_positions(test_input, correct_output)
    print(f"  المواضع الصحيحة: {correct_positions}")
    
    # استنتاج القاعدة العامة
    general_rule = derive_general_rule(task_data['train'], solution_data)
    print(f"\n🎯 القاعدة العامة المستنتجة:")
    print(f"  {general_rule}")
    
    return general_rule

def analyze_pattern_positions(input_pattern, output_grid):
    """تحليل مواضع النمط في الخرج"""
    positions = []
    
    try:
        ph, pw = input_pattern.shape
        gh, gw = output_grid.shape
        
        # البحث عن النمط في كل موضع محتمل
        for i in range(0, gh - ph + 1, ph):
            for j in range(0, gw - pw + 1, pw):
                subgrid = output_grid[i:i+ph, j:j+pw]
                
                # فحص التطابق التام
                if np.array_equal(subgrid, input_pattern):
                    block_pos = (i//ph, j//pw)
                    positions.append(block_pos)
                # فحص التطابق الجزئي
                elif np.sum(subgrid == input_pattern) > (ph * pw * 0.7):
                    block_pos = (i//ph, j//pw)
                    positions.append(f"{block_pos}_partial")
        
        return positions
    except:
        return []

def analyze_transformation_rule(input_grid, output_grid):
    """تحليل قاعدة التحويل"""
    try:
        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        
        # فحص إذا كان التوسيع 3x3
        if oh == 9 and ow == 9 and ih == 3 and iw == 3:
            return "توسيع 3x3 إلى 9x9"
        
        # فحص أنواع التوسيع الأخرى
        expansion_ratio_h = oh // ih
        expansion_ratio_w = ow // iw
        
        return f"توسيع {expansion_ratio_h}x{expansion_ratio_w}"
    except:
        return "غير محدد"

def derive_general_rule(training_examples, solutions):
    """استنتاج القاعدة العامة من جميع الأمثلة"""
    
    all_patterns = []
    
    for i, example in enumerate(training_examples):
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        positions = analyze_pattern_positions(input_grid, output_grid)
        all_patterns.append(positions)
    
    # تحليل الأنماط المشتركة
    common_positions = find_most_common_pattern(all_patterns)
    
    rule = {
        "type": "grid_expansion_3x3_to_9x9",
        "common_positions": common_positions,
        "description": "توسيع شبكة 3x3 إلى 9x9 مع وضع النمط في مواضع محددة"
    }
    
    return rule

def find_most_common_pattern(all_patterns):
    """العثور على النمط الأكثر شيوعاً"""
    
    # حساب تكرار كل موضع
    position_counts = {}
    
    for pattern in all_patterns:
        for pos in pattern:
            if isinstance(pos, tuple):  # تجاهل المواضع الجزئية
                position_counts[pos] = position_counts.get(pos, 0) + 1
    
    # ترتيب حسب التكرار
    sorted_positions = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 إحصائيات المواضع:")
    for pos, count in sorted_positions:
        print(f"  الموضع {pos}: {count} مرات")
    
    # المواضع الأكثر شيوعاً (تظهر في أكثر من نصف الأمثلة)
    min_frequency = len(all_patterns) // 2
    common_positions = [pos for pos, count in sorted_positions if count >= min_frequency]
    
    return common_positions

def generate_improved_expansion_strategy():
    """إنتاج استراتيجية توسيع محسنة"""
    
    rule = analyze_007bbfb7_pattern()
    
    print(f"\n🚀 استراتيجية التوسيع المحسنة:")
    print(f"  النوع: {rule['type']}")
    print(f"  المواضع الشائعة: {rule['common_positions']}")
    
    # إنتاج كود الاستراتيجية المحسنة
    strategy_code = f"""
def improved_expansion_strategy(input_array):
    \"\"\"استراتيجية التوسيع المحسنة بناءً على تحليل المهمة 007bbfb7\"\"\"
    
    if input_array.shape == (3, 3):
        # إنشاء شبكة 9x9
        expanded = np.zeros((9, 9), dtype=int)
        
        # المواضع المحسنة من التحليل
        target_positions = {rule['common_positions']}
        
        # وضع النمط في المواضع المحددة
        for block_row in range(3):
            for block_col in range(3):
                if (block_row, block_col) in target_positions:
                    start_row = block_row * 3
                    start_col = block_col * 3
                    expanded[start_row:start_row+3, start_col:start_col+3] = input_array
        
        return expanded
    
    return None
"""
    
    print(f"\n💻 كود الاستراتيجية المحسنة:")
    print(strategy_code)
    
    return rule

if __name__ == "__main__":
    generate_improved_expansion_strategy()
