from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ نظام التحسين المباشر - تحسينات فورية وفعالة
"""

import json
import numpy as np
import time
from collections.abc import Callable
from typing import Dict, List, Any

class DirectImprovementSystem:
    """نظام التحسين المباشر"""
    
    def __init__(self):
        self.improvements_applied = []
        self.test_results = []
        
        # تحميل البيانات
        self.challenges, self.solutions = self._load_data()
        
        print("⚡ نظام التحسين المباشر جاهز")
        print(f"📊 {len(self.challenges)} مهمة متاحة")
    
    def _load_data(self):
        """تحميل بيانات التدريب"""
        try:
            with open('arc-agi_training_challenges.json', 'r') as f:
                challenges = json.load(f)
            with open('arc-agi_training_solutions.json', 'r') as f:
                solutions = json.load(f)
            return challenges, solutions
        except Exception as e:
            print(f"❌ خطأ في تحميل البيانات: {e}")
            return {}, {}
    
    def start_direct_improvement(self):
        """بدء التحسين المباشر"""
        
        print("🚀 بدء التحسين المباشر...")
        print("="*50)
        
        # اختبار أولي
        print("\n📊 اختبار أولي...")
        initial_results = self._quick_test(10)
        initial_rate = self._calculate_success_rate(initial_results)
        print(f"معدل النجاح الأولي: {initial_rate:.1%}")
        
        # تطبيق تحسينات مباشرة
        improvements = [
            self._improve_grid_completion,
            self._improve_pattern_matching,
            self._improve_color_consistency,
            self._improve_size_handling,
            self._improve_symmetry_detection
        ]
        
        for i, improvement_func in enumerate(improvements):
            print(f"\n🔧 تطبيق التحسين {i+1}/5...")
            
            try:
                success = improvement_func()
                if success:
                    print(f"   ✅ تم تطبيق التحسين {i+1}")
                    
                    # اختبار بعد التحسين
                    test_results = self._quick_test(10)
                    current_rate = self._calculate_success_rate(test_results)
                    print(f"   📊 معدل النجاح الحالي: {current_rate:.1%}")
                    
                    if current_rate > initial_rate:
                        print(f"   📈 تحسن! (+{(current_rate - initial_rate)*100:.1f}%)")
                        initial_rate = current_rate
                    
                else:
                    print(f"   ❌ فشل في تطبيق التحسين {i+1}")
                    
            except Exception as e:
                print(f"   ❌ خطأ في التحسين {i+1}: {e}")
        
        # اختبار نهائي
        print(f"\n🎯 اختبار نهائي...")
        final_results = self._quick_test(20)
        final_rate = self._calculate_success_rate(final_results)
        
        print(f"\n🏆 النتائج النهائية:")
        print(f"📊 معدل النجاح النهائي: {final_rate:.1%}")
        print(f"🔧 تحسينات مطبقة: {len(self.improvements_applied)}")
        
        return final_rate
    
    def _quick_test(self, num_tasks: int = 10):
        """اختبار سريع"""
        
        task_ids = list(self.challenges.keys())[:num_tasks]
        results = []
        
        for task_id in task_ids:
            try:
                result = self._solve_task(task_id)
                results.append(result)
                
                status = "✅" if result['solved_correctly'] else f"📊 {result['similarity']:.2f}"
                print(f"   {task_id[:8]}: {status}")
                
            except Exception as e:
                results.append({
                    'task_id': task_id,
                    'solved_correctly': False,
                    'similarity': 0.0,
                    'error': str(e)
                })
                print(f"   {task_id[:8]}: ❌")
        
        return results
    
    def _solve_task(self, task_id: str):
        """حل مهمة واحدة"""
        
        challenge = self.challenges[task_id]
        solution = self.solutions[task_id]
        
        test_case = challenge['test'][0]
        input_grid = np.array(test_case['input'])
        expected_output = np.array(solution[0])
        
        # حل باستخدام EfficientZero
        from efficient_zero_engine import EfficientZeroEngine
        ez = EfficientZeroEngine()
        
        result = ez.solve_arc_problem(input_grid, max_steps=5)
        
        if result.get('success', True):
            output_grid = np.array(result.get('solution_grid', input_grid))
            
            # حساب التشابه
            if output_grid.shape == expected_output.shape:
                similarity = np.sum(output_grid == expected_output) / output_grid.size
            else:
                similarity = 0.0
            
            return {
                'task_id': task_id,
                'similarity': similarity,
                'solved_correctly': similarity >= 0.99
            }
        else:
            raise Exception(result.get('error', 'فشل في الحل'))
    
    def _calculate_success_rate(self, results: List[Dict]) -> float:
        """حساب معدل النجاح"""
        if not results:
            return 0.0
        
        success_count = sum(1 for r in results if r.get('solved_correctly', False))
        return success_count / len(results)
    
    def _improve_grid_completion(self) -> bool:
        """تحسين إكمال الشبكة"""
        try:
            # قراءة الملف الحالي
            with open('efficient_zero_engine.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # إضافة دالة إكمال محسنة
            improvement_code = '''
    def _smart_grid_completion(self, grid: np.ndarray) -> np.ndarray:
        """إكمال ذكي للشبكة"""
        h, w = grid.shape
        new_grid = grid.copy()
        
        # ملء الفجوات الصغيرة
        for i in range(1, h-1):
            for j in range(1, w-1):
                if grid[i, j] == 0:
                    # فحص الجيران
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    
                    if len(non_zero_neighbors) >= 3:
                        # إذا كان 3 جيران أو أكثر لهم نفس القيمة
                        most_common = max(set(non_zero_neighbors), key=non_zero_neighbors.count)
                        if non_zero_neighbors.count(most_common) >= 2:
                            new_grid[i, j] = most_common
        
        return new_grid
            '''
            
            # إضافة الكود إذا لم يكن موجوداً
            if "_smart_grid_completion" not in content:
                # البحث عن مكان مناسب للإدراج
                insertion_point = content.find("def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:")
                if insertion_point != -1:
                    content = content[:insertion_point] + improvement_code + "\n    " + content[insertion_point:]
                    
                    # حفظ الملف
                    with open('efficient_zero_engine.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.improvements_applied.append("smart_grid_completion")
                    return True
            
            return False
            
        except Exception as e:
            print(f"خطأ في تحسين إكمال الشبكة: {e}")
            return False
    
    def _improve_pattern_matching(self) -> bool:
        """تحسين مطابقة الأنماط"""
        try:
            with open('efficient_zero_engine.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            improvement_code = '''
    def _enhanced_pattern_matching(self, grid: np.ndarray) -> np.ndarray:
        """مطابقة أنماط محسنة"""
        h, w = grid.shape
        
        # البحث عن أنماط متكررة
        for pattern_size in [2, 3]:
            if h >= pattern_size * 2 and w >= pattern_size * 2:
                # فحص النمط في الزاوية العلوية اليسرى
                pattern = grid[:pattern_size, :pattern_size]
                
                # التحقق من تكرار النمط
                is_repeating = True
                for i in range(0, h, pattern_size):
                    for j in range(0, w, pattern_size):
                        if i + pattern_size <= h and j + pattern_size <= w:
                            current_section = grid[i:i+pattern_size, j:j+pattern_size]
                            if not np.array_equal(current_section, pattern):
                                is_repeating = False
                                break
                    if not is_repeating:
                        break
                
                if is_repeating:
                    # تطبيق النمط على كامل الشبكة
                    new_grid = np.zeros_like(grid)
                    for i in range(0, h, pattern_size):
                        for j in range(0, w, pattern_size):
                            end_i = min(i + pattern_size, h)
                            end_j = min(j + pattern_size, w)
                            new_grid[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]
                    return new_grid
        
        return grid
            '''
            
            if "_enhanced_pattern_matching" not in content:
                insertion_point = content.find("def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:")
                if insertion_point != -1:
                    content = content[:insertion_point] + improvement_code + "\n    " + content[insertion_point:]
                    
                    with open('efficient_zero_engine.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.improvements_applied.append("enhanced_pattern_matching")
                    return True
            
            return False
            
        except Exception as e:
            print(f"خطأ في تحسين مطابقة الأنماط: {e}")
            return False
    
    def _improve_color_consistency(self) -> bool:
        """تحسين اتساق الألوان"""
        try:
            with open('efficient_zero_engine.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            improvement_code = '''
    def _ensure_color_consistency(self, grid: np.ndarray, reference_grid: np.ndarray) -> np.ndarray:
        """ضمان اتساق الألوان"""
        # الحصول على الألوان المستخدمة في المرجع
        reference_colors = set(reference_grid.flatten())
        current_colors = set(grid.flatten())
        
        # إذا كانت الألوان مختلفة، حاول التصحيح
        if reference_colors != current_colors:
            new_grid = grid.copy()
            
            # إذا كان هناك لون واحد زائد، حاول استبداله
            extra_colors = current_colors - reference_colors
            missing_colors = reference_colors - current_colors
            
            if len(extra_colors) == 1 and len(missing_colors) == 1:
                extra_color = list(extra_colors)[0]
                missing_color = list(missing_colors)[0]
                new_grid[grid == extra_color] = missing_color
                return new_grid
        
        return grid
            '''
            
            if "_ensure_color_consistency" not in content:
                insertion_point = content.find("def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:")
                if insertion_point != -1:
                    content = content[:insertion_point] + improvement_code + "\n    " + content[insertion_point:]
                    
                    with open('efficient_zero_engine.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.improvements_applied.append("color_consistency")
                    return True
            
            return False
            
        except Exception as e:
            print(f"خطأ في تحسين اتساق الألوان: {e}")
            return False
    
    def _improve_size_handling(self) -> bool:
        """تحسين التعامل مع الأحجام"""
        try:
            with open('efficient_zero_engine.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # تحسين دالة التحجيم الموجودة
            old_scale_function = "def _scale_grid(self, grid: np.ndarray, factor: float) -> np.ndarray:"
            
            if old_scale_function in content:
                # استبدال الدالة بنسخة محسنة
                improved_scale = '''def _scale_grid(self, grid: np.ndarray, factor: float) -> np.ndarray:
        """تحجيم محسن للشبكة"""
        if factor == 1.0:
            return grid
        
        h, w = grid.shape
        new_h, new_w = int(h * factor), int(w * factor)
        
        if new_h <= 0 or new_w <= 0:
            return grid
        
        new_grid = np.zeros((new_h, new_w), dtype=grid.dtype)
        
        # تحجيم ذكي
        for i in range(new_h):
            for j in range(new_w):
                orig_i = min(int(i / factor), h - 1)
                orig_j = min(int(j / factor), w - 1)
                new_grid[i, j] = grid[orig_i, orig_j]
        
        return new_grid'''
                
                # استبدال الدالة
                start_idx = content.find(old_scale_function)
                if start_idx != -1:
                    # البحث عن نهاية الدالة
                    end_idx = content.find("\n    def ", start_idx + 1)
                    if end_idx == -1:
                        end_idx = len(content)
                    
                    content = content[:start_idx] + improved_scale + content[end_idx:]
                    
                    with open('efficient_zero_engine.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.improvements_applied.append("improved_scaling")
                    return True
            
            return False
            
        except Exception as e:
            print(f"خطأ في تحسين التعامل مع الأحجام: {e}")
            return False
    
    def _improve_symmetry_detection(self) -> bool:
        """تحسين اكتشاف التماثل"""
        try:
            with open('efficient_zero_engine.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            improvement_code = '''
    def _advanced_symmetry_completion(self, grid: np.ndarray) -> np.ndarray:
        """إكمال التماثل المتقدم"""
        h, w = grid.shape
        new_grid = grid.copy()
        
        # فحص التماثل الأفقي
        if w % 2 == 0:
            left_half = grid[:, :w//2]
            right_half = grid[:, w//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # حساب التشابه
            similarity = np.sum(left_half == right_half_flipped) / left_half.size
            
            if similarity > 0.7:  # إذا كان هناك تماثل جزئي
                # إكمال التماثل
                for i in range(h):
                    for j in range(w//2):
                        left_val = grid[i, j]
                        right_val = grid[i, w-1-j]
                        
                        if left_val != 0 and right_val == 0:
                            new_grid[i, w-1-j] = left_val
                        elif left_val == 0 and right_val != 0:
                            new_grid[i, j] = right_val
        
        # فحص التماثل العمودي
        if h % 2 == 0:
            top_half = grid[:h//2, :]
            bottom_half = grid[h//2:, :]
            bottom_half_flipped = np.flipud(bottom_half)
            
            similarity = np.sum(top_half == bottom_half_flipped) / top_half.size
            
            if similarity > 0.7:
                for i in range(h//2):
                    for j in range(w):
                        top_val = grid[i, j]
                        bottom_val = grid[h-1-i, j]
                        
                        if top_val != 0 and bottom_val == 0:
                            new_grid[h-1-i, j] = top_val
                        elif top_val == 0 and bottom_val != 0:
                            new_grid[i, j] = bottom_val
        
        return new_grid
            '''
            
            if "_advanced_symmetry_completion" not in content:
                insertion_point = content.find("def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:")
                if insertion_point != -1:
                    content = content[:insertion_point] + improvement_code + "\n    " + content[insertion_point:]
                    
                    with open('efficient_zero_engine.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.improvements_applied.append("advanced_symmetry")
                    return True
            
            return False
            
        except Exception as e:
            print(f"خطأ في تحسين اكتشاف التماثل: {e}")
            return False

def main():
    """الدالة الرئيسية"""
    
    system = DirectImprovementSystem()
    final_rate = system.start_direct_improvement()
    
    print("\n" + "="*50)
    print("🏆 ملخص النتائج:")
    print(f"📊 معدل النجاح النهائي: {final_rate:.1%}")
    print(f"🔧 تحسينات مطبقة: {len(system.improvements_applied)}")
    
    if final_rate > 0:
        print("🎉 تم تحقيق تقدم!")
    else:
        print("⚠️ يحتاج مزيد من التطوير")

if __name__ == "__main__":
    main()
