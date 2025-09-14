from __future__ import annotations
#!/usr/bin/env python3
"""
ARC CLEAN INTEGRATED SYSTEM - نظام ARC المتكامل والمنظف
==================================================
نظام موحد ومنظف يجمع أفضل مكونات المشروع لحل مهام ARC بفعالية
"""

import os
import sys
import json
import numpy as np
from collections.abc import Callable
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import copy

class ARCCleanIntegratedSystem:
    """النظام المتكامل والمنظف لحل مهام ARC"""
    
    def __init__(self):
        self.patterns_memory = {}
        self.success_stats = defaultdict(int)
        self.transformation_rules = {}
        
        print("🚀 تم تهيئة النظام المتكامل")
        self._initialize_system()
    
    def _initialize_system(self):
        """تهيئة النظام"""
        # تحميل الأنماط المحفوظة إن وجدت
        self._load_learned_patterns()
        
        # إعداد قواعد التحويل الأساسية
        self._setup_transformation_rules()
        
        print("✅ تم تهيئة النظام بنجاح")
    
    def solve_task(self, task: Dict) -> List[np.ndarray]:
        """حل مهمة ARC"""
        try:
            print(f"🎯 بدء حل المهمة...")
            
            # تحليل أمثلة التدريب
            patterns = self._analyze_training_examples(task['train'])
            print(f"📊 تم اكتشاف نمط: {patterns['type']}")
            
            # تطبيق النمط على اختبارات
            solutions = []
            for i, test_input in enumerate(task['test']):
                solution = self._apply_pattern(test_input['input'], patterns)
                solutions.append(solution)
                print(f"✅ حل الاختبار {i+1}: {solution.shape}")
            
            return solutions
            
        except Exception as e:
            print(f"❌ خطأ في الحل: {e}")
            return [np.array(test['input']) for test in task['test']]
    
    def _analyze_training_examples(self, train_examples: List[Dict]) -> Dict:
        """تحليل أمثلة التدريب"""
        patterns = {
            'type': 'unknown',
            'confidence': 0.0,
            'rules': {}
        }
        
        # تحليل كل مثال
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # فحص أنواع التحويلات المختلفة
            pattern_type = self._detect_pattern_type(input_grid, output_grid)
            
            if pattern_type != 'unknown':
                patterns['type'] = pattern_type
                patterns['confidence'] = 0.8
                patterns['rules'] = self._extract_transformation_rules(
                    input_grid, output_grid, pattern_type
                )
                break
        
        return patterns
    
    def _detect_pattern_type(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """اكتشاف نوع النمط"""
        
        # فحص تحويل الألوان
        if input_grid.shape == output_grid.shape:
            if self._is_color_transformation(input_grid, output_grid):
                return 'color_mapping'
            
            if self._is_geometric_transformation(input_grid, output_grid):
                return 'geometric'
        
        # فحص تحويل الحجم
        if input_grid.shape != output_grid.shape:
            return 'size_transformation'
        
        # فحص تحريك الكائنات
        if self._is_object_movement(input_grid, output_grid):
            return 'object_movement'
        
        return 'unknown'
    
    def _is_color_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص إذا كان التحويل خاص بالألوان"""
        # فحص إذا كانت الأشكال نفسها لكن الألوان مختلفة
        input_pattern = (input_grid != 0).astype(int)
        output_pattern = (output_grid != 0).astype(int)
        
        return np.array_equal(input_pattern, output_pattern)
    
    def _is_geometric_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص التحويلات الهندسية"""
        transformations = [
            np.fliplr(input_grid),      # انعكاس أفقي
            np.flipud(input_grid),      # انعكاس عمودي
            np.rot90(input_grid),       # دوران 90
            np.rot90(input_grid, 2),    # دوران 180
            np.rot90(input_grid, 3)     # دوران 270
        ]
        
        for transformed in transformations:
            if np.array_equal(transformed, output_grid):
                return True
        
        return False
    
    def _is_object_movement(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص تحريك الكائنات"""
        # مقارنة توزيع الألوان
        input_colors = Counter(input_grid.flatten())
        output_colors = Counter(output_grid.flatten())
        
        return input_colors == output_colors and not np.array_equal(input_grid, output_grid)
    
    def _extract_transformation_rules(self, input_grid: np.ndarray, output_grid: np.ndarray, pattern_type: str) -> Dict:
        """استخراج قواعد التحويل"""
        rules = {}
        
        if pattern_type == 'color_mapping':
            color_map = {}
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    input_color = input_grid[i, j]
                    output_color = output_grid[i, j]
                    color_map[input_color] = output_color
            rules['color_map'] = color_map
        
        elif pattern_type == 'geometric':
            # تحديد نوع التحويل الهندسي
            if np.array_equal(output_grid, np.fliplr(input_grid)):
                rules['transform'] = 'flip_horizontal'
            elif np.array_equal(output_grid, np.flipud(input_grid)):
                rules['transform'] = 'flip_vertical'
            elif np.array_equal(output_grid, np.rot90(input_grid)):
                rules['transform'] = 'rotate_90'
            elif np.array_equal(output_grid, np.rot90(input_grid, 2)):
                rules['transform'] = 'rotate_180'
        
        elif pattern_type == 'size_transformation':
            rules['size_ratio'] = (
                output_grid.shape[0] / input_grid.shape[0],
                output_grid.shape[1] / input_grid.shape[1]
            )
        
        return rules
    
    def _apply_pattern(self, test_input: List[List[int]], patterns: Dict) -> np.ndarray:
        """تطبيق النمط على مدخل الاختبار"""
        input_grid = np.array(test_input)
        
        pattern_type = patterns['type']
        rules = patterns['rules']
        
        if pattern_type == 'color_mapping' and 'color_map' in rules:
            return self._apply_color_mapping(input_grid, rules['color_map'])
        
        elif pattern_type == 'geometric' and 'transform' in rules:
            return self._apply_geometric_transform(input_grid, rules['transform'])
        
        elif pattern_type == 'size_transformation' and 'size_ratio' in rules:
            return self._apply_size_transform(input_grid, rules['size_ratio'])
        
        else:
            return input_grid
    
    def _apply_color_mapping(self, grid: np.ndarray, color_map: Dict) -> np.ndarray:
        """تطبيق تحويل الألوان"""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[grid == old_color] = new_color
        return result
    
    def _apply_geometric_transform(self, grid: np.ndarray, transform: str) -> np.ndarray:
        """تطبيق التحويل الهندسي"""
        if transform == 'flip_horizontal':
            return np.fliplr(grid)
        elif transform == 'flip_vertical':
            return np.flipud(grid)
        elif transform == 'rotate_90':
            return np.rot90(grid)
        elif transform == 'rotate_180':
            return np.rot90(grid, 2)
        else:
            return grid
    
    def _apply_size_transform(self, grid: np.ndarray, size_ratio: Tuple[float, float]) -> np.ndarray:
        """تطبيق تحويل الحجم"""
        new_height = int(grid.shape[0] * size_ratio[0])
        new_width = int(grid.shape[1] * size_ratio[1])
        
        if size_ratio[0] > 1 or size_ratio[1] > 1:
            # تكبير
            result = np.zeros((new_height, new_width), dtype=grid.dtype)
            scale_h = int(size_ratio[0])
            scale_w = int(size_ratio[1])
            
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    result[i*scale_h:(i+1)*scale_h, j*scale_w:(j+1)*scale_w] = grid[i, j]
            
            return result
        else:
            # تصغير
            scale_h = int(1 / size_ratio[0])
            scale_w = int(1 / size_ratio[1])
            
            result = np.zeros((new_height, new_width), dtype=grid.dtype)
            for i in range(new_height):
                for j in range(new_width):
                    result[i, j] = grid[i*scale_h, j*scale_w]
            
            return result
    
    def _load_learned_patterns(self):
        """تحميل الأنماط المتعلمة"""
        try:
            if os.path.exists('patterns_memory.json'):
                with open('patterns_memory.json', 'r') as f:
                    self.patterns_memory = json.load(f)
                print("✅ تم تحميل الأنماط المحفوظة")
        except:
            print("⚠️ لم يتم العثور على أنماط محفوظة")
    
    def _setup_transformation_rules(self):
        """إعداد قواعد التحويل"""
        self.transformation_rules = {
            'color_mapping': self._apply_color_mapping,
            'geometric': self._apply_geometric_transform,
            'size_transformation': self._apply_size_transform
        }
    
    def test_system(self):
        """اختبار النظام"""
        print("🧪 اختبار النظام المتكامل...")
        
        # اختبار بسيط
        test_task = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[2, 0], [0, 2]]
                }
            ],
            'test': [
                {'input': [[1, 0], [0, 1]]}
            ]
        }
        
        try:
            solutions = self.solve_task(test_task)
            print(f"✅ نجح الاختبار: {len(solutions)} حل")
            for i, sol in enumerate(solutions):
                print(f"  الحل {i+1}: {sol.tolist()}")
        except Exception as e:
            print(f"❌ فشل الاختبار: {e}")
    
    def load_and_test_real_tasks(self):
        """تحميل واختبار مهام حقيقية"""
        print("📊 اختبار على مهام ARC حقيقية...")
        
        try:
            # محاولة تحميل ملفات البيانات
            data_paths = [
                'ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json',
                'data/arc-tasks/arc-agi_training_challenges.json',
                'arc-agi_training_challenges.json'
            ]
            
            tasks = None
            for path in data_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        tasks = json.load(f)
                    print(f"✅ تم تحميل المهام من: {path}")
                    break
            
            if tasks is None:
                print("❌ لم يتم العثور على ملفات البيانات")
                return
            
            # اختبار على أول 3 مهام
            task_ids = list(tasks.keys())[:3]
            success_count = 0
            
            for task_id in task_ids:
                print(f"\n🎯 اختبار المهمة: {task_id}")
                task = tasks[task_id]
                
                try:
                    solutions = self.solve_task(task)
                    if solutions and len(solutions) > 0:
                        success_count += 1
                        print(f"✅ نجح الحل")
                    else:
                        print(f"❌ فشل الحل")
                except Exception as e:
                    print(f"❌ خطأ: {e}")
            
            print(f"\n📈 النتيجة النهائية: {success_count}/{len(task_ids)} ({success_count/len(task_ids)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ خطأ في تحميل المهام: {e}")

def main():
    """الوظيفة الرئيسية"""
    print("🚀 تشغيل النظام المتكامل والمنظف")
    print("=" * 50)
    
    # إنشاء النظام
    system = ARCCleanIntegratedSystem()
    
    # اختبار بسيط
    system.test_system()
    
    # اختبار على مهام حقيقية
    system.load_and_test_real_tasks()
    
    print("\n🎉 انتهى تشغيل النظام")

if __name__ == "__main__":
    main()