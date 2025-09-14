from __future__ import annotations
#!/usr/bin/env python3
"""
ADVANCED PATTERN DETECTOR - كاشف الأنماط المتقدم
==============================================
نظام متقدم لاكتشاف الأنماط المعقدة في مهام ARC
"""

import numpy as np
from collections.abc import Callable
from typing import Dict, List, Tuple, Any
from collections import Counter
import json

class AdvancedPatternDetector:
    """كاشف الأنماط المتقدم"""
    
    def __init__(self):
        # إحصائيات الأنماط من التحليل السابق
        self.pattern_priorities = {
            'object_manipulation': 1234,  # الأكثر شيوعاً
            'complex_unknown': 1003,
            'size_asymmetric': 756,
            'color_mapping': 90,
            'size_expand_2x': 89
        }
        
    def detect_advanced_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """اكتشاف الأنماط المتقدمة"""
        
        patterns = {
            'primary_pattern': 'unknown',
            'confidence': 0.0,
            'transformations': [],
            'complexity': 'simple'
        }
        
        # فحص الأنماط بترتيب الأولوية
        if self._detect_object_manipulation(input_grid, output_grid):
            patterns['primary_pattern'] = 'object_manipulation'
            patterns['confidence'] = 0.9
            patterns['transformations'] = self._analyze_object_operations(input_grid, output_grid)
            
        elif self._detect_size_asymmetric(input_grid, output_grid):
            patterns['primary_pattern'] = 'size_asymmetric'
            patterns['confidence'] = 0.8
            patterns['transformations'] = self._analyze_size_changes(input_grid, output_grid)
            
        elif self._detect_color_mapping(input_grid, output_grid):
            patterns['primary_pattern'] = 'color_mapping'
            patterns['confidence'] = 0.85
            patterns['transformations'] = self._analyze_color_changes(input_grid, output_grid)
        
        # تحديد مستوى التعقيد
        patterns['complexity'] = self._assess_complexity(input_grid, output_grid)
        
        return patterns
    
    def _detect_object_manipulation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """اكتشاف معالجة الكائنات"""
        
        # استخراج الكائنات
        input_objects = self._extract_objects(input_grid)
        output_objects = self._extract_objects(output_grid)
        
        # فحص التغييرات في الكائنات
        if len(input_objects) != len(output_objects):
            return True  # تغيير في عدد الكائنات
        
        # فحص تحريك أو تعديل الكائنات
        for i, (inp_obj, out_obj) in enumerate(zip(input_objects, output_objects)):
            if not np.array_equal(inp_obj['shape'], out_obj['shape']):
                return True  # تغيير في شكل الكائن
            
            if inp_obj['position'] != out_obj['position']:
                return True  # تحريك الكائن
        
        return False
    
    def _detect_size_asymmetric(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """اكتشاف تغيير الحجم غير المتماثل"""
        
        if input_grid.shape == output_grid.shape:
            return False
        
        # فحص إذا كان التغيير غير متماثل
        height_ratio = output_grid.shape[0] / input_grid.shape[0]
        width_ratio = output_grid.shape[1] / input_grid.shape[1]
        
        return abs(height_ratio - width_ratio) > 0.1  # غير متماثل
    
    def _detect_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """اكتشاف تحويل الألوان"""
        
        if input_grid.shape != output_grid.shape:
            return False
        
        # فحص إذا كانت الأشكال نفسها لكن الألوان مختلفة
        input_pattern = (input_grid != 0).astype(int)
        output_pattern = (output_grid != 0).astype(int)
        
        if not np.array_equal(input_pattern, output_pattern):
            return False
        
        # فحص وجود تحويل ألوان فعلي
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        
        return input_colors != output_colors
    
    def _extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """استخراج الكائنات من الشبكة"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0 and not visited[i, j]:
                    obj = self._extract_single_object(grid, visited, i, j, grid[i, j])
                    if obj['size'] > 0:
                        objects.append(obj)
        
        return objects
    
    def _extract_single_object(self, grid: np.ndarray, visited: np.ndarray, 
                             start_i: int, start_j: int, color: int) -> Dict:
        """استخراج كائن واحد"""
        
        positions = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or
                visited[i, j] or grid[i, j] != color):
                continue
            
            visited[i, j] = True
            positions.append((i, j))
            
            # إضافة الجيران
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((i + di, j + dj))
        
        if not positions:
            return {'size': 0, 'color': color, 'position': (0, 0), 'shape': np.array([])}
        
        # حساب الصندوق المحيط
        min_i = min(pos[0] for pos in positions)
        max_i = max(pos[0] for pos in positions)
        min_j = min(pos[1] for pos in positions)
        max_j = max(pos[1] for pos in positions)
        
        # استخراج شكل الكائن
        shape = np.zeros((max_i - min_i + 1, max_j - min_j + 1), dtype=int)
        for i, j in positions:
            shape[i - min_i, j - min_j] = color
        
        return {
            'size': len(positions),
            'color': color,
            'position': (min_i, min_j),
            'shape': shape,
            'bounding_box': (min_i, min_j, max_i, max_j)
        }
    
    def _analyze_object_operations(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """تحليل عمليات الكائنات"""
        operations = []
        
        input_objects = self._extract_objects(input_grid)
        output_objects = self._extract_objects(output_grid)
        
        if len(output_objects) > len(input_objects):
            operations.append('duplicate_objects')
        elif len(output_objects) < len(input_objects):
            operations.append('remove_objects')
        
        # فحص تحريك الكائنات
        if len(input_objects) == len(output_objects):
            moved = False
            for inp_obj, out_obj in zip(input_objects, output_objects):
                if inp_obj['position'] != out_obj['position']:
                    moved = True
                    break
            
            if moved:
                operations.append('move_objects')
        
        return operations
    
    def _analyze_size_changes(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """تحليل تغييرات الحجم"""
        transformations = []
        
        height_ratio = output_grid.shape[0] / input_grid.shape[0]
        width_ratio = output_grid.shape[1] / input_grid.shape[1]
        
        if height_ratio > 1:
            transformations.append(f'expand_height_{height_ratio:.1f}x')
        elif height_ratio < 1:
            transformations.append(f'shrink_height_{1/height_ratio:.1f}x')
        
        if width_ratio > 1:
            transformations.append(f'expand_width_{width_ratio:.1f}x')
        elif width_ratio < 1:
            transformations.append(f'shrink_width_{1/width_ratio:.1f}x')
        
        return transformations
    
    def _analyze_color_changes(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[str]:
        """تحليل تغييرات الألوان"""
        transformations = []
        
        # إنشاء خريطة تحويل الألوان
        color_map = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                input_color = input_grid[i, j]
                output_color = output_grid[i, j]
                
                if input_color in color_map:
                    if color_map[input_color] != output_color:
                        transformations.append('inconsistent_color_mapping')
                        return transformations
                else:
                    color_map[input_color] = output_color
        
        # تحليل نوع التحويل
        changed_colors = sum(1 for k, v in color_map.items() if k != v)
        if changed_colors > 0:
            transformations.append(f'color_mapping_{changed_colors}_colors')
        
        return transformations
    
    def _assess_complexity(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """تقييم مستوى التعقيد"""
        
        # عوامل التعقيد
        size_factor = max(input_grid.size, output_grid.size)
        color_factor = len(set(input_grid.flatten()) | set(output_grid.flatten()))
        shape_factor = abs(input_grid.shape[0] - output_grid.shape[0]) + abs(input_grid.shape[1] - output_grid.shape[1])
        
        complexity_score = (size_factor / 100) + (color_factor / 10) + (shape_factor / 10)
        
        if complexity_score < 1.0:
            return 'simple'
        elif complexity_score < 3.0:
            return 'medium'
        else:
            return 'complex'

# دمج الكاشف المتقدم مع النظام الرئيسي
def enhance_main_system():
    """تحسين النظام الرئيسي بالكاشف المتقدم"""
    
    enhancement_code = '''
# إضافة للنظام الرئيسي
from advanced_pattern_detector import AdvancedPatternDetector

class EnhancedARCSystem(ARCCleanIntegratedSystem):
    def __init__(self):
        super().__init__()
        self.advanced_detector = AdvancedPatternDetector()
        print("🔬 تم تفعيل الكاشف المتقدم")
    
    def _analyze_training_examples(self, train_examples: List[Dict]) -> Dict:
        """تحليل محسن للأمثلة"""
        patterns = super()._analyze_training_examples(train_examples)
        
        # استخدام الكاشف المتقدم
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            advanced_patterns = self.advanced_detector.detect_advanced_patterns(input_grid, output_grid)
            
            if advanced_patterns['confidence'] > patterns.get('confidence', 0):
                patterns.update(advanced_patterns)
                break
        
        return patterns
'''
    
    with open('enhanced_system.py', 'w', encoding='utf-8') as f:
        f.write(enhancement_code)
    
    print("✅ تم إنشاء النظام المحسن: enhanced_system.py")

if __name__ == "__main__":
    detector = AdvancedPatternDetector()
    enhance_main_system()
    print("🚀 الكاشف المتقدم جاهز للاستخدام")
