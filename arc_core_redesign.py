from __future__ import annotations
#!/usr/bin/env python3
"""
ARC Core Redesign - النواة الجديدة المبسطة والفعالة
=======================================================
التركيز على الفهم المجرد والتعلم الحقيقي من الأمثلة
"""

import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class ARCPattern:
    """نمط ARC مع قاعدة التحويل"""
    name: str
    rule: Callable[[np.ndarray], np.ndarray]
    confidence: float
    parameters: Dict[str, Any]
    
class AbstractRuleExtractor:
    """مستخرج القواعد المجردة - القلب الحقيقي للنظام"""
    
    def __init__(self):
        self.discovered_rules = []
        self.rule_patterns = {
            'spatial_transformations': self._detect_spatial_transforms,
            'color_mappings': self._detect_color_mappings,
            'shape_operations': self._detect_shape_operations,
            'pattern_completion': self._detect_pattern_completion,
            'object_manipulation': self._detect_object_manipulation
        }
    
    def extract_rule_from_examples(self, train_pairs: List[Dict]) -> List[ARCPattern]:
        """استخراج القاعدة المجردة من الأمثلة"""
        patterns = []
        
        for pattern_type, detector in self.rule_patterns.items():
            try:
                detected_patterns = detector(train_pairs)
                patterns.extend(detected_patterns)
            except Exception as e:
                continue
                
        # ترتيب الأنماط حسب الثقة
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        return patterns[:5]  # أفضل 5 أنماط
    
    def _detect_spatial_transforms(self, pairs: List[Dict]) -> List[ARCPattern]:
        """كشف التحويلات المكانية"""
        patterns = []
        
        # فحص التحويلات الأساسية
        transforms = {
            'rotate_90': lambda x: np.rot90(x, 1),
            'rotate_180': lambda x: np.rot90(x, 2),
            'rotate_270': lambda x: np.rot90(x, 3),
            'flip_horizontal': np.fliplr,
            'flip_vertical': np.flipud,
            'transpose': np.transpose
        }
        
        for name, transform in transforms.items():
            confidence = self._test_transform_consistency(pairs, transform)
            if confidence > 0.8:
                patterns.append(ARCPattern(
                    name=name,
                    rule=transform,
                    confidence=confidence,
                    parameters={'type': 'spatial'}
                ))
        
        return patterns
    
    def _detect_color_mappings(self, pairs: List[Dict]) -> List[ARCPattern]:
        """كشف تحويلات الألوان"""
        patterns = []
        
        # استخراج خريطة الألوان من جميع الأمثلة
        color_maps = []
        for pair in pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            if input_grid.shape == output_grid.shape:
                color_map = self._extract_color_mapping(input_grid, output_grid)
                if color_map:
                    color_maps.append(color_map)
        
        # البحث عن خريطة ألوان متسقة
        if color_maps and self._is_consistent_color_mapping(color_maps):
            consistent_map = color_maps[0]  # استخدام الخريطة الأولى
            
            def apply_color_mapping(grid):
                result = grid.copy()
                for old_color, new_color in consistent_map.items():
                    result[grid == old_color] = new_color
                return result
            
            patterns.append(ARCPattern(
                name='color_mapping',
                rule=apply_color_mapping,
                confidence=0.9,
                parameters={'mapping': consistent_map}
            ))
        
        return patterns
    
    def _detect_shape_operations(self, pairs: List[Dict]) -> List[ARCPattern]:
        """كشف عمليات الأشكال"""
        patterns = []
        
        # فحص عمليات التكرار والتوسيع
        for pair in pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # فحص التكرار الأفقي/العمودي
            if self._is_repetition_pattern(input_grid, output_grid):
                rep_type, factor = self._get_repetition_details(input_grid, output_grid)
                
                def create_repetition_rule(rep_type, factor):
                    def apply_repetition(grid):
                        if rep_type == 'horizontal':
                            return np.tile(grid, (1, factor))
                        elif rep_type == 'vertical':
                            return np.tile(grid, (factor, 1))
                        elif rep_type == 'both':
                            return np.tile(grid, (factor, factor))
                        return grid
                    return apply_repetition
                
                patterns.append(ARCPattern(
                    name=f'repeat_{rep_type}',
                    rule=create_repetition_rule(rep_type, factor),
                    confidence=0.85,
                    parameters={'type': rep_type, 'factor': factor}
                ))
        
        return patterns
    
    def _detect_pattern_completion(self, pairs: List[Dict]) -> List[ARCPattern]:
        """كشف إكمال الأنماط"""
        patterns = []
        
        for pair in pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            # فحص إكمال الأنماط المفقودة
            if self._is_pattern_completion(input_grid, output_grid):
                completion_rule = self._extract_completion_rule(input_grid, output_grid)
                if completion_rule:
                    patterns.append(ARCPattern(
                        name='pattern_completion',
                        rule=completion_rule,
                        confidence=0.8,
                        parameters={'type': 'completion'}
                    ))
        
        return patterns
    
    def _detect_object_manipulation(self, pairs: List[Dict]) -> List[ARCPattern]:
        """كشف معالجة الكائنات"""
        patterns = []
        
        # هذا يحتاج تطوير أكثر تعقيداً
        # لكن الفكرة هي كشف حركة/تغيير الكائنات
        
        return patterns
    
    # Helper methods
    def _test_transform_consistency(self, pairs: List[Dict], transform: Callable) -> float:
        """اختبار ثبات التحويل عبر جميع الأمثلة"""
        matches = 0
        total = len(pairs)
        
        for pair in pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            try:
                transformed = transform(input_grid)
                if transformed.shape == output_grid.shape and np.array_equal(transformed, output_grid):
                    matches += 1
            except:
                continue
        
        return matches / max(total, 1)
    
    def _extract_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """استخراج خريطة تحويل الألوان"""
        if input_grid.shape != output_grid.shape:
            return {}
        
        color_map = {}
        unique_inputs = np.unique(input_grid)
        
        for color in unique_inputs:
            mask = input_grid == color
            output_colors = output_grid[mask]
            unique_outputs = np.unique(output_colors)
            
            if len(unique_outputs) == 1:
                color_map[int(color)] = int(unique_outputs[0])
        
        return color_map
    
    def _is_consistent_color_mapping(self, color_maps: List[Dict]) -> bool:
        """فحص ثبات خريطة الألوان"""
        if not color_maps:
            return False
        
        first_map = color_maps[0]
        return all(cm == first_map for cm in color_maps[1:])
    
    def _is_repetition_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص نمط التكرار"""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        # فحص التكرار الأفقي أو العمودي
        return (w_out % w_in == 0 and h_out == h_in) or (h_out % h_in == 0 and w_out == w_in)
    
    def _get_repetition_details(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[str, int]:
        """الحصول على تفاصيل التكرار"""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        if w_out % w_in == 0 and h_out == h_in:
            return 'horizontal', w_out // w_in
        elif h_out % h_in == 0 and w_out == w_in:
            return 'vertical', h_out // h_in
        elif h_out % h_in == 0 and w_out % w_in == 0:
            return 'both', max(h_out // h_in, w_out // w_in)
        
        return 'none', 1
    
    def _is_pattern_completion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص إكمال النمط"""
        # منطق بسيط: إذا كان الإخراج أكبر من الإدخال ويحتوي عليه
        return output_grid.shape[0] >= input_grid.shape[0] and output_grid.shape[1] >= input_grid.shape[1]
    
    def _extract_completion_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Callable]:
        """استخراج قاعدة الإكمال"""
        # هذا يحتاج تطوير أكثر تعقيداً
        # لكن الفكرة هي إيجاد النمط المفقود وإكماله
        
        def simple_completion(grid):
            # قاعدة بسيطة: توسيع الشبكة بنفس النمط
            return np.pad(grid, ((0, 1), (0, 1)), mode='wrap')
        
        return simple_completion

class ARCCoreSolver:
    """الحلال الأساسي المبسط والفعال"""
    
    def __init__(self):
        self.rule_extractor = AbstractRuleExtractor()
        self.solution_cache = {}
    
    def solve_task(self, task_data: Dict) -> List[np.ndarray]:
        """حل مهمة ARC وإرجاع حل لكل إدخال اختبار"""
        train_pairs = task_data.get('train', [])
        test_items = task_data.get('test', [])
        
        # استخراج القواعد من أمثلة التدريب
        patterns = self.rule_extractor.extract_rule_from_examples(train_pairs)
        
        if not patterns:
            # في حال عدم وجود أنماط، أعد هوية الإدخال لكل اختبار
            return [self._fallback_solution(np.array(ti['input'])) for ti in test_items]
        
        # تطبيق أفضل قاعدة على جميع اختبارات التقييم
        best_pattern = patterns[0]
        try:
            solutions = [best_pattern.rule(np.array(ti['input'])) for ti in test_items]
            return solutions
        except Exception:
            return [self._fallback_solution(np.array(ti['input'])) for ti in test_items]
    
    def _fallback_solution(self, test_input: np.ndarray) -> np.ndarray:
        """حل احتياطي"""
        # في حالة الفشل، إرجاع نفس الإدخال
        return test_input

# مثال على الاستخدام
if __name__ == "__main__":
    solver = ARCCoreSolver()
    
    # مثال على مهمة بسيطة
    sample_task = {
        'train': [
            {
                'input': [[1, 2], [3, 4]], 
                'output': [[2, 1], [4, 3]]
            },
            {
                'input': [[5, 6], [7, 8]], 
                'output': [[6, 5], [8, 7]]
            }
        ],
        'test': [{'input': [[9, 0], [1, 2]]}]
    }
    
    solution = solver.solve_task(sample_task)
    print("الحل المقترح:")
    print(solution)
