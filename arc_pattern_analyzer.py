from __future__ import annotations
#!/usr/bin/env python3
"""
محلل أنماط ARC الثوري
====================
تحليل عميق لأنماط ARC المعقدة وفهم القواعد الأساسية
"""
import json
import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class ARCPattern:
    """نمط ARC محدد"""
    pattern_type: str
    confidence: float
    description: str
    transformation_rules: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]

class RevolutionaryARCPatternAnalyzer:
    """محلل الأنماط الثوري لـ ARC"""
    
    def __init__(self):
        self.patterns_database = {}
        self.learned_patterns = {}
        self.transformation_rules = {}
        self._initialize_core_patterns()
    
    def _initialize_core_patterns(self):
        """تهيئة الأنماط الأساسية لـ ARC"""
        
        # 1. أنماط التكبير والتصغير
        self.patterns_database['scaling'] = {
            'horizontal_scale': {
                'description': 'تكبير أفقي للصورة',
                'rules': ['repeat_horizontal', 'stretch_horizontal'],
                'confidence': 0.9
            },
            'vertical_scale': {
                'description': 'تكبير عمودي للصورة', 
                'rules': ['repeat_vertical', 'stretch_vertical'],
                'confidence': 0.9
            },
            'uniform_scale': {
                'description': 'تكبير موحد للصورة',
                'rules': ['repeat_both', 'stretch_both'],
                'confidence': 0.95
            }
        }
        
        # 2. أنماط التماثل
        self.patterns_database['symmetry'] = {
            'horizontal_symmetry': {
                'description': 'تماثل أفقي',
                'rules': ['flip_horizontal', 'mirror_horizontal'],
                'confidence': 0.9
            },
            'vertical_symmetry': {
                'description': 'تماثل عمودي',
                'rules': ['flip_vertical', 'mirror_vertical'], 
                'confidence': 0.9
            },
            'rotational_symmetry': {
                'description': 'تماثل دوراني',
                'rules': ['rotate_90', 'rotate_180', 'rotate_270'],
                'confidence': 0.85
            }
        }
        
        # 3. أنماط الألوان
        self.patterns_database['color'] = {
            'color_mapping': {
                'description': 'تحويل الألوان',
                'rules': ['map_colors', 'replace_colors', 'cycle_colors'],
                'confidence': 0.9
            },
            'color_inversion': {
                'description': 'عكس الألوان',
                'rules': ['invert_colors', 'complement_colors'],
                'confidence': 0.8
            }
        }
        
        # 4. أنماط الحركة
        self.patterns_database['movement'] = {
            'translation': {
                'description': 'نقل العناصر',
                'rules': ['move_objects', 'shift_patterns'],
                'confidence': 0.85
            },
            'rotation': {
                'description': 'دوران العناصر',
                'rules': ['rotate_objects', 'spin_patterns'],
                'confidence': 0.8
            }
        }
        
        # 5. أنماط التركيب
        self.patterns_database['composition'] = {
            'pattern_repetition': {
                'description': 'تكرار الأنماط',
                'rules': ['repeat_pattern', 'tile_pattern'],
                'confidence': 0.9
            },
            'pattern_combination': {
                'description': 'دمج الأنماط',
                'rules': ['combine_patterns', 'merge_patterns'],
                'confidence': 0.8
            }
        }
    
    def analyze_comprehensive_patterns(self, input_grid: np.ndarray, 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """تحليل شامل للأنماط"""
        
        analysis = {
            'input_shape': input_grid.shape,
            'unique_colors': len(np.unique(input_grid)),
            'detected_patterns': [],
            'transformation_suggestions': [],
            'confidence_scores': {}
        }
        
        # تحليل التكبير/التصغير
        scaling_patterns = self._analyze_scaling_patterns(input_grid)
        analysis['detected_patterns'].extend(scaling_patterns)
        
        # تحليل التماثل
        symmetry_patterns = self._analyze_symmetry_patterns(input_grid)
        analysis['detected_patterns'].extend(symmetry_patterns)
        
        # تحليل الألوان
        color_patterns = self._analyze_color_patterns(input_grid)
        analysis['detected_patterns'].extend(color_patterns)
        
        # تحليل الحركة
        movement_patterns = self._analyze_movement_patterns(input_grid)
        analysis['detected_patterns'].extend(movement_patterns)
        
        # تحليل التركيب
        composition_patterns = self._analyze_composition_patterns(input_grid)
        analysis['detected_patterns'].extend(composition_patterns)
        
        # اقتراح التحويلات
        analysis['transformation_suggestions'] = self._suggest_transformations(analysis['detected_patterns'])
        
        return analysis
    
    def _analyze_scaling_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """تحليل أنماط التكبير/التصغير"""
        patterns = []
        
        # تحليل التكبير الأفقي
        if self._is_horizontal_scalable(grid):
            patterns.append({
                'type': 'horizontal_scaling',
                'confidence': 0.9,
                'scale_factor': self._calculate_scale_factor(grid, 'horizontal'),
                'description': 'يمكن تكبير الصورة أفقياً'
            })
        
        # تحليل التكبير العمودي
        if self._is_vertical_scalable(grid):
            patterns.append({
                'type': 'vertical_scaling', 
                'confidence': 0.9,
                'scale_factor': self._calculate_scale_factor(grid, 'vertical'),
                'description': 'يمكن تكبير الصورة عمودياً'
            })
        
        return patterns
    
    def _analyze_symmetry_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """تحليل أنماط التماثل"""
        patterns = []
        
        # تماثل أفقي
        if self._is_horizontally_symmetric(grid):
            patterns.append({
                'type': 'horizontal_symmetry',
                'confidence': 0.95,
                'description': 'الصورة متناظرة أفقياً'
            })
        
        # تماثل عمودي
        if self._is_vertically_symmetric(grid):
            patterns.append({
                'type': 'vertical_symmetry',
                'confidence': 0.95,
                'description': 'الصورة متناظرة عمودياً'
            })
        
        return patterns
    
    def _analyze_color_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """تحليل أنماط الألوان"""
        patterns = []
        
        unique_colors = np.unique(grid)
        
        # تحليل توزيع الألوان
        color_distribution = {}
        for color in unique_colors:
            count = np.sum(grid == color)
            color_distribution[color] = count
        
        # نمط تحويل الألوان
        if len(unique_colors) > 1:
            patterns.append({
                'type': 'color_transformation',
                'confidence': 0.8,
                'color_mapping': self._suggest_color_mapping(unique_colors),
                'description': 'تحويل محتمل للألوان'
            })
        
        return patterns
    
    def _analyze_movement_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """تحليل أنماط الحركة"""
        patterns = []
        
        # تحليل النقل
        if self._has_translation_pattern(grid):
            patterns.append({
                'type': 'translation',
                'confidence': 0.8,
                'direction': self._detect_translation_direction(grid),
                'description': 'نمط نقل محتمل'
            })
        
        return patterns
    
    def _analyze_composition_patterns(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """تحليل أنماط التركيب"""
        patterns = []
        
        # تحليل التكرار
        if self._has_repetition_pattern(grid):
            patterns.append({
                'type': 'repetition',
                'confidence': 0.9,
                'repeat_count': self._count_repetitions(grid),
                'description': 'نمط تكرار محتمل'
            })
        
        return patterns
    
    def _is_horizontal_scalable(self, grid: np.ndarray) -> bool:
        """فحص إمكانية التكبير الأفقي"""
        h, w = grid.shape
        if w < 2:
            return False
        
        # فحص التكرار الأفقي
        for i in range(h):
            row = grid[i, :]
            if len(set(row)) == 1:  # صف متجانس
                return True
        
        return False
    
    def _is_vertical_scalable(self, grid: np.ndarray) -> bool:
        """فحص إمكانية التكبير العمودي"""
        h, w = grid.shape
        if h < 2:
            return False
        
        # فحص التكرار العمودي
        for j in range(w):
            col = grid[:, j]
            if len(set(col)) == 1:  # عمود متجانس
                return True
        
        return False
    
    def _is_horizontally_symmetric(self, grid: np.ndarray) -> bool:
        """فحص التماثل الأفقي"""
        return np.array_equal(grid, np.fliplr(grid))
    
    def _is_vertically_symmetric(self, grid: np.ndarray) -> bool:
        """فحص التماثل العمودي"""
        return np.array_equal(grid, np.flipud(grid))
    
    def _has_translation_pattern(self, grid: np.ndarray) -> bool:
        """فحص نمط النقل"""
        # فحص بسيط للنقل
        h, w = grid.shape
        if h < 2 or w < 2:
            return False
        
        # فحص النقل الأفقي
        for shift in range(1, w):
            if np.array_equal(grid[:, shift:], grid[:, :-shift]):
                return True
        
        return False
    
    def _has_repetition_pattern(self, grid: np.ndarray) -> bool:
        """فحص نمط التكرار"""
        h, w = grid.shape
        
        # فحص التكرار الأفقي
        if w >= 4:
            half_w = w // 2
            if np.array_equal(grid[:, :half_w], grid[:, half_w:2*half_w]):
                return True
        
        # فحص التكرار العمودي
        if h >= 4:
            half_h = h // 2
            if np.array_equal(grid[:half_h, :], grid[half_h:2*half_h, :]):
                return True
        
        return False
    
    def _calculate_scale_factor(self, grid: np.ndarray, direction: str) -> int:
        """حساب عامل التكبير"""
        if direction == 'horizontal':
            w = grid.shape[1]
            return max(2, w // 2)
        else:
            h = grid.shape[0]
            return max(2, h // 2)
    
    def _suggest_color_mapping(self, colors: np.ndarray) -> Dict[int, int]:
        """اقتراح تحويل الألوان"""
        mapping = {}
        for i, color in enumerate(colors):
            if color != 0:  # لا نحول الخلفية
                mapping[color] = (color + 1) % 10  # تحويل بسيط
        return mapping
    
    def _detect_translation_direction(self, grid: np.ndarray) -> str:
        """كشف اتجاه النقل"""
        h, w = grid.shape
        for shift in range(1, w):
            if np.array_equal(grid[:, shift:], grid[:, :-shift]):
                return 'right'
        return 'unknown'
    
    def _count_repetitions(self, grid: np.ndarray) -> int:
        """عد التكرارات"""
        h, w = grid.shape
        max_repeats = 1
        
        # فحص التكرار الأفقي
        for i in range(h):
            row = grid[i, :]
            for pattern_len in range(1, w // 2 + 1):
                if w % pattern_len == 0:
                    pattern = row[:pattern_len]
                    repeats = w // pattern_len
                    if np.array_equal(row, np.tile(pattern, repeats)):
                        max_repeats = max(max_repeats, repeats)
        
        return max_repeats
    
    def _suggest_transformations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """اقتراح التحويلات بناءً على الأنماط المكتشفة"""
        transformations = []
        
        for pattern in patterns:
            if pattern['type'] == 'horizontal_scaling':
                transformations.append({
                    'type': 'scale_horizontal',
                    'factor': pattern.get('scale_factor', 2),
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'vertical_scaling':
                transformations.append({
                    'type': 'scale_vertical', 
                    'factor': pattern.get('scale_factor', 2),
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'horizontal_symmetry':
                transformations.append({
                    'type': 'flip_horizontal',
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'vertical_symmetry':
                transformations.append({
                    'type': 'flip_vertical',
                    'confidence': pattern['confidence']
                })
        
        return transformations

# إنشاء مثيل عالمي
revolutionary_analyzer = RevolutionaryARCPatternAnalyzer()
