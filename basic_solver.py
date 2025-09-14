from __future__ import annotations
#!/usr/bin/env python3
"""
Basic ARC Solver
نظام حل أساسي لمهام ARC
"""

import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any
import json

class BasicARCSolver:
    """حل أساسي لمهام ARC"""
    
    def __init__(self):
        self.transformations = [
            self.identity,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.flip_horizontal,
            self.flip_vertical,
            self.transpose,
            self.inverse_colors
        ]
        
    def solve(self, task_data: Dict) -> np.ndarray:
        """حل المهمة"""
        if 'train' not in task_data or not task_data['train']:
            return np.zeros((3, 3))
            
        # تحليل أمثلة التدريب
        train_examples = task_data['train']
        
        # محاولة إيجاد التحويل المناسب
        best_transform = None
        best_score = 0
        
        for transform in self.transformations:
            score = self.evaluate_transform(transform, train_examples)
            if score > best_score:
                best_score = score
                best_transform = transform
                
        # تطبيق أفضل تحويل على الاختبار
        if 'test' in task_data and task_data['test']:
            test_input = np.array(task_data['test'][0]['input'])
            if best_transform:
                return best_transform(test_input)
            else:
                # إذا لم نجد تحويل مناسب، نرجع أول output من التدريب
                return np.array(train_examples[0]['output'])
        else:
            return np.array(train_examples[0]['output'])
            
    def evaluate_transform(self, transform, examples):
        """تقييم التحويل على الأمثلة"""
        total_score = 0
        count = 0
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            try:
                transformed = transform(input_grid)
                if transformed.shape == output_grid.shape:
                    score = np.mean(transformed == output_grid)
                    total_score += score
                    count += 1
            except:
                continue
                
        return total_score / max(count, 1)
        
    # التحولات الأساسية
    def identity(self, grid):
        return grid
        
    def rotate_90(self, grid):
        return np.rot90(grid, k=1)
        
    def rotate_180(self, grid):
        return np.rot90(grid, k=2)
        
    def rotate_270(self, grid):
        return np.rot90(grid, k=3)
        
    def flip_horizontal(self, grid):
        return np.flip(grid, axis=0)
        
    def flip_vertical(self, grid):
        return np.flip(grid, axis=1)
        
    def transpose(self, grid):
        return np.transpose(grid)
        
    def inverse_colors(self, grid):
        max_val = np.max(grid)
        if max_val > 0:
            return max_val - grid
        return grid

# دالة للاستخدام المباشر
def solve_task(task_data: Dict) -> np.ndarray:
    solver = BasicARCSolver()
    return solver.solve(task_data)
