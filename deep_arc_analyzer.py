from __future__ import annotations
#!/usr/bin/env python3
"""
DEEP ARC ANALYZER - تحليل عميق لمهام المسابقة
===============================================
تحليل شامل لجميع مهام ARC لفهم الأنماط الحقيقية
"""

import json
import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import os

class DeepARCAnalyzer:
    """محلل عميق لمهام ARC"""
    
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.training_tasks = {}
        self.evaluation_tasks = {}
        self.evaluation_solutions = {}
        self.pattern_statistics = defaultdict(int)
        self.transformation_types = defaultdict(list)
        
    def load_all_data(self):
        """تحميل جميع بيانات المسابقة"""
        # تحميل مهام التدريب
        train_file = os.path.join(self.data_folder, "arc-agi_training_challenges.json")
        if os.path.exists(train_file):
            with open(train_file, 'r') as f:
                self.training_tasks = json.load(f)
            print(f"✅ تم تحميل {len(self.training_tasks)} مهمة تدريب")
        
        # تحميل مهام التقييم
        eval_file = os.path.join(self.data_folder, "arc-agi_evaluation_challenges.json")
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                self.evaluation_tasks = json.load(f)
            print(f"✅ تم تحميل {len(self.evaluation_tasks)} مهمة تقييم")
        
        # تحميل حلول التقييم
        sol_file = os.path.join(self.data_folder, "arc-agi_evaluation_solutions.json")
        if os.path.exists(sol_file):
            with open(sol_file, 'r') as f:
                self.evaluation_solutions = json.load(f)
            print(f"✅ تم تحميل حلول {len(self.evaluation_solutions)} مهمة")
    
    def analyze_all_patterns(self):
        """تحليل جميع الأنماط في المهام"""
        print("\n🔍 بدء التحليل العميق للأنماط...")
        
        # تحليل مهام التدريب
        for task_id, task_data in self.training_tasks.items():
            self._analyze_single_task(task_id, task_data, "training")
        
        # تحليل مهام التقييم
        for task_id, task_data in self.evaluation_tasks.items():
            self._analyze_single_task(task_id, task_data, "evaluation")
        
        self._generate_pattern_report()
    
    def _analyze_single_task(self, task_id: str, task_data: Dict, task_type: str):
        """تحليل مهمة واحدة"""
        train_examples = task_data.get('train', [])
        
        for i, example in enumerate(train_examples):
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # تحليل نوع التحويل
            transform_type = self._classify_transformation(input_grid, output_grid)
            self.transformation_types[transform_type].append(f"{task_id}_{i}")
            
            # تحليل خصائص الشبكة
            self._analyze_grid_properties(input_grid, output_grid, task_id)
    
    def _classify_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """تصنيف نوع التحويل"""
        
        # فحص تغيير الحجم
        if input_grid.shape != output_grid.shape:
            h_ratio = output_grid.shape[0] / input_grid.shape[0]
            w_ratio = output_grid.shape[1] / input_grid.shape[1]
            
            if h_ratio == w_ratio:
                if h_ratio > 1:
                    return f"size_expand_{int(h_ratio)}x"
                else:
                    return f"size_shrink_{int(1/h_ratio)}x"
            else:
                return "size_asymmetric"
        
        # فحص التحويلات المكانية
        if np.array_equal(output_grid, np.rot90(input_grid, 1)):
            return "rotate_90"
        elif np.array_equal(output_grid, np.rot90(input_grid, 2)):
            return "rotate_180"
        elif np.array_equal(output_grid, np.rot90(input_grid, 3)):
            return "rotate_270"
        elif np.array_equal(output_grid, np.fliplr(input_grid)):
            return "flip_horizontal"
        elif np.array_equal(output_grid, np.flipud(input_grid)):
            return "flip_vertical"
        elif np.array_equal(output_grid, input_grid.T):
            return "transpose"
        
        # فحص تحويلات الألوان
        if input_grid.shape == output_grid.shape:
            if self._is_color_mapping(input_grid, output_grid):
                return "color_mapping"
            elif self._is_color_shift(input_grid, output_grid):
                return "color_shift"
        
        # فحص أنماط التكرار
        if self._is_tiling_pattern(input_grid, output_grid):
            return "tiling_pattern"
        
        # فحص إكمال الأنماط
        if self._is_pattern_completion(input_grid, output_grid):
            return "pattern_completion"
        
        # فحص معالجة الكائنات
        if self._is_object_manipulation(input_grid, output_grid):
            return "object_manipulation"
        
        return "complex_unknown"
    
    def _is_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص إذا كان التحويل هو تحويل ألوان"""
        if input_grid.shape != output_grid.shape:
            return False
        
        color_map = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                input_color = input_grid[i, j]
                output_color = output_grid[i, j]
                
                if input_color in color_map:
                    if color_map[input_color] != output_color:
                        return False
                else:
                    color_map[input_color] = output_color
        
        return len(color_map) > 1  # يجب أن يكون هناك تحويل فعلي
    
    def _is_color_shift(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص إذا كان التحويل هو إزاحة ألوان"""
        if input_grid.shape != output_grid.shape:
            return False
        
        # فحص إزاحة ثابتة
        diff = output_grid - input_grid
        return np.all(diff == diff[0, 0])
    
    def _is_tiling_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص نمط التكرار"""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        # فحص إذا كان الإخراج مضاعف للإدخال
        if h_out % h_in == 0 and w_out % w_in == 0:
            h_factor = h_out // h_in
            w_factor = w_out // w_in
            
            # فحص إذا كان التكرار صحيح
            tiled = np.tile(input_grid, (h_factor, w_factor))
            return np.array_equal(tiled, output_grid)
        
        return False
    
    def _is_pattern_completion(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص إكمال النمط"""
        # منطق بسيط: إذا كان الإخراج أكبر ويحتوي على الإدخال
        if (output_grid.shape[0] >= input_grid.shape[0] and 
            output_grid.shape[1] >= input_grid.shape[1]):
            
            # فحص إذا كان الإدخال موجود في الإخراج
            h_in, w_in = input_grid.shape
            return np.array_equal(output_grid[:h_in, :w_in], input_grid)
        
        return False
    
    def _is_object_manipulation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """فحص معالجة الكائنات"""
        # فحص إذا كانت الألوان نفسها موجودة
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        
        return input_colors == output_colors and not np.array_equal(input_grid, output_grid)
    
    def _analyze_grid_properties(self, input_grid: np.ndarray, output_grid: np.ndarray, task_id: str):
        """تحليل خصائص الشبكة"""
        # حجم الشبكة
        self.pattern_statistics[f"input_size_{input_grid.shape}"] += 1
        self.pattern_statistics[f"output_size_{output_grid.shape}"] += 1
        
        # عدد الألوان
        input_colors = len(np.unique(input_grid))
        output_colors = len(np.unique(output_grid))
        self.pattern_statistics[f"input_colors_{input_colors}"] += 1
        self.pattern_statistics[f"output_colors_{output_colors}"] += 1
        
        # نسبة الألوان غير الصفرية
        input_nonzero = np.count_nonzero(input_grid) / input_grid.size
        output_nonzero = np.count_nonzero(output_grid) / output_grid.size
        
        self.pattern_statistics[f"input_density_{int(input_nonzero*10)}"] += 1
        self.pattern_statistics[f"output_density_{int(output_nonzero*10)}"] += 1
    
    def _generate_pattern_report(self):
        """إنتاج تقرير الأنماط"""
        print("\n📊 تقرير تحليل الأنماط:")
        print("=" * 50)
        
        print("\n🔄 أنواع التحويلات:")
        for transform_type, examples in self.transformation_types.items():
            print(f"  {transform_type}: {len(examples)} مثال")
        
        print(f"\n📈 إحصائيات الأنماط (أهم 20):")
        top_patterns = sorted(self.pattern_statistics.items(), 
                            key=lambda x: x[1], reverse=True)[:20]
        
        for pattern, count in top_patterns:
            print(f"  {pattern}: {count}")
        
        # حفظ التقرير
        self._save_analysis_report()
    
    def _save_analysis_report(self):
        """حفظ تقرير التحليل"""
        report = {
            'transformation_types': dict(self.transformation_types),
            'pattern_statistics': dict(self.pattern_statistics),
            'total_training_tasks': len(self.training_tasks),
            'total_evaluation_tasks': len(self.evaluation_tasks),
            'analysis_summary': self._generate_summary()
        }
        
        with open('arc_deep_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ التقرير في: arc_deep_analysis_report.json")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """إنتاج ملخص التحليل"""
        return {
            'most_common_transformations': [
                transform for transform, examples in 
                sorted(self.transformation_types.items(), 
                      key=lambda x: len(x[1]), reverse=True)[:10]
            ],
            'complexity_distribution': self._analyze_complexity(),
            'size_patterns': self._analyze_size_patterns(),
            'color_patterns': self._analyze_color_patterns()
        }
    
    def _analyze_complexity(self) -> Dict[str, int]:
        """تحليل توزيع التعقيد"""
        complexity = {'simple': 0, 'medium': 0, 'complex': 0}
        
        for transform_type, examples in self.transformation_types.items():
            if transform_type in ['rotate_90', 'rotate_180', 'flip_horizontal', 'flip_vertical']:
                complexity['simple'] += len(examples)
            elif transform_type in ['color_mapping', 'tiling_pattern', 'size_expand_2x']:
                complexity['medium'] += len(examples)
            else:
                complexity['complex'] += len(examples)
        
        return complexity
    
    def _analyze_size_patterns(self) -> Dict[str, int]:
        """تحليل أنماط الأحجام"""
        size_patterns = defaultdict(int)
        
        for pattern, count in self.pattern_statistics.items():
            if 'size_' in pattern:
                size_patterns[pattern] = count
        
        return dict(size_patterns)
    
    def _analyze_color_patterns(self) -> Dict[str, int]:
        """تحليل أنماط الألوان"""
        color_patterns = defaultdict(int)
        
        for pattern, count in self.pattern_statistics.items():
            if 'colors_' in pattern:
                color_patterns[pattern] = count
        
        return dict(color_patterns)

    def get_insights_for_solver(self) -> Dict[str, Any]:
        """استخراج رؤى لتطوير الحلال"""
        insights = {
            'priority_transformations': [],
            'common_patterns': [],
            'complexity_levels': {},
            'recommendations': []
        }
        
        # أهم التحويلات حسب التكرار
        sorted_transforms = sorted(self.transformation_types.items(), 
                                 key=lambda x: len(x[1]), reverse=True)
        
        insights['priority_transformations'] = [
            {'type': t, 'frequency': len(examples), 'examples': examples[:5]}
            for t, examples in sorted_transforms[:15]
        ]
        
        # الأنماط الشائعة
        insights['common_patterns'] = [
            {'pattern': p, 'count': c}
            for p, c in sorted(self.pattern_statistics.items(), 
                             key=lambda x: x[1], reverse=True)[:20]
        ]
        
        # مستويات التعقيد
        insights['complexity_levels'] = self._analyze_complexity()
        
        # توصيات للتطوير
        insights['recommendations'] = self._generate_development_recommendations()
        
        return insights
    
    def _generate_development_recommendations(self) -> List[str]:
        """إنتاج توصيات للتطوير"""
        recommendations = []
        
        # تحليل أهم التحويلات
        top_transforms = sorted(self.transformation_types.items(), 
                              key=lambda x: len(x[1]), reverse=True)[:5]
        
        recommendations.append(f"🎯 ركز على تطوير {len(top_transforms)} تحويلات رئيسية")
        
        for transform, examples in top_transforms:
            recommendations.append(f"  - {transform}: {len(examples)} مثال")
        
        # تحليل التعقيد
        complexity = self._analyze_complexity()
        total = sum(complexity.values())
        
        if complexity['complex'] / total > 0.3:
            recommendations.append("⚠️ 30%+ من المهام معقدة - طور خوارزميات متقدمة")
        
        if complexity['simple'] / total > 0.4:
            recommendations.append("✅ 40%+ من المهام بسيطة - ابدأ بالتحويلات الأساسية")
        
        return recommendations

# تشغيل التحليل
if __name__ == "__main__":
    data_folder = "ملفات المسابقةarc-prize-2025"
    
    analyzer = DeepARCAnalyzer(data_folder)
    analyzer.load_all_data()
    analyzer.analyze_all_patterns()
    
    # الحصول على رؤى للتطوير
    insights = analyzer.get_insights_for_solver()
    
    print("\n🚀 رؤى لتطوير الحلال:")
    print("=" * 40)
    
    for rec in insights['recommendations']:
        print(rec)
    
    print(f"\n📊 أهم {len(insights['priority_transformations'])} تحويلات:")
    for i, transform in enumerate(insights['priority_transformations'][:10], 1):
        print(f"{i}. {transform['type']}: {transform['frequency']} مرة")
