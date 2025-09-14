# -*- coding: utf-8 -*-
"""
Pattern Discovery Engine - محرك اكتشاف الأنماط المتقدم
يحلل المهام الفاشلة ويستخرج أنماط ARC شائعة لتحسين الأداء
"""
from __future__ import annotations
from collections.abc import Callable
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# استيراد الأدوات الأساسية
from symbolic_rule_engine import Grid, Example, Task, dims, equal

PATTERNS_PATH = Path('discovered_patterns.json')


def load_patterns() -> Dict[str, Any]:
    if PATTERNS_PATH.exists():
        try:
            return json.loads(PATTERNS_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_patterns(patterns: Dict[str, Any]) -> None:
    try:
        PATTERNS_PATH.write_text(json.dumps(patterns, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def extract_grid_features(grid: Grid) -> Dict[str, Any]:
    """استخراج ملامح الشبكة للتحليل"""
    h, w = dims(grid)
    flat = [cell for row in grid for cell in row]
    colors = set(flat)
    
    # إحصائيات أساسية
    features = {
        'dims': (h, w),
        'colors': sorted(colors),
        'color_count': len(colors),
        'total_cells': h * w,
        'color_distribution': dict(Counter(flat))
    }
    
    # أنماط هندسية
    features['has_border'] = (
        all(grid[0][j] == grid[0][0] for j in range(w)) and
        all(grid[h-1][j] == grid[h-1][0] for j in range(w)) and
        all(grid[i][0] == grid[0][0] for i in range(h)) and
        all(grid[i][w-1] == grid[0][0] for i in range(h))
    )
    
    # تماثل
    features['symmetric_h'] = all(grid[i][j] == grid[i][w-1-j] for i in range(h) for j in range(w//2))
    features['symmetric_v'] = all(grid[i][j] == grid[h-1-i][j] for i in range(h//2) for j in range(w))
    
    return features


def analyze_transformation_pattern(train: List[Example]) -> Dict[str, Any]:
    """تحليل نمط التحويل من أمثلة التدريب"""
    if not train:
        return {}
    
    patterns = {
        'size_changes': [],
        'color_mappings': [],
        'geometric_transforms': [],
        'common_features': {}
    }
    
    for ex in train:
        inp_features = extract_grid_features(ex['input'])
        out_features = extract_grid_features(ex['output'])
        
        # تغيير الحجم
        size_change = (out_features['dims'][0] / inp_features['dims'][0],
                      out_features['dims'][1] / inp_features['dims'][1])
        patterns['size_changes'].append(size_change)
        
        # تبديل الألوان
        inp_colors = set(inp_features['colors'])
        out_colors = set(out_features['colors'])
        if inp_colors != out_colors:
            patterns['color_mappings'].append({
                'input_colors': inp_features['colors'],
                'output_colors': out_features['colors']
            })
    
    # البحث عن أنماط شائعة
    if patterns['size_changes']:
        size_counter = Counter(patterns['size_changes'])
        most_common_size = size_counter.most_common(1)[0]
        if most_common_size[1] >= len(train) * 0.8:  # 80% من الأمثلة
            patterns['common_features']['consistent_size_change'] = most_common_size[0]
    
    return patterns


def discover_failure_patterns(failed_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """اكتشاف أنماط الفشل الشائعة"""
    failure_patterns = {
        'high_similarity_failures': [],
        'dimension_mismatches': [],
        'color_confusion': [],
        'geometric_errors': []
    }
    
    for task_result in failed_tasks:
        similarity = task_result.get('similarity', 0)
        
        if similarity > 0.85:  # فشل عالي التشابه
            failure_patterns['high_similarity_failures'].append({
                'task_id': task_result['task_id'],
                'similarity': similarity,
                'error_type': 'near_miss'
            })
        
        # تحليل أخطاء الأبعاد
        actual = task_result.get('actual_output')
        expected = task_result.get('expected_output')
        if actual is not None and expected is not None:
            if actual.shape != expected.shape:
                failure_patterns['dimension_mismatches'].append({
                    'task_id': task_result['task_id'],
                    'actual_dims': actual.shape,
                    'expected_dims': expected.shape
                })
    
    return failure_patterns


class PatternDiscoveryEngine:
    def __init__(self):
        self.patterns = load_patterns()
        self.discovered_rules = []
    
    def analyze_task_batch(self, task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تحليل دفعة من نتائج المهام لاكتشاف أنماط"""
        failed_tasks = [r for r in task_results if not r.get('solved_correctly', False)]
        successful_tasks = [r for r in task_results if r.get('solved_correctly', False)]
        
        analysis = {
            'total_tasks': len(task_results),
            'success_rate': len(successful_tasks) / len(task_results) if task_results else 0,
            'failure_patterns': discover_failure_patterns(failed_tasks),
            'high_potential_tasks': []
        }
        
        # تحديد المهام ذات الإمكانية العالية للتحسين
        for task_result in failed_tasks:
            if task_result.get('similarity', 0) > 0.9:
                analysis['high_potential_tasks'].append({
                    'task_id': task_result['task_id'],
                    'similarity': task_result['similarity'],
                    'priority': 'high'
                })
        
        return analysis
    
    def suggest_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """اقتراح تحسينات بناءً على التحليل"""
        suggestions = []
        
        # تحسينات للمهام عالية التشابه
        high_sim_count = len(analysis['failure_patterns']['high_similarity_failures'])
        if high_sim_count > 0:
            suggestions.append({
                'type': 'color_correction',
                'description': f'إضافة تصحيح لوني للمهام عالية التشابه ({high_sim_count} مهمة)',
                'priority': 'high',
                'implementation': 'enhance_color_mapping'
            })
        
        # تحسينات لأخطاء الأبعاد
        dim_errors = len(analysis['failure_patterns']['dimension_mismatches'])
        if dim_errors > 0:
            suggestions.append({
                'type': 'dimension_handling',
                'description': f'تحسين معالجة الأبعاد ({dim_errors} خطأ)',
                'priority': 'medium',
                'implementation': 'improve_scaling'
            })
        
        return suggestions
    
    def update_patterns(self, new_patterns: Dict[str, Any]) -> None:
        """تحديث قاعدة الأنماط المكتشفة"""
        self.patterns.update(new_patterns)
        save_patterns(self.patterns)
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى من الأنماط المكتشفة"""
        return {
            'total_patterns': len(self.patterns),
            'most_common_failures': self._get_common_failures(),
            'success_predictors': self._get_success_predictors()
        }
    
    def _get_common_failures(self) -> List[str]:
        """الحصول على أكثر أنماط الفشل شيوعاً"""
        # تحليل بسيط للأنماط المحفوظة
        return ['color_mapping_errors', 'dimension_mismatches', 'geometric_transforms']
    
    def _get_success_predictors(self) -> List[str]:
        """الحصول على مؤشرات النجاح"""
        return ['consistent_transformations', 'simple_color_maps', 'preserved_structure']


def create_adaptive_strategy(pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """إنشاء استراتيجية تكيفية بناءً على تحليل الأنماط"""
    strategy = {
        'focus_areas': [],
        'optimization_targets': [],
        'rule_priorities': {}
    }
    
    # تحديد مناطق التركيز
    if pattern_analysis.get('failure_patterns', {}).get('high_similarity_failures'):
        strategy['focus_areas'].append('color_correction')
        strategy['rule_priorities']['color_mapping'] = 'high'
    
    if pattern_analysis.get('failure_patterns', {}).get('dimension_mismatches'):
        strategy['focus_areas'].append('scaling_improvement')
        strategy['rule_priorities']['geometric_transforms'] = 'medium'
    
    return strategy
