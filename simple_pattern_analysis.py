from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 تحليل بسيط للأنماط في مهام التدريب ARC
==========================================

تحليل مبسط بدون matplotlib لفهم خصائص مهام التدريب.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def load_training_data() -> Dict[str, Any]:
    """تحميل بيانات التدريب"""
    
    challenges_file = Path("arc-agi_training_challenges.json")
    
    if not challenges_file.exists():
        raise FileNotFoundError(f"Training challenges file not found: {challenges_file}")
    
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    logging.info(f"📚 Loaded {len(challenges)} training challenges")
    return challenges


def analyze_grid_properties(grid: List[List[int]]) -> Dict[str, Any]:
    """تحليل خصائص الشبكة"""
    
    grid_np = np.array(grid)
    height, width = grid_np.shape
    
    # Basic properties
    properties = {
        'height': height,
        'width': width,
        'area': height * width,
        'aspect_ratio': width / height,
        'unique_colors': len(np.unique(grid_np)),
        'color_distribution': Counter(grid_np.flatten()),
        'is_square': height == width,
        'max_dimension': max(height, width),
        'min_dimension': min(height, width)
    }
    
    # Pattern analysis
    properties['has_symmetry_h'] = np.array_equal(grid_np, np.fliplr(grid_np))
    properties['has_symmetry_v'] = np.array_equal(grid_np, np.flipud(grid_np))
    properties['has_symmetry_d1'] = np.array_equal(grid_np, grid_np.T)
    
    # Color analysis
    background_color = Counter(grid_np.flatten()).most_common(1)[0][0]
    properties['background_color'] = background_color
    properties['background_ratio'] = properties['color_distribution'][background_color] / properties['area']
    
    # Connected components (simple version)
    non_background = grid_np != background_color
    properties['non_background_cells'] = np.sum(non_background)
    properties['density'] = properties['non_background_cells'] / properties['area']
    
    return properties


def analyze_transformation_patterns(train_pairs: List[Dict]) -> Dict[str, Any]:
    """تحليل أنماط التحويل"""
    
    patterns = {
        'size_changes': [],
        'color_changes': [],
        'shape_preservation': [],
        'scaling_factors': []
    }
    
    for pair in train_pairs:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])
        
        input_props = analyze_grid_properties(pair['input'])
        output_props = analyze_grid_properties(pair['output'])
        
        # Size changes
        size_change = {
            'input_size': input_props['area'],
            'output_size': output_props['area'],
            'size_ratio': output_props['area'] / input_props['area'],
            'width_change': output_props['width'] - input_props['width'],
            'height_change': output_props['height'] - input_props['height']
        }
        patterns['size_changes'].append(size_change)
        
        # Scaling factors
        width_scale = output_props['width'] / input_props['width']
        height_scale = output_props['height'] / input_props['height']
        patterns['scaling_factors'].append({
            'width_scale': width_scale,
            'height_scale': height_scale,
            'uniform_scale': abs(width_scale - height_scale) < 0.1
        })
        
        # Color changes
        input_colors = set(input_props['color_distribution'].keys())
        output_colors = set(output_props['color_distribution'].keys())
        
        color_change = {
            'input_colors': len(input_colors),
            'output_colors': len(output_colors),
            'new_colors': len(output_colors - input_colors),
            'removed_colors': len(input_colors - output_colors),
            'preserved_colors': len(input_colors & output_colors)
        }
        patterns['color_changes'].append(color_change)
        
        # Shape preservation
        if input_grid.shape == output_grid.shape:
            patterns['shape_preservation'].append(True)
        else:
            patterns['shape_preservation'].append(False)
    
    return patterns


def analyze_problem_complexity(challenge_data: Dict) -> Dict[str, Any]:
    """تحليل تعقيد المسألة"""
    
    train_pairs = challenge_data.get('train', [])
    test_pairs = challenge_data.get('test', [])
    
    complexity = {
        'num_train_examples': len(train_pairs),
        'num_test_examples': len(test_pairs),
        'avg_input_size': 0,
        'avg_output_size': 0,
        'max_colors_used': 0,
        'size_consistency': 0,
        'transformation_type': 'unknown'
    }
    
    if not train_pairs:
        return complexity
    
    # Calculate averages
    input_sizes = []
    output_sizes = []
    colors_used = set()
    
    for pair in train_pairs:
        input_props = analyze_grid_properties(pair['input'])
        output_props = analyze_grid_properties(pair['output'])
        
        input_sizes.append(input_props['area'])
        output_sizes.append(output_props['area'])
        colors_used.update(input_props['color_distribution'].keys())
        colors_used.update(output_props['color_distribution'].keys())
    
    complexity['avg_input_size'] = np.mean(input_sizes)
    complexity['avg_output_size'] = np.mean(output_sizes)
    complexity['max_colors_used'] = len(colors_used)
    
    # Analyze transformation patterns
    transformation_patterns = analyze_transformation_patterns(train_pairs)
    
    # Calculate consistency score based on size changes
    size_ratios = [sc['size_ratio'] for sc in transformation_patterns['size_changes']]
    if size_ratios:
        complexity['size_consistency'] = 1.0 - np.std(size_ratios)
    
    # Determine transformation type
    scaling_factors = transformation_patterns['scaling_factors']
    if scaling_factors:
        uniform_scales = [sf['uniform_scale'] for sf in scaling_factors]
        if all(uniform_scales):
            complexity['transformation_type'] = 'uniform_scaling'
        elif any(sf['width_scale'] != 1 or sf['height_scale'] != 1 for sf in scaling_factors):
            complexity['transformation_type'] = 'scaling'
        elif all(transformation_patterns['shape_preservation']):
            complexity['transformation_type'] = 'shape_preserving'
        else:
            complexity['transformation_type'] = 'shape_changing'
    
    return complexity


def generate_statistics(challenges: Dict[str, Any]) -> Dict[str, Any]:
    """إنتاج إحصائيات شاملة"""
    
    stats = {
        'total_problems': len(challenges),
        'grid_sizes': [],
        'color_usage': Counter(),
        'complexity_distribution': [],
        'transformation_types': defaultdict(int),
        'size_patterns': [],
        'symmetry_patterns': defaultdict(int),
        'scaling_patterns': defaultdict(int)
    }
    
    for problem_id, challenge_data in challenges.items():
        
        # Analyze complexity
        complexity = analyze_problem_complexity(challenge_data)
        stats['complexity_distribution'].append(complexity)
        stats['transformation_types'][complexity['transformation_type']] += 1
        
        # Analyze all grids in the problem
        all_grids = []
        
        # Training examples
        for pair in challenge_data.get('train', []):
            all_grids.extend([pair['input'], pair['output']])
        
        # Test examples
        for pair in challenge_data.get('test', []):
            all_grids.append(pair['input'])
        
        # Analyze each grid
        for grid in all_grids:
            props = analyze_grid_properties(grid)
            
            stats['grid_sizes'].append((props['height'], props['width']))
            stats['color_usage'].update(props['color_distribution'])
            
            # Symmetry patterns
            if props['has_symmetry_h']:
                stats['symmetry_patterns']['horizontal'] += 1
            if props['has_symmetry_v']:
                stats['symmetry_patterns']['vertical'] += 1
            if props['has_symmetry_d1']:
                stats['symmetry_patterns']['diagonal'] += 1
        
        # Analyze scaling patterns
        train_pairs = challenge_data.get('train', [])
        if train_pairs:
            transformation_patterns = analyze_transformation_patterns(train_pairs)
            for sf in transformation_patterns['scaling_factors']:
                if sf['uniform_scale']:
                    scale_factor = sf['width_scale']
                    if scale_factor == 2:
                        stats['scaling_patterns']['2x'] += 1
                    elif scale_factor == 3:
                        stats['scaling_patterns']['3x'] += 1
                    elif scale_factor == 0.5:
                        stats['scaling_patterns']['0.5x'] += 1
                    elif scale_factor == 1:
                        stats['scaling_patterns']['1x'] += 1
                    else:
                        stats['scaling_patterns']['other'] += 1
    
    return stats


def print_summary(stats: Dict[str, Any]):
    """طباعة ملخص التحليل"""
    
    print("\n" + "="*80)
    print("📊 ملخص تحليل أنماط مهام التدريب")
    print("="*80)
    
    print(f"📚 إجمالي المهام: {stats['total_problems']}")
    print(f"🎨 إجمالي الألوان المستخدمة: {len(stats['color_usage'])}")
    print(f"📐 إجمالي الشبكات المحللة: {len(stats['grid_sizes'])}")
    print()
    
    # Grid size analysis
    sizes = stats['grid_sizes']
    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    
    print("📏 تحليل أحجام الشبكات:")
    print(f"  - متوسط الارتفاع: {np.mean(heights):.1f}")
    print(f"  - متوسط العرض: {np.mean(widths):.1f}")
    print(f"  - أكبر شبكة: {max(heights)}×{max(widths)}")
    print(f"  - أصغر شبكة: {min(heights)}×{min(widths)}")
    print()
    
    # Color analysis
    most_common_colors = stats['color_usage'].most_common(5)
    print("🎨 الألوان الأكثر استخداماً:")
    for color, count in most_common_colors:
        print(f"  - اللون {color}: {count} مرة")
    print()
    
    # Transformation types
    print("🔄 أنواع التحويلات:")
    for trans_type, count in stats['transformation_types'].items():
        percentage = (count / stats['total_problems']) * 100
        print(f"  - {trans_type}: {count} مهمة ({percentage:.1f}%)")
    print()
    
    # Scaling patterns
    if stats['scaling_patterns']:
        print("📈 أنماط التحجيم:")
        for pattern, count in stats['scaling_patterns'].items():
            print(f"  - {pattern}: {count} مرة")
        print()
    
    # Symmetry analysis
    if stats['symmetry_patterns']:
        print("🔄 أنماط التماثل:")
        for pattern, count in stats['symmetry_patterns'].items():
            print(f"  - {pattern}: {count} شبكة")
    
    print("\n" + "="*80)


def main():
    """الدالة الرئيسية"""
    
    print("🔍 بدء تحليل أنماط مهام التدريب...")
    
    try:
        # Load training data
        challenges = load_training_data()
        
        # Generate statistics
        print("📊 إنتاج الإحصائيات...")
        stats = generate_statistics(challenges)
        
        # Print summary
        print_summary(stats)
        
        # Save statistics
        with open('simple_training_patterns_stats.json', 'w', encoding='utf-8') as f:
            # Convert Counter objects and numpy types to regular dicts for JSON serialization
            json_stats = stats.copy()
            json_stats['color_usage'] = {str(k): int(v) for k, v in stats['color_usage'].items()}
            json_stats['symmetry_patterns'] = {str(k): int(v) for k, v in stats['symmetry_patterns'].items()}
            json_stats['transformation_types'] = {str(k): int(v) for k, v in stats['transformation_types'].items()}
            json_stats['scaling_patterns'] = {str(k): int(v) for k, v in stats['scaling_patterns'].items()}

            # Convert grid sizes to regular lists
            json_stats['grid_sizes'] = [(int(h), int(w)) for h, w in stats['grid_sizes']]

            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print("💾 الإحصائيات محفوظة في: simple_training_patterns_stats.json")
        print("✅ انتهى التحليل بنجاح!")
        
    except Exception as e:
        logging.error(f"❌ خطأ في التحليل: {e}")
        raise


if __name__ == "__main__":
    main()
