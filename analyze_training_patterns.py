from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 تحليل الأنماط في مهام التدريب ARC
===================================

هذا السكريپت يحلل مهام التدريب لاستخراج الأنماط والخصائص
المشتركة التي يمكن أن تساعد في تحسين النظام.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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
        'rotation_patterns': [],
        'scaling_patterns': []
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
        'consistency_score': 0
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
        'symmetry_patterns': defaultdict(int)
    }
    
    for problem_id, challenge_data in challenges.items():
        
        # Analyze complexity
        complexity = analyze_problem_complexity(challenge_data)
        stats['complexity_distribution'].append(complexity)
        
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
    
    return stats


def create_visualizations(stats: Dict[str, Any]):
    """إنشاء الرسوم البيانية"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔍 تحليل أنماط مهام التدريب ARC', fontsize=16, fontweight='bold')
    
    # 1. Grid size distribution
    sizes = stats['grid_sizes']
    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    
    axes[0, 0].scatter(widths, heights, alpha=0.6)
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Height')
    axes[0, 0].set_title('Grid Size Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Color usage
    colors = list(stats['color_usage'].keys())
    counts = list(stats['color_usage'].values())
    
    axes[0, 1].bar(colors, counts)
    axes[0, 1].set_xlabel('Color ID')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Color Usage Distribution')
    
    # 3. Complexity distribution
    complexities = [c['avg_input_size'] for c in stats['complexity_distribution']]
    
    axes[0, 2].hist(complexities, bins=20, alpha=0.7)
    axes[0, 2].set_xlabel('Average Input Size')
    axes[0, 2].set_ylabel('Number of Problems')
    axes[0, 2].set_title('Problem Complexity Distribution')
    
    # 4. Training examples per problem
    train_counts = [c['num_train_examples'] for c in stats['complexity_distribution']]
    
    axes[1, 0].hist(train_counts, bins=range(1, max(train_counts)+2), alpha=0.7)
    axes[1, 0].set_xlabel('Number of Training Examples')
    axes[1, 0].set_ylabel('Number of Problems')
    axes[1, 0].set_title('Training Examples Distribution')
    
    # 5. Symmetry patterns
    symmetry_types = list(stats['symmetry_patterns'].keys())
    symmetry_counts = list(stats['symmetry_patterns'].values())
    
    if symmetry_types:
        axes[1, 1].pie(symmetry_counts, labels=symmetry_types, autopct='%1.1f%%')
        axes[1, 1].set_title('Symmetry Patterns')
    
    # 6. Colors per problem
    colors_per_problem = [c['max_colors_used'] for c in stats['complexity_distribution']]
    
    axes[1, 2].hist(colors_per_problem, bins=range(1, max(colors_per_problem)+2), alpha=0.7)
    axes[1, 2].set_xlabel('Number of Colors Used')
    axes[1, 2].set_ylabel('Number of Problems')
    axes[1, 2].set_title('Colors per Problem Distribution')
    
    plt.tight_layout()
    plt.savefig('training_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


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
    
    # Complexity analysis
    complexities = stats['complexity_distribution']
    avg_train_examples = np.mean([c['num_train_examples'] for c in complexities])
    avg_colors = np.mean([c['max_colors_used'] for c in complexities])
    
    print("🧩 تحليل التعقيد:")
    print(f"  - متوسط أمثلة التدريب لكل مسألة: {avg_train_examples:.1f}")
    print(f"  - متوسط الألوان لكل مسألة: {avg_colors:.1f}")
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
        
        # Create visualizations
        print("📈 إنشاء الرسوم البيانية...")
        create_visualizations(stats)
        
        # Save statistics
        with open('training_patterns_stats.json', 'w', encoding='utf-8') as f:
            # Convert Counter objects to regular dicts for JSON serialization
            json_stats = stats.copy()
            json_stats['color_usage'] = dict(stats['color_usage'])
            json_stats['symmetry_patterns'] = dict(stats['symmetry_patterns'])
            
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print("💾 الإحصائيات محفوظة في: training_patterns_stats.json")
        print("🖼️  الرسوم البيانية محفوظة في: training_patterns_analysis.png")
        print("✅ انتهى التحليل بنجاح!")
        
    except Exception as e:
        logging.error(f"❌ خطأ في التحليل: {e}")
        raise


if __name__ == "__main__":
    main()
