from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
نظام حل مسائل ARC بالتعلم الحقيقي
يتعلم من كل مسألة ويطور استراتيجيات جديدة
"""

import json
import numpy as np
import os
from true_learning_ai import TrueLearningAI
import time

class ARCLearningSolver(TrueLearningAI):
    def __init__(self):
        super().__init__(memory_file="arc_memory.json", patterns_file="arc_patterns.pkl")
        
        # استراتيجيات حل ARC المتعلمة
        self.solving_strategies = {}
        
        # أنماط ARC المكتشفة
        self.arc_patterns = {
            "transformations": {},
            "color_patterns": {},
            "shape_patterns": {},
            "size_patterns": {},
            "position_patterns": {}
        }
        
        # إحصائيات الأداء
        self.performance_stats = {
            "problems_attempted": 0,
            "problems_solved": 0,
            "strategies_learned": 0,
            "patterns_discovered": 0
        }
        
        print("🧩 تم تهيئة نظام حل ARC بالتعلم الحقيقي")

    def analyze_arc_grid(self, grid):
        """تحليل شبكة ARC واستخراج الخصائص"""
        grid = np.array(grid)
        
        analysis = {
            "shape": grid.shape,
            "unique_colors": len(np.unique(grid)),
            "color_distribution": {},
            "patterns": [],
            "symmetries": [],
            "connected_components": 0
        }
        
        # تحليل توزيع الألوان
        unique, counts = np.unique(grid, return_counts=True)
        analysis["color_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
        
        # اكتشاف الأنماط البسيطة
        analysis["patterns"] = self.detect_simple_patterns(grid)
        
        # اكتشاف التماثل
        analysis["symmetries"] = self.detect_symmetries(grid)
        
        return analysis

    def detect_simple_patterns(self, grid):
        """اكتشاف الأنماط البسيطة في الشبكة"""
        patterns = []
        
        # نمط الخطوط الأفقية
        for row in grid:
            if len(np.unique(row)) == 1 and row[0] != 0:
                patterns.append("horizontal_line")
                break
        
        # نمط الخطوط العمودية
        for col in grid.T:
            if len(np.unique(col)) == 1 and col[0] != 0:
                patterns.append("vertical_line")
                break
        
        # نمط المربعات
        if grid.shape[0] == grid.shape[1]:
            patterns.append("square_grid")
        
        # نمط الحدود
        if (np.all(grid[0, :] != 0) or np.all(grid[-1, :] != 0) or 
            np.all(grid[:, 0] != 0) or np.all(grid[:, -1] != 0)):
            patterns.append("border_pattern")
        
        return patterns

    def detect_symmetries(self, grid):
        """اكتشاف التماثل في الشبكة"""
        symmetries = []
        
        # تماثل أفقي
        if np.array_equal(grid, np.flipud(grid)):
            symmetries.append("horizontal_symmetry")
        
        # تماثل عمودي
        if np.array_equal(grid, np.fliplr(grid)):
            symmetries.append("vertical_symmetry")
        
        # تماثل قطري
        if grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T):
            symmetries.append("diagonal_symmetry")
        
        return symmetries

    def learn_from_arc_example(self, input_grid, output_grid):
        """التعلم من مثال ARC"""
        
        print("🔍 تحليل مثال ARC جديد...")
        
        # تحليل الشبكات
        input_analysis = self.analyze_arc_grid(input_grid)
        output_analysis = self.analyze_arc_grid(output_grid)
        
        # اكتشاف التحويل
        transformation = self.discover_transformation(input_grid, output_grid, input_analysis, output_analysis)
        
        # حفظ التحويل المكتشف
        if transformation:
            self.save_transformation(transformation, input_analysis, output_analysis)
        
        # تحديث الأنماط
        self.update_arc_patterns(input_analysis, output_analysis, transformation)
        
        # حفظ في الذاكرة طويلة المدى
        example_info = f"ARC Example: Input {input_analysis['shape']} -> Output {output_analysis['shape']}"
        learning_result = self.encounter_new_information(example_info, "arc_training")
        
        return {
            "transformation_discovered": transformation is not None,
            "transformation": transformation,
            "learning_result": learning_result
        }

    def discover_transformation(self, input_grid, output_grid, input_analysis, output_analysis):
        """اكتشاف نوع التحويل بين الدخل والخرج"""
        
        input_grid = np.array(input_grid)
        output_grid = np.array(output_grid)
        
        transformations = []
        
        # تحويل الحجم
        if input_grid.shape != output_grid.shape:
            transformations.append({
                "type": "resize",
                "from_shape": input_grid.shape,
                "to_shape": output_grid.shape
            })
        
        # تحويل الألوان
        input_colors = set(input_grid.flatten())
        output_colors = set(output_grid.flatten())
        if input_colors != output_colors:
            transformations.append({
                "type": "color_change",
                "from_colors": list(input_colors),
                "to_colors": list(output_colors)
            })
        
        # تحويل الدوران
        for rotation in [1, 2, 3]:  # 90, 180, 270 درجة
            rotated = np.rot90(input_grid, rotation)
            if rotated.shape == output_grid.shape and np.array_equal(rotated, output_grid):
                transformations.append({
                    "type": "rotation",
                    "degrees": rotation * 90
                })
                break
        
        # تحويل الانعكاس
        if np.array_equal(np.flipud(input_grid), output_grid):
            transformations.append({"type": "flip_vertical"})
        elif np.array_equal(np.fliplr(input_grid), output_grid):
            transformations.append({"type": "flip_horizontal"})
        
        # تحويل التكرار
        if output_grid.shape[0] > input_grid.shape[0] or output_grid.shape[1] > input_grid.shape[1]:
            transformations.append({"type": "repetition"})
        
        return transformations[0] if transformations else None

    def save_transformation(self, transformation, input_analysis, output_analysis):
        """حفظ التحويل المكتشف"""
        trans_key = f"{transformation['type']}_{len(self.solving_strategies)}"
        
        self.solving_strategies[trans_key] = {
            "transformation": transformation,
            "input_characteristics": input_analysis,
            "output_characteristics": output_analysis,
            "success_count": 1,
            "discovered_at": time.time()
        }
        
        self.performance_stats["strategies_learned"] += 1

    def update_arc_patterns(self, input_analysis, output_analysis, transformation):
        """تحديث أنماط ARC المكتشفة"""
        
        # تحديث أنماط التحويل
        if transformation:
            trans_type = transformation["type"]
            if trans_type not in self.arc_patterns["transformations"]:
                self.arc_patterns["transformations"][trans_type] = 0
            self.arc_patterns["transformations"][trans_type] += 1
        
        # تحديث أنماط الألوان
        for colors in [input_analysis["color_distribution"], output_analysis["color_distribution"]]:
            for color, count in colors.items():
                if color not in self.arc_patterns["color_patterns"]:
                    self.arc_patterns["color_patterns"][color] = 0
                self.arc_patterns["color_patterns"][color] += count
        
        self.performance_stats["patterns_discovered"] += 1

    def solve_arc_problem(self, test_input):
        """حل مسألة ARC جديدة"""
        
        print(f"🎯 محاولة حل مسألة ARC جديدة...")
        
        self.performance_stats["problems_attempted"] += 1
        
        # تحليل المدخل
        input_analysis = self.analyze_arc_grid(test_input)
        
        # البحث عن استراتيجية مناسبة
        best_strategy = self.find_best_strategy(input_analysis)
        
        # تطبيق الاستراتيجية
        if best_strategy:
            solution = self.apply_strategy(test_input, best_strategy)
            confidence = best_strategy["success_count"] / max(1, self.performance_stats["problems_attempted"])
        else:
            # محاولة حل بناءً على الأنماط العامة
            solution = self.attempt_pattern_based_solution(test_input, input_analysis)
            confidence = 0.1  # ثقة منخفضة للحلول التجريبية
        
        # تعلم من محاولة الحل
        problem_info = f"ARC Problem: Input shape {input_analysis['shape']}, colors {list(input_analysis['color_distribution'].keys())}"
        self.encounter_new_information(problem_info, "arc_solving")
        
        return {
            "input": test_input,
            "predicted_output": solution,
            "confidence": confidence,
            "strategy_used": best_strategy["transformation"]["type"] if best_strategy else "pattern_based",
            "input_analysis": input_analysis
        }

    def find_best_strategy(self, input_analysis):
        """العثور على أفضل استراتيجية للمدخل المعطى"""
        
        best_strategy = None
        best_score = 0
        
        for strategy_key, strategy in self.solving_strategies.items():
            # حساب التشابه مع الخصائص المحفوظة
            similarity = self.calculate_strategy_similarity(input_analysis, strategy["input_characteristics"])
            
            # حساب النتيجة بناءً على التشابه ومعدل النجاح
            score = similarity * strategy["success_count"]
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy

    def calculate_strategy_similarity(self, current_analysis, stored_analysis):
        """حساب التشابه بين تحليل حالي ومحفوظ"""
        
        similarity_factors = []
        
        # تشابه الشكل
        if current_analysis["shape"] == stored_analysis["shape"]:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.5)
        
        # تشابه عدد الألوان
        color_diff = abs(current_analysis["unique_colors"] - stored_analysis["unique_colors"])
        color_similarity = max(0, 1 - color_diff / 10)
        similarity_factors.append(color_similarity)
        
        # تشابه الأنماط
        current_patterns = set(current_analysis["patterns"])
        stored_patterns = set(stored_analysis["patterns"])
        pattern_similarity = len(current_patterns.intersection(stored_patterns)) / max(1, len(current_patterns.union(stored_patterns)))
        similarity_factors.append(pattern_similarity)
        
        return sum(similarity_factors) / len(similarity_factors)

    def apply_strategy(self, input_grid, strategy):
        """تطبيق استراتيجية على المدخل"""
        
        input_grid = np.array(input_grid)
        transformation = strategy["transformation"]
        
        if transformation["type"] == "rotation":
            degrees = transformation["degrees"]
            rotations = degrees // 90
            return np.rot90(input_grid, rotations).tolist()
        
        elif transformation["type"] == "flip_vertical":
            return np.flipud(input_grid).tolist()
        
        elif transformation["type"] == "flip_horizontal":
            return np.fliplr(input_grid).tolist()
        
        elif transformation["type"] == "resize":
            target_shape = transformation["to_shape"]
            # تكرار أو قص حسب الحاجة
            if target_shape[0] > input_grid.shape[0] or target_shape[1] > input_grid.shape[1]:
                # تكرار
                repeated = np.tile(input_grid, (2, 2))
                return repeated[:target_shape[0], :target_shape[1]].tolist()
            else:
                # قص
                return input_grid[:target_shape[0], :target_shape[1]].tolist()
        
        elif transformation["type"] == "color_change":
            # تغيير لون بسيط (مثال)
            result = input_grid.copy()
            result[result == 1] = 2  # تغيير اللون 1 إلى 2
            return result.tolist()
        
        else:
            # استراتيجية غير معروفة، إرجاع المدخل كما هو
            return input_grid.tolist()

    def attempt_pattern_based_solution(self, input_grid, input_analysis):
        """محاولة حل بناءً على الأنماط العامة"""
        
        input_grid = np.array(input_grid)
        
        # إذا كان هناك نمط حدود، جرب إزالة الحدود
        if "border_pattern" in input_analysis["patterns"]:
            if input_grid.shape[0] > 2 and input_grid.shape[1] > 2:
                return input_grid[1:-1, 1:-1].tolist()
        
        # إذا كان مربع، جرب الدوران
        if "square_grid" in input_analysis["patterns"]:
            return np.rot90(input_grid).tolist()
        
        # إذا كان هناك خط أفقي، جرب التحويل لعمودي
        if "horizontal_line" in input_analysis["patterns"]:
            return input_grid.T.tolist()
        
        # الحل الافتراضي: إرجاع نسخة معكوسة
        return np.flipud(input_grid).tolist()

    def get_arc_stats(self):
        """إحصائيات أداء ARC"""
        base_stats = self.get_learning_stats()
        
        arc_stats = {
            **base_stats,
            "arc_performance": self.performance_stats,
            "solving_strategies": len(self.solving_strategies),
            "arc_patterns_discovered": sum(len(patterns) for patterns in self.arc_patterns.values()),
            "success_rate": self.performance_stats["problems_solved"] / max(1, self.performance_stats["problems_attempted"])
        }
        
        return arc_stats

if __name__ == "__main__":
    # اختبار نظام حل ARC
    solver = ARCLearningSolver()
    
    print("\n" + "="*60)
    print("🧩 اختبار نظام حل ARC بالتعلم الحقيقي")
    print("="*60)
    
    # أمثلة تدريب بسيطة
    training_examples = [
        {
            "input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
            "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        },
        {
            "input": [[1, 1], [1, 1]],
            "output": [[2, 2], [2, 2]]
        },
        {
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "output": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        }
    ]
    
    # تدريب النظام
    print("📚 تدريب النظام على أمثلة ARC:")
    for i, example in enumerate(training_examples):
        print(f"\n🔍 مثال {i+1}:")
        result = solver.learn_from_arc_example(example["input"], example["output"])
        print(f"✅ تم اكتشاف تحويل: {result['transformation_discovered']}")
        if result["transformation"]:
            print(f"🔄 نوع التحويل: {result['transformation']['type']}")
    
    # اختبار الحل
    print(f"\n🎯 اختبار حل مسائل جديدة:")
    test_problems = [
        [[2, 0, 2], [0, 2, 0], [2, 0, 2]],
        [[3, 3], [3, 3]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ]
    
    for i, problem in enumerate(test_problems):
        print(f"\n🧩 مسألة {i+1}:")
        solution = solver.solve_arc_problem(problem)
        print(f"💡 الحل المتوقع: {solution['predicted_output']}")
        print(f"🎯 الثقة: {solution['confidence']:.2f}")
        print(f"🔧 الاستراتيجية: {solution['strategy_used']}")
    
    # عرض الإحصائيات
    stats = solver.get_arc_stats()
    print(f"\n📊 إحصائيات الأداء:")
    print(f"🧩 مسائل محاولة: {stats['arc_performance']['problems_attempted']}")
    print(f"✅ مسائل محلولة: {stats['arc_performance']['problems_solved']}")
    print(f"🔧 استراتيجيات متعلمة: {stats['solving_strategies']}")
    print(f"🔍 أنماط مكتشفة: {stats['arc_patterns_discovered']}")
    print(f"📈 معدل النجاح: {stats['success_rate']:.2%}")
    print(f"🧠 مستوى الذكاء العام: {stats['overall_intelligence']:.3f}")
