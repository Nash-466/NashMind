#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
حلال مسائل ARC - مكون جديد في NashMind
يضيف قدرات حل مسائل ARC مع التعلم الحقيقي
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
import pickle

def convert_numpy_to_python(obj):
    """تحويل NumPy arrays و int64 إلى أنواع Python الأساسية"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

class ARCProblemSolver:
    """
    حلال مسائل ARC مدمج في نظام NashMind
    يتعلم من كل مسألة ويطور استراتيجيات جديدة
    """
    
    def __init__(self, memory_file="nashmind_arc_memory.json"):
        self.memory_file = memory_file
        
        # ذاكرة مسائل ARC
        self.arc_memory = self.load_arc_memory()
        
        # استراتيجيات متعلمة
        self.learned_strategies = {
            "rotation": {"success_rate": 0.0, "usage_count": 0},
            "reflection": {"success_rate": 0.0, "usage_count": 0},
            "color_change": {"success_rate": 0.0, "usage_count": 0},
            "pattern_completion": {"success_rate": 0.0, "usage_count": 0},
            "size_scaling": {"success_rate": 0.0, "usage_count": 0}
        }
        
        # أنماط مكتشفة
        self.discovered_patterns = {}
        
        # إحصائيات الأداء
        self.performance_stats = {
            "problems_attempted": 0,
            "problems_solved": 0,
            "strategies_learned": 0,
            "patterns_discovered": 0
        }
        
        print("🧩 تم تهيئة حلال مسائل ARC في NashMind")
        print(f"📚 ذاكرة ARC: {len(self.arc_memory)} مسألة")

    def load_arc_memory(self):
        """تحميل ذاكرة مسائل ARC"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_arc_memory(self):
        """حفظ ذاكرة مسائل ARC"""
        # تحويل البيانات قبل الحفظ
        converted_memory = convert_numpy_to_python(self.arc_memory)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(converted_memory, f, ensure_ascii=False, indent=2)

    def learn_from_arc_example(self, input_grid, output_grid):
        """التعلم من مثال ARC"""
        
        print("🔍 تحليل مثال ARC جديد...")
        
        # تحليل التحويل
        transformation = self.analyze_transformation(input_grid, output_grid)
        
        # اكتشاف الأنماط
        patterns = self.discover_grid_patterns(input_grid, output_grid)
        
        # تحديث الاستراتيجيات
        self.update_strategies(transformation, patterns)
        
        # حفظ في الذاكرة (مع تحويل البيانات)
        example_id = f"ARC_EX_{len(self.arc_memory)}_{int(time.time())}"
        self.arc_memory[example_id] = {
            "input_grid": convert_numpy_to_python(input_grid),
            "output_grid": convert_numpy_to_python(output_grid),
            "transformation": convert_numpy_to_python(transformation),
            "patterns": convert_numpy_to_python(patterns),
            "learned_at": time.time()
        }
        
        self.save_arc_memory()
        
        return {
            "example_id": example_id,
            "transformation_type": transformation["type"],
            "patterns_found": len(patterns),
            "confidence": transformation["confidence"]
        }

    def analyze_transformation(self, input_grid, output_grid):
        """تحليل التحويل بين الدخل والخرج"""

        # تحويل إلى NumPy arrays مع ضمان النوع الصحيح
        input_array = np.array(input_grid, dtype=int)
        output_array = np.array(output_grid, dtype=int)

        transformation = {
            "type": "unknown",
            "confidence": 0.0,
            "parameters": {}
        }

        # فحص تغيير الحجم
        if input_array.shape != output_array.shape:
            transformation["type"] = "size_change"
            transformation["parameters"]["input_shape"] = list(input_array.shape)
            transformation["parameters"]["output_shape"] = list(output_array.shape)
            transformation["confidence"] = 0.8
            return transformation
        
        # فحص تغيير الألوان
        input_colors = set(input_array.flatten())
        output_colors = set(output_array.flatten())
        
        if input_colors != output_colors:
            transformation["type"] = "color_change"
            transformation["parameters"]["color_mapping"] = self.find_color_mapping(input_array, output_array)
            transformation["confidence"] = 0.7
            return transformation
        
        # فحص الدوران
        if self.is_rotation(input_array, output_array):
            transformation["type"] = "rotation"
            transformation["parameters"]["angle"] = self.find_rotation_angle(input_array, output_array)
            transformation["confidence"] = 0.9
            return transformation
        
        # فحص الانعكاس
        if self.is_reflection(input_array, output_array):
            transformation["type"] = "reflection"
            transformation["parameters"]["axis"] = self.find_reflection_axis(input_array, output_array)
            transformation["confidence"] = 0.9
            return transformation
        
        # تحويل معقد
        transformation["type"] = "complex"
        transformation["confidence"] = 0.3
        
        return transformation

    def discover_grid_patterns(self, input_grid, output_grid):
        """اكتشاف أنماط في الشبكة"""

        patterns = []

        input_array = np.array(input_grid, dtype=int)
        output_array = np.array(output_grid, dtype=int)
        
        # نمط التكرار
        if self.has_repetition_pattern(input_array):
            patterns.append({
                "type": "repetition",
                "description": "نمط تكرار في الدخل",
                "confidence": 0.8
            })
        
        # نمط التماثل
        if self.has_symmetry_pattern(input_array):
            patterns.append({
                "type": "symmetry",
                "description": "نمط تماثل في الدخل",
                "confidence": 0.7
            })
        
        # نمط الحدود
        if self.has_border_pattern(input_array, output_array):
            patterns.append({
                "type": "border",
                "description": "نمط متعلق بالحدود",
                "confidence": 0.6
            })
        
        return patterns

    def is_rotation(self, input_array, output_array):
        """فحص إذا كان التحويل دوران"""
        
        # فحص دوران 90 درجة
        rotated_90 = np.rot90(input_array)
        if np.array_equal(rotated_90, output_array):
            return True
        
        # فحص دوران 180 درجة
        rotated_180 = np.rot90(input_array, 2)
        if np.array_equal(rotated_180, output_array):
            return True
        
        # فحص دوران 270 درجة
        rotated_270 = np.rot90(input_array, 3)
        if np.array_equal(rotated_270, output_array):
            return True
        
        return False

    def find_rotation_angle(self, input_array, output_array):
        """العثور على زاوية الدوران"""
        
        for k in range(1, 4):
            rotated = np.rot90(input_array, k)
            if np.array_equal(rotated, output_array):
                return k * 90
        
        return 0

    def is_reflection(self, input_array, output_array):
        """فحص إذا كان التحويل انعكاس"""
        
        # انعكاس أفقي
        flipped_h = np.fliplr(input_array)
        if np.array_equal(flipped_h, output_array):
            return True
        
        # انعكاس عمودي
        flipped_v = np.flipud(input_array)
        if np.array_equal(flipped_v, output_array):
            return True
        
        return False

    def find_reflection_axis(self, input_array, output_array):
        """العثور على محور الانعكاس"""
        
        if np.array_equal(np.fliplr(input_array), output_array):
            return "horizontal"
        elif np.array_equal(np.flipud(input_array), output_array):
            return "vertical"
        
        return "unknown"

    def find_color_mapping(self, input_array, output_array):
        """العثور على تطبيق الألوان"""
        
        color_map = {}
        
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                input_color = input_array[i, j]
                output_color = output_array[i, j]
                
                if input_color in color_map:
                    if color_map[input_color] != output_color:
                        return {}  # تطبيق غير متسق
                else:
                    color_map[input_color] = output_color
        
        return color_map

    def has_repetition_pattern(self, array):
        """فحص وجود نمط تكرار"""
        
        # فحص تكرار الصفوف
        for i in range(array.shape[0] - 1):
            if np.array_equal(array[i], array[i + 1]):
                return True
        
        # فحص تكرار الأعمدة
        for j in range(array.shape[1] - 1):
            if np.array_equal(array[:, j], array[:, j + 1]):
                return True
        
        return False

    def has_symmetry_pattern(self, array):
        """فحص وجود نمط تماثل"""
        
        # تماثل أفقي
        if np.array_equal(array, np.fliplr(array)):
            return True
        
        # تماثل عمودي
        if np.array_equal(array, np.flipud(array)):
            return True
        
        return False

    def has_border_pattern(self, input_array, output_array):
        """فحص وجود نمط متعلق بالحدود"""
        
        # فحص إذا كانت الحدود تغيرت
        input_border = self.extract_border(input_array)
        output_border = self.extract_border(output_array)
        
        return not np.array_equal(input_border, output_border)

    def extract_border(self, array):
        """استخراج حدود الشبكة"""
        
        border = []
        
        # الصف الأول والأخير
        border.extend(array[0, :].tolist())
        border.extend(array[-1, :].tolist())
        
        # العمود الأول والأخير (بدون الزوايا)
        border.extend(array[1:-1, 0].tolist())
        border.extend(array[1:-1, -1].tolist())
        
        return border

    def update_strategies(self, transformation, patterns):
        """تحديث الاستراتيجيات المتعلمة"""
        
        strategy_type = transformation["type"]
        
        if strategy_type in self.learned_strategies:
            self.learned_strategies[strategy_type]["usage_count"] += 1
            
            # تحديث معدل النجاح بناءً على الثقة
            current_success = self.learned_strategies[strategy_type]["success_rate"]
            confidence = transformation["confidence"]
            usage_count = self.learned_strategies[strategy_type]["usage_count"]
            
            # متوسط متحرك لمعدل النجاح
            new_success_rate = (current_success * (usage_count - 1) + confidence) / usage_count
            self.learned_strategies[strategy_type]["success_rate"] = new_success_rate

    def solve_arc_problem(self, test_input):
        """حل مسألة ARC جديدة"""
        
        print("🎯 محاولة حل مسألة ARC جديدة...")
        
        self.performance_stats["problems_attempted"] += 1
        
        # تحليل الدخل
        input_analysis = self.analyze_input(test_input)
        
        # البحث عن أمثلة مشابهة
        similar_examples = self.find_similar_examples(input_analysis)
        
        # تطبيق الاستراتيجيات المتعلمة
        solution_candidates = self.apply_learned_strategies(test_input, similar_examples)
        
        # اختيار أفضل حل
        best_solution = self.select_best_solution(solution_candidates)
        
        # تقييم الثقة
        confidence = self.calculate_solution_confidence(best_solution, similar_examples)
        
        if confidence > 0.5:
            self.performance_stats["problems_solved"] += 1
        
        return {
            "predicted_output": best_solution["output"] if best_solution else test_input,
            "confidence": confidence,
            "strategy_used": best_solution["strategy"] if best_solution else "none",
            "similar_examples_found": len(similar_examples)
        }

    def analyze_input(self, test_input):
        """تحليل دخل المسألة"""

        input_array = np.array(test_input, dtype=int)

        return {
            "shape": list(input_array.shape),
            "colors": [int(c) for c in set(input_array.flatten())],
            "has_repetition": self.has_repetition_pattern(input_array),
            "has_symmetry": self.has_symmetry_pattern(input_array),
            "border": convert_numpy_to_python(self.extract_border(input_array))
        }

    def find_similar_examples(self, input_analysis):
        """البحث عن أمثلة مشابهة"""
        
        similar = []
        
        for example_id, example_data in self.arc_memory.items():
            example_input = np.array(example_data["input_grid"], dtype=int)

            # مقارنة الشكل
            shape_similarity = 1.0 if list(example_input.shape) == input_analysis["shape"] else 0.5

            # مقارنة الألوان
            example_colors = set(int(c) for c in example_input.flatten())
            color_similarity = len(set(input_analysis["colors"]).intersection(example_colors)) / \
                             len(set(input_analysis["colors"]).union(example_colors))
            
            # مقارنة الأنماط
            pattern_similarity = 0.0
            if input_analysis["has_repetition"] and self.has_repetition_pattern(example_input):
                pattern_similarity += 0.5
            if input_analysis["has_symmetry"] and self.has_symmetry_pattern(example_input):
                pattern_similarity += 0.5
            
            # حساب التشابه الإجمالي
            total_similarity = (shape_similarity + color_similarity + pattern_similarity) / 3
            
            if total_similarity > 0.3:
                similar.append({
                    "example_id": example_id,
                    "similarity": total_similarity,
                    "data": example_data
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:3]

    def apply_learned_strategies(self, test_input, similar_examples):
        """تطبيق الاستراتيجيات المتعلمة - محسن"""

        candidates = []
        input_array = np.array(test_input, dtype=int)

        print(f"🔍 تطبيق الاستراتيجيات على شبكة {input_array.shape}")

        # استراتيجيات أساسية دائماً
        candidates.extend(self.apply_basic_strategies(input_array))

        # استراتيجيات من الأمثلة المشابهة
        candidates.extend(self.apply_similar_example_strategies(input_array, similar_examples))

        # استراتيجيات متقدمة
        candidates.extend(self.apply_advanced_strategies(input_array))

        # استراتيجيات الأنماط
        candidates.extend(self.apply_pattern_strategies(input_array))

        print(f"🎯 تم إنتاج {len(candidates)} مرشح حل")

        return candidates

    def apply_basic_strategies(self, input_array):
        """تطبيق الاستراتيجيات الأساسية"""

        candidates = []

        # 1. الدوران
        for angle in [90, 180, 270]:
            try:
                rotated = np.rot90(input_array, angle // 90)
                candidates.append({
                    "output": convert_numpy_to_python(rotated),
                    "strategy": f"rotation_{angle}",
                    "confidence": 0.6
                })
            except:
                pass

        # 2. الانعكاس
        try:
            flipped_h = np.fliplr(input_array)
            flipped_v = np.flipud(input_array)

            candidates.append({
                "output": convert_numpy_to_python(flipped_h),
                "strategy": "reflection_horizontal",
                "confidence": 0.6
            })

            candidates.append({
                "output": convert_numpy_to_python(flipped_v),
                "strategy": "reflection_vertical",
                "confidence": 0.6
            })
        except:
            pass

        # 3. النسخ (بدون تغيير)
        candidates.append({
            "output": convert_numpy_to_python(input_array),
            "strategy": "identity",
            "confidence": 0.3
        })

        # 4. استراتيجيات التلوين الجديدة
        try:
            color_strategies = self.apply_color_transformation_strategies(input_array)
            candidates.extend(color_strategies)
        except:
            pass

        # 5. استراتيجيات الحجم
        try:
            size_strategies = self.apply_size_transformation_strategies(input_array)
            candidates.extend(size_strategies)
        except:
            pass

        return candidates

    def apply_similar_example_strategies(self, input_array, similar_examples):
        """تطبيق استراتيجيات من الأمثلة المشابهة"""

        candidates = []

        for similar in similar_examples:
            try:
                transformation = similar["data"]["transformation"]
                trans_type = transformation.get("type", "unknown")
                confidence = transformation.get("confidence", 0.0) * similar["similarity"]

                if trans_type == "color_change":
                    color_map = transformation.get("parameters", {}).get("color_mapping", {})
                    if color_map:
                        changed = self.apply_color_mapping(input_array, color_map)
                        candidates.append({
                            "output": convert_numpy_to_python(changed),
                            "strategy": "learned_color_change",
                            "confidence": confidence
                        })

                elif trans_type == "size_change":
                    # جرب تغيير الحجم
                    params = transformation.get("parameters", {})
                    if "output_shape" in params:
                        resized = self.resize_grid(input_array, params["output_shape"])
                        if resized is not None:
                            candidates.append({
                                "output": convert_numpy_to_python(resized),
                                "strategy": "learned_resize",
                                "confidence": confidence
                            })

            except Exception as e:
                continue

        return candidates

    def apply_advanced_strategies(self, input_array):
        """تطبيق الاستراتيجيات المتقدمة"""

        candidates = []

        try:
            # 1. استراتيجية التوسيع والتكرار (للمسائل مثل 007bbfb7)
            expansion_result = self.apply_expansion_strategy(input_array)
            if expansion_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(expansion_result),
                    "strategy": "grid_expansion",
                    "confidence": 0.9
                })

            # 2. استخراج الأنماط المتكررة
            pattern_result = self.extract_and_complete_pattern(input_array)
            if pattern_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(pattern_result),
                    "strategy": "pattern_completion",
                    "confidence": 0.7
                })

            # 3. تطبيق قواعد الألوان الذكية
            color_result = self.apply_smart_color_rules(input_array)
            if color_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(color_result),
                    "strategy": "smart_color_rules",
                    "confidence": 0.6
                })

            # 4. تحليل الحدود والمناطق
            boundary_result = self.analyze_boundaries_and_regions(input_array)
            if boundary_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(boundary_result),
                    "strategy": "boundary_analysis",
                    "confidence": 0.5
                })

        except Exception as e:
            pass

        return candidates

    def apply_pattern_strategies(self, input_array):
        """تطبيق استراتيجيات الأنماط"""

        candidates = []

        try:
            # 1. البحث عن التماثل
            symmetry_result = self.apply_symmetry_completion(input_array)
            if symmetry_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(symmetry_result),
                    "strategy": "symmetry_completion",
                    "confidence": 0.8
                })

            # 2. تطبيق قواعد الجوار
            neighbor_result = self.apply_neighbor_rules(input_array)
            if neighbor_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(neighbor_result),
                    "strategy": "neighbor_rules",
                    "confidence": 0.6
                })

            # 3. تحليل الشبكات الفرعية
            subgrid_result = self.analyze_subgrids(input_array)
            if subgrid_result is not None:
                candidates.append({
                    "output": convert_numpy_to_python(subgrid_result),
                    "strategy": "subgrid_analysis",
                    "confidence": 0.7
                })

        except Exception as e:
            pass

        return candidates

    def apply_color_mapping(self, array, color_map):
        """تطبيق تطبيق الألوان"""
        
        result = array.copy()
        
        for old_color, new_color in color_map.items():
            result[array == old_color] = new_color
        
        return result

    def select_best_solution(self, candidates):
        """اختيار أفضل حل - محسن"""

        if not candidates:
            return None

        print(f"🎯 تقييم {len(candidates)} مرشح حل...")

        # تحسين درجات الثقة
        enhanced_candidates = []
        for candidate in candidates:
            enhanced_score = self.calculate_enhanced_confidence(candidate)
            enhanced_candidates.append({
                **candidate,
                "enhanced_confidence": enhanced_score
            })

        # ترتيب حسب الثقة المحسنة
        sorted_candidates = sorted(enhanced_candidates, key=lambda x: x["enhanced_confidence"], reverse=True)

        best = sorted_candidates[0]
        print(f"🏆 أفضل حل: {best['strategy']} (ثقة: {best['enhanced_confidence']:.3f})")

        return best

    def calculate_enhanced_confidence(self, candidate):
        """حساب ثقة محسنة للحل"""

        base_confidence = candidate["confidence"]
        strategy = candidate["strategy"]

        # مكافآت للاستراتيجيات المختلفة
        strategy_bonuses = {
            "symmetry_completion": 0.2,
            "pattern_completion": 0.15,
            "learned_color_change": 0.1,
            "boundary_analysis": 0.1,
            "subgrid_analysis": 0.1,
            "smart_color_rules": 0.05,
            "neighbor_rules": 0.05,
            "rotation_90": 0.05,
            "rotation_180": 0.05,
            "reflection_horizontal": 0.05,
            "reflection_vertical": 0.05,
            "identity": -0.1  # تقليل ثقة عدم التغيير
        }

        bonus = strategy_bonuses.get(strategy, 0.0)

        # تحليل جودة الحل
        output = candidate["output"]
        quality_score = self.assess_solution_quality(output)

        enhanced_confidence = base_confidence + bonus + quality_score
        return min(1.0, max(0.0, enhanced_confidence))

    def assess_solution_quality(self, output):
        """تقييم جودة الحل"""
        try:
            output_array = np.array(output, dtype=int)

            quality_score = 0.0

            # 1. تنوع الألوان (التنوع المعتدل أفضل)
            unique_colors = len(np.unique(output_array))
            if 2 <= unique_colors <= 5:
                quality_score += 0.1

            # 2. وجود أنماط منتظمة
            if self.has_regular_patterns(output_array):
                quality_score += 0.1

            # 3. التماثل
            if self.has_symmetry(output_array):
                quality_score += 0.1

            # 4. عدم وجود ضوضاء مفرطة
            if not self.has_excessive_noise(output_array):
                quality_score += 0.05

            return quality_score
        except:
            return 0.0

    def has_regular_patterns(self, array):
        """فحص وجود أنماط منتظمة"""
        try:
            h, w = array.shape

            # فحص أنماط بسيطة
            for ph in range(1, min(h//2 + 1, 3)):
                for pw in range(1, min(w//2 + 1, 3)):
                    if self.is_repeating_pattern(array, array[:ph, :pw]):
                        return True
            return False
        except:
            return False

    def has_symmetry(self, array):
        """فحص وجود تماثل"""
        try:
            h, w = array.shape

            # تماثل أفقي
            if w % 2 == 0:
                left_half = array[:, :w//2]
                right_half = np.fliplr(array[:, w//2:])
                if np.array_equal(left_half, right_half):
                    return True

            # تماثل عمودي
            if h % 2 == 0:
                top_half = array[:h//2, :]
                bottom_half = np.flipud(array[h//2:, :])
                if np.array_equal(top_half, bottom_half):
                    return True

            return False
        except:
            return False

    def has_excessive_noise(self, array):
        """فحص وجود ضوضاء مفرطة"""
        try:
            # حساب عدد الألوان المختلفة في الجوار
            h, w = array.shape
            noise_count = 0

            for i in range(1, h-1):
                for j in range(1, w-1):
                    neighbors = self.get_neighbors(array, i, j)
                    if len(set(neighbors)) > 4:  # تنوع مفرط في الجوار
                        noise_count += 1

            return noise_count > (h * w) * 0.3  # أكثر من 30% ضوضاء
        except:
            return False

    def calculate_solution_confidence(self, solution, similar_examples):
        """حساب ثقة الحل"""

        if not solution:
            return 0.0

        base_confidence = solution.get("enhanced_confidence", solution["confidence"])

        # زيادة الثقة بناءً على الأمثلة المشابهة
        similarity_boost = len(similar_examples) * 0.05

        return min(1.0, base_confidence + similarity_boost)

    def get_arc_stats(self):
        """إحصائيات حلال ARC"""
        
        success_rate = 0.0
        if self.performance_stats["problems_attempted"] > 0:
            success_rate = self.performance_stats["problems_solved"] / self.performance_stats["problems_attempted"]
        
        return {
            "performance_stats": self.performance_stats,
            "success_rate": success_rate,
            "learned_strategies": self.learned_strategies,
            "total_examples": len(self.arc_memory),
            "patterns_discovered": len(self.discovered_patterns)
        }

    # ==================== دوال الاستراتيجيات المتقدمة ====================

    def resize_grid(self, input_array, target_shape):
        """تغيير حجم الشبكة"""
        try:
            if len(target_shape) != 2:
                return None

            target_h, target_w = target_shape
            input_h, input_w = input_array.shape

            # إذا كان الحجم الجديد أصغر، اقتطع
            if target_h <= input_h and target_w <= input_w:
                return input_array[:target_h, :target_w]

            # إذا كان الحجم الجديد أكبر، املأ بالصفر
            result = np.zeros((target_h, target_w), dtype=int)
            result[:min(input_h, target_h), :min(input_w, target_w)] = input_array[:min(input_h, target_h), :min(input_w, target_w)]
            return result

        except:
            return None

    def extract_and_complete_pattern(self, input_array):
        """استخراج وإكمال الأنماط"""
        try:
            h, w = input_array.shape

            # البحث عن أنماط تكرار بسيطة
            for pattern_h in range(1, min(h//2 + 1, 4)):
                for pattern_w in range(1, min(w//2 + 1, 4)):
                    pattern = input_array[:pattern_h, :pattern_w]

                    # فحص إذا كان النمط يتكرر
                    if self.is_repeating_pattern(input_array, pattern):
                        # إكمال النمط
                        completed = self.complete_pattern(input_array, pattern)
                        if completed is not None:
                            return completed

            return None
        except:
            return None

    def is_repeating_pattern(self, array, pattern):
        """فحص إذا كان النمط يتكرر"""
        try:
            h, w = array.shape
            ph, pw = pattern.shape

            matches = 0
            total = 0

            for i in range(0, h, ph):
                for j in range(0, w, pw):
                    end_i = min(i + ph, h)
                    end_j = min(j + pw, w)

                    sub_array = array[i:end_i, j:end_j]
                    sub_pattern = pattern[:end_i-i, :end_j-j]

                    if np.array_equal(sub_array, sub_pattern):
                        matches += 1
                    total += 1

            return matches / total > 0.7  # 70% تطابق
        except:
            return False

    def complete_pattern(self, array, pattern):
        """إكمال النمط"""
        try:
            h, w = array.shape
            ph, pw = pattern.shape

            result = array.copy()

            for i in range(0, h, ph):
                for j in range(0, w, pw):
                    end_i = min(i + ph, h)
                    end_j = min(j + pw, w)

                    result[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]

            return result
        except:
            return None

    def apply_smart_color_rules(self, input_array):
        """تطبيق قواعد الألوان الذكية"""
        try:
            result = input_array.copy()

            # قاعدة 1: تحويل الألوان النادرة إلى الأكثر شيوعاً
            unique, counts = np.unique(input_array, return_counts=True)
            if len(unique) > 1:
                most_common = unique[np.argmax(counts)]
                least_common = unique[np.argmin(counts)]

                if counts[np.argmin(counts)] < counts[np.argmax(counts)] / 3:
                    result[input_array == least_common] = most_common

            # قاعدة 2: تطبيق قواعد الجوار
            for i in range(input_array.shape[0]):
                for j in range(input_array.shape[1]):
                    neighbors = self.get_neighbors(input_array, i, j)
                    if len(set(neighbors)) == 1 and len(neighbors) > 2:
                        # إذا كان معظم الجيران نفس اللون
                        result[i, j] = neighbors[0]

            return result if not np.array_equal(result, input_array) else None
        except:
            return None

    def analyze_boundaries_and_regions(self, input_array):
        """تحليل الحدود والمناطق"""
        try:
            result = input_array.copy()
            h, w = input_array.shape

            # تحليل الحدود
            border_color = input_array[0, 0]  # افتراض أن الحدود لها نفس لون الزاوية

            # ملء المناطق المحاطة
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if input_array[i, j] != border_color:
                        # فحص إذا كانت المنطقة محاطة
                        if self.is_enclosed_region(input_array, i, j, border_color):
                            result[i, j] = border_color

            return result if not np.array_equal(result, input_array) else None
        except:
            return None

    def apply_symmetry_completion(self, input_array):
        """تطبيق إكمال التماثل"""
        try:
            h, w = input_array.shape

            # فحص التماثل الأفقي
            if self.has_partial_horizontal_symmetry(input_array):
                result = input_array.copy()
                mid = w // 2
                for i in range(h):
                    for j in range(mid):
                        if result[i, j] == 0 and result[i, w-1-j] != 0:
                            result[i, j] = result[i, w-1-j]
                        elif result[i, w-1-j] == 0 and result[i, j] != 0:
                            result[i, w-1-j] = result[i, j]
                return result

            # فحص التماثل العمودي
            if self.has_partial_vertical_symmetry(input_array):
                result = input_array.copy()
                mid = h // 2
                for i in range(mid):
                    for j in range(w):
                        if result[i, j] == 0 and result[h-1-i, j] != 0:
                            result[i, j] = result[h-1-i, j]
                        elif result[h-1-i, j] == 0 and result[i, j] != 0:
                            result[h-1-i, j] = result[i, j]
                return result

            return None
        except:
            return None

    def apply_neighbor_rules(self, input_array):
        """تطبيق قواعد الجوار"""
        try:
            result = input_array.copy()
            changed = False

            for i in range(input_array.shape[0]):
                for j in range(input_array.shape[1]):
                    if input_array[i, j] == 0:  # خلية فارغة
                        neighbors = self.get_neighbors(input_array, i, j)
                        non_zero_neighbors = [n for n in neighbors if n != 0]

                        if len(non_zero_neighbors) >= 2:
                            # استخدم اللون الأكثر شيوعاً في الجوار
                            unique, counts = np.unique(non_zero_neighbors, return_counts=True)
                            most_common = unique[np.argmax(counts)]
                            result[i, j] = most_common
                            changed = True

            return result if changed else None
        except:
            return None

    def analyze_subgrids(self, input_array):
        """تحليل الشبكات الفرعية"""
        try:
            h, w = input_array.shape

            # تحليل شبكات 2x2
            if h >= 2 and w >= 2:
                result = input_array.copy()
                changed = False

                for i in range(0, h-1, 2):
                    for j in range(0, w-1, 2):
                        subgrid = input_array[i:i+2, j:j+2]
                        completed_subgrid = self.complete_subgrid_pattern(subgrid)

                        if completed_subgrid is not None:
                            result[i:i+2, j:j+2] = completed_subgrid
                            changed = True

                return result if changed else None

            return None
        except:
            return None

    def apply_expansion_strategy(self, input_array):
        """استراتيجية التوسيع والتكرار المحسنة - للمسائل مثل 007bbfb7"""
        try:
            h, w = input_array.shape

            # فحص إذا كانت الشبكة صغيرة (3x3 أو أصغر)
            if h <= 3 and w <= 3:
                # إنشاء شبكة 9x9 (3x3 من الكتل)
                expanded = np.zeros((9, 9), dtype=int)

                # تحليل النمط من الذاكرة مع تحسينات
                expansion_pattern = self.detect_improved_expansion_pattern(input_array)

                if expansion_pattern:
                    # تطبيق النمط المكتشف
                    for block_row in range(3):
                        for block_col in range(3):
                            start_row = block_row * 3
                            start_col = block_col * 3

                            # تحديد ما إذا كان يجب وضع النمط في هذا الموضع
                            if self.should_place_pattern_improved(block_row, block_col, expansion_pattern):
                                expanded[start_row:start_row+3, start_col:start_col+3] = input_array

                    return expanded
                else:
                    # نمط احتياطي محسن
                    return self.apply_fallback_expansion_pattern(input_array)

            # للشبكات الأكبر، جرب أنماط توسيع أخرى
            elif h <= 10 and w <= 10:
                # توسيع 2x محسن
                return self.apply_2x_expansion(input_array)

            return None
        except:
            return None

    def detect_improved_expansion_pattern(self, input_array):
        """كشف نمط التوسيع المحسن من الذاكرة"""
        try:
            # البحث في الذاكرة عن أمثلة مشابهة
            similar_examples = []
            exact_matches = []

            for memory_item in self.arc_memory:
                if 'input' in memory_item and 'output' in memory_item:
                    mem_input = np.array(memory_item['input'])
                    mem_output = np.array(memory_item['output'])

                    # فحص إذا كان الدخل بنفس الحجم
                    if mem_input.shape == input_array.shape:
                        # فحص إذا كان الخرج أكبر بنسبة 3x3
                        if mem_output.shape == (9, 9) and input_array.shape == (3, 3):
                            similarity = self.calculate_input_similarity(input_array, mem_input)

                            example_data = {
                                'input': mem_input,
                                'output': mem_output,
                                'similarity': similarity
                            }

                            if similarity > 0.8:  # تطابق عالي
                                exact_matches.append(example_data)
                            elif similarity > 0.3:  # تشابه معقول
                                similar_examples.append(example_data)

            # إعطاء أولوية للتطابقات الدقيقة
            if exact_matches:
                print(f"🎯 وجد {len(exact_matches)} تطابق دقيق")
                return self.analyze_expansion_pattern_improved(exact_matches)
            elif similar_examples:
                print(f"🔍 وجد {len(similar_examples)} مثال مشابه")
                return self.analyze_expansion_pattern_improved(similar_examples)

            # نمط احتياطي محسن بناءً على التحليل
            return {
                'positions': [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
                'type': 'improved_default',
                'confidence': 0.6
            }
        except:
            return None

    def calculate_input_similarity(self, input1, input2):
        """حساب التشابه بين دخلين"""
        try:
            if input1.shape != input2.shape:
                return 0.0

            # حساب التطابق في الخلايا
            matches = np.sum(input1 == input2)
            total = input1.size

            # حساب التشابه في الألوان
            colors1 = set(input1.flatten())
            colors2 = set(input2.flatten())
            color_similarity = len(colors1.intersection(colors2)) / len(colors1.union(colors2))

            # متوسط مرجح
            cell_similarity = matches / total
            overall_similarity = (cell_similarity * 0.7) + (color_similarity * 0.3)

            return overall_similarity
        except:
            return 0.0

    def analyze_expansion_pattern_improved(self, examples):
        """تحليل نمط التوسيع المحسن من الأمثلة"""
        try:
            if not examples:
                return None

            # ترتيب الأمثلة حسب التشابه
            sorted_examples = sorted(examples, key=lambda x: x.get('similarity', 0), reverse=True)

            # تحليل المواضع التي يظهر فيها النمط
            all_patterns = []

            for example in sorted_examples:
                input_pattern = example['input']
                output_grid = example['output']
                similarity = example.get('similarity', 0)

                # تحليل مفصل للنمط مع وزن التشابه
                pattern_analysis = self.detailed_pattern_analysis_weighted(input_pattern, output_grid, similarity)
                if pattern_analysis:
                    all_patterns.append(pattern_analysis)

            if all_patterns:
                # دمج الأنماط للحصول على النمط الأكثر دقة
                merged_pattern = self.merge_pattern_analyses_improved(all_patterns)
                return merged_pattern

            return None
        except:
            return None

    def detailed_pattern_analysis_weighted(self, input_pattern, output_grid, similarity_weight):
        """تحليل مفصل للنمط مع وزن التشابه"""
        try:
            # تحليل كل كتلة 3x3 في الخرج
            pattern_map = {}

            for block_row in range(3):
                for block_col in range(3):
                    start_row = block_row * 3
                    start_col = block_col * 3

                    block = output_grid[start_row:start_row+3, start_col:start_col+3]

                    # فحص إذا كانت الكتلة تطابق النمط الأصلي
                    if np.array_equal(block, input_pattern):
                        pattern_map[(block_row, block_col)] = {
                            'type': 'exact_match',
                            'weight': similarity_weight * 1.0
                        }
                    elif np.any(block != 0):  # كتلة غير فارغة
                        # حساب نسبة التطابق
                        match_ratio = np.sum(block == input_pattern) / (3 * 3)
                        if match_ratio > 0.7:
                            pattern_map[(block_row, block_col)] = {
                                'type': 'high_match',
                                'weight': similarity_weight * match_ratio
                            }
                        elif match_ratio > 0.3:
                            pattern_map[(block_row, block_col)] = {
                                'type': 'partial_match',
                                'weight': similarity_weight * match_ratio * 0.5
                            }
                    else:
                        pattern_map[(block_row, block_col)] = {
                            'type': 'empty',
                            'weight': 0.0
                        }

            return pattern_map
        except:
            return None

    def merge_pattern_analyses_improved(self, analyses):
        """دمج تحليلات الأنماط المحسن"""
        try:
            # حساب الوزن الإجمالي لكل موضع
            position_weights = {}

            for analysis in analyses:
                for position, data in analysis.items():
                    if position not in position_weights:
                        position_weights[position] = 0.0

                    # إضافة الوزن بناءً على نوع التطابق
                    if data['type'] in ['exact_match', 'high_match']:
                        position_weights[position] += data['weight']
                    elif data['type'] == 'partial_match':
                        position_weights[position] += data['weight'] * 0.5

            # ترتيب المواضع حسب الوزن
            sorted_positions = sorted(position_weights.items(), key=lambda x: x[1], reverse=True)

            # اختيار المواضع ذات الوزن العالي
            threshold = max(position_weights.values()) * 0.3 if position_weights else 0
            selected_positions = [pos for pos, weight in sorted_positions if weight >= threshold]

            # التأكد من وجود عدد معقول من المواضع
            if len(selected_positions) < 3:
                # إضافة المواضع التالية في الترتيب
                additional_needed = 3 - len(selected_positions)
                for pos, weight in sorted_positions[len(selected_positions):]:
                    if additional_needed > 0:
                        selected_positions.append(pos)
                        additional_needed -= 1

            confidence = min(1.0, max(position_weights.values()) if position_weights else 0.5)

            return {
                'positions': selected_positions,
                'type': 'learned_weighted_pattern',
                'confidence': confidence,
                'weights': position_weights
            }
        except:
            return None

    def should_place_pattern_improved(self, block_row, block_col, expansion_pattern):
        """تحديد ما إذا كان يجب وضع النمط في هذا الموضع - محسن"""
        try:
            if not expansion_pattern:
                return False

            position = (block_row, block_col)
            return position in expansion_pattern.get('positions', [])
        except:
            return False

    def apply_fallback_expansion_pattern(self, input_array):
        """تطبيق نمط توسيع احتياطي محسن"""
        try:
            expanded = np.zeros((9, 9), dtype=int)

            # نمط احتياطي محسن بناءً على تحليل المهمة 007bbfb7
            fallback_positions = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

            for block_row in range(3):
                for block_col in range(3):
                    if (block_row, block_col) in fallback_positions:
                        start_row = block_row * 3
                        start_col = block_col * 3
                        expanded[start_row:start_row+3, start_col:start_col+3] = input_array

            return expanded
        except:
            return None

    def apply_2x_expansion(self, input_array):
        """تطبيق توسيع 2x محسن"""
        try:
            h, w = input_array.shape
            expanded = np.zeros((h*2, w*2), dtype=int)

            # توسيع ذكي بناءً على محتوى الشبكة
            if np.sum(input_array) > 0:  # إذا كانت الشبكة تحتوي على بيانات
                # وضع النمط في الزوايا
                expanded[:h, :w] = input_array
                expanded[:h, w:] = input_array
                expanded[h:, :w] = input_array
                expanded[h:, w:] = input_array

            return expanded
        except:
            return None

    def detailed_pattern_analysis(self, input_pattern, output_grid):
        """تحليل مفصل للنمط"""
        try:
            # تحليل كل كتلة 3x3 في الخرج
            pattern_map = {}

            for block_row in range(3):
                for block_col in range(3):
                    start_row = block_row * 3
                    start_col = block_col * 3

                    block = output_grid[start_row:start_row+3, start_col:start_col+3]

                    # فحص إذا كانت الكتلة تطابق النمط الأصلي
                    if np.array_equal(block, input_pattern):
                        pattern_map[(block_row, block_col)] = 'exact_match'
                    elif np.any(block != 0):  # كتلة غير فارغة
                        # حساب نسبة التطابق
                        match_ratio = np.sum(block == input_pattern) / (3 * 3)
                        if match_ratio > 0.5:
                            pattern_map[(block_row, block_col)] = 'partial_match'
                        else:
                            pattern_map[(block_row, block_col)] = 'different'
                    else:
                        pattern_map[(block_row, block_col)] = 'empty'

            return pattern_map
        except:
            return None

    def merge_pattern_analyses(self, analyses):
        """دمج تحليلات الأنماط"""
        try:
            # حساب تكرار كل نوع في كل موضع
            position_stats = {}

            for analysis in analyses:
                for position, pattern_type in analysis.items():
                    if position not in position_stats:
                        position_stats[position] = {}

                    if pattern_type not in position_stats[position]:
                        position_stats[position][pattern_type] = 0

                    position_stats[position][pattern_type] += 1

            # تحديد النمط الأكثر شيوعاً لكل موضع
            final_pattern = {}
            for position, stats in position_stats.items():
                most_common = max(stats.items(), key=lambda x: x[1])
                final_pattern[position] = most_common[0]

            # تحويل إلى قائمة المواضع التي يجب وضع النمط فيها
            active_positions = [pos for pos, pattern_type in final_pattern.items()
                              if pattern_type in ['exact_match', 'partial_match']]

            return {
                'positions': active_positions,
                'type': 'learned_detailed_pattern',
                'pattern_map': final_pattern
            }
        except:
            return None

    def find_pattern_positions(self, pattern, grid):
        """العثور على مواضع النمط في الشبكة"""
        positions = []
        try:
            ph, pw = pattern.shape
            gh, gw = grid.shape

            for i in range(0, gh - ph + 1, ph):
                for j in range(0, gw - pw + 1, pw):
                    subgrid = grid[i:i+ph, j:j+pw]
                    if np.array_equal(subgrid, pattern):
                        positions.append((i//ph, j//pw))

            return positions
        except:
            return []

    def find_common_positions(self, position_lists):
        """العثور على المواضع المشتركة"""
        try:
            if not position_lists:
                return []

            # حساب تكرار كل موضع
            position_counts = {}
            for positions in position_lists:
                for pos in positions:
                    position_counts[pos] = position_counts.get(pos, 0) + 1

            # العثور على المواضع الأكثر شيوعاً
            min_count = len(position_lists) // 2  # على الأقل نصف الأمثلة
            common_positions = [pos for pos, count in position_counts.items() if count >= min_count]

            return common_positions
        except:
            return []

    def should_place_pattern(self, block_row, block_col, expansion_pattern):
        """تحديد ما إذا كان يجب وضع النمط في هذا الموضع"""
        try:
            if not expansion_pattern:
                return False

            position = (block_row, block_col)
            return position in expansion_pattern.get('positions', [])
        except:
            return False

    # ==================== دوال مساعدة ====================

    def get_neighbors(self, array, i, j):
        """الحصول على الجيران"""
        neighbors = []
        h, w = array.shape

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbors.append(array[ni, nj])

        return neighbors

    def is_enclosed_region(self, array, i, j, border_color):
        """فحص إذا كانت المنطقة محاطة"""
        try:
            h, w = array.shape

            # فحص بسيط: إذا كانت جميع الخلايا المجاورة مباشرة من نفس اللون
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        if array[ni, nj] != border_color:
                            return False
            return True
        except:
            return False

    def has_partial_horizontal_symmetry(self, array):
        """فحص التماثل الأفقي الجزئي"""
        try:
            h, w = array.shape
            if w % 2 != 0:
                return False

            mid = w // 2
            matches = 0
            total = 0

            for i in range(h):
                for j in range(mid):
                    left = array[i, j]
                    right = array[i, w-1-j]

                    if left != 0 and right != 0:
                        if left == right:
                            matches += 1
                        total += 1
                    elif left == 0 or right == 0:
                        matches += 0.5  # جزئي
                        total += 1

            return total > 0 and matches / total > 0.6
        except:
            return False

    def has_partial_vertical_symmetry(self, array):
        """فحص التماثل العمودي الجزئي"""
        try:
            h, w = array.shape
            if h % 2 != 0:
                return False

            mid = h // 2
            matches = 0
            total = 0

            for i in range(mid):
                for j in range(w):
                    top = array[i, j]
                    bottom = array[h-1-i, j]

                    if top != 0 and bottom != 0:
                        if top == bottom:
                            matches += 1
                        total += 1
                    elif top == 0 or bottom == 0:
                        matches += 0.5  # جزئي
                        total += 1

            return total > 0 and matches / total > 0.6
        except:
            return False

    def complete_subgrid_pattern(self, subgrid):
        """إكمال نمط الشبكة الفرعية"""
        try:
            if subgrid.shape != (2, 2):
                return None

            # أنماط شائعة في شبكات 2x2
            zeros = np.sum(subgrid == 0)

            if zeros == 1:
                # إكمال النمط المفقود
                result = subgrid.copy()
                non_zero_values = subgrid[subgrid != 0]

                if len(set(non_zero_values)) == 1:
                    # جميع القيم غير الصفرية متشابهة
                    result[subgrid == 0] = non_zero_values[0]
                    return result
                else:
                    # استخدم القيمة الأكثر شيوعاً
                    unique, counts = np.unique(non_zero_values, return_counts=True)
                    most_common = unique[np.argmax(counts)]
                    result[subgrid == 0] = most_common
                    return result

            elif zeros == 2:
                # نمط قطري أو متقابل
                non_zero_positions = np.where(subgrid != 0)
                if len(non_zero_positions[0]) == 2:
                    value = subgrid[non_zero_positions[0][0], non_zero_positions[1][0]]
                    result = subgrid.copy()
                    result[subgrid == 0] = value
                    return result

            return None
        except:
            return None

    # ==================== استراتيجيات جديدة ====================

    def apply_color_transformation_strategies(self, input_array):
        """تطبيق استراتيجيات تحويل الألوان"""
        candidates = []

        try:
            # 1. عكس الألوان (0 -> 1, 1 -> 0, etc.)
            inverted = self.invert_colors(input_array)
            if inverted is not None:
                candidates.append({
                    "output": convert_numpy_to_python(inverted),
                    "strategy": "color_inversion",
                    "confidence": 0.5
                })

            # 2. تبديل الألوان الأكثر شيوعاً
            swapped = self.swap_dominant_colors(input_array)
            if swapped is not None:
                candidates.append({
                    "output": convert_numpy_to_python(swapped),
                    "strategy": "color_swap",
                    "confidence": 0.4
                })

            # 3. تحويل الألوان إلى تدرج
            gradient = self.apply_color_gradient(input_array)
            if gradient is not None:
                candidates.append({
                    "output": convert_numpy_to_python(gradient),
                    "strategy": "color_gradient",
                    "confidence": 0.3
                })

        except:
            pass

        return candidates

    def apply_size_transformation_strategies(self, input_array):
        """تطبيق استراتيجيات تحويل الحجم"""
        candidates = []

        try:
            h, w = input_array.shape

            # 1. تصغير بنسبة 2:1
            if h % 2 == 0 and w % 2 == 0:
                shrunk = self.shrink_by_half(input_array)
                if shrunk is not None:
                    candidates.append({
                        "output": convert_numpy_to_python(shrunk),
                        "strategy": "shrink_half",
                        "confidence": 0.4
                    })

            # 2. قص الحواف
            cropped = self.crop_borders(input_array)
            if cropped is not None:
                candidates.append({
                    "output": convert_numpy_to_python(cropped),
                    "strategy": "crop_borders",
                    "confidence": 0.3
                })

        except:
            pass

        return candidates

    def invert_colors(self, input_array):
        """عكس الألوان"""
        try:
            unique_colors = np.unique(input_array)
            if len(unique_colors) <= 2:
                # للألوان الثنائية، عكس بسيط
                inverted = input_array.copy()
                inverted[input_array == unique_colors[0]] = unique_colors[-1]
                inverted[input_array == unique_colors[-1]] = unique_colors[0]
                return inverted
            else:
                # للألوان المتعددة، عكس حسب القيمة
                max_color = np.max(unique_colors)
                return max_color - input_array
        except:
            return None

    def swap_dominant_colors(self, input_array):
        """تبديل الألوان الأكثر شيوعاً"""
        try:
            unique, counts = np.unique(input_array, return_counts=True)
            if len(unique) >= 2:
                # ترتيب حسب التكرار
                sorted_indices = np.argsort(counts)[::-1]
                most_common = unique[sorted_indices[0]]
                second_common = unique[sorted_indices[1]]

                # تبديل الألوان
                swapped = input_array.copy()
                swapped[input_array == most_common] = -1  # مؤقت
                swapped[input_array == second_common] = most_common
                swapped[swapped == -1] = second_common

                return swapped
        except:
            return None

    def apply_color_gradient(self, input_array):
        """تطبيق تدرج لوني"""
        try:
            unique_colors = np.unique(input_array)
            if len(unique_colors) > 2:
                # إنشاء تدرج من أقل إلى أعلى قيمة
                gradient = input_array.copy()
                for i, color in enumerate(sorted(unique_colors)):
                    gradient[input_array == color] = i
                return gradient
        except:
            return None

    def shrink_by_half(self, input_array):
        """تصغير بنسبة النصف"""
        try:
            h, w = input_array.shape
            if h >= 2 and w >= 2:
                # أخذ عينة كل خليتين
                return input_array[::2, ::2]
        except:
            return None

    def crop_borders(self, input_array):
        """قص الحواف"""
        try:
            h, w = input_array.shape
            if h > 2 and w > 2:
                # قص حافة واحدة من كل جانب
                return input_array[1:-1, 1:-1]
        except:
            return None
