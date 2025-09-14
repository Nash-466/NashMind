from __future__ import annotations
#!/usr/bin/env python3
"""
Enhanced ARC Solver with Pattern Recognition
نظام محسّن لحل مهام ARC مع التعرف على الأنماط
"""

import numpy as np
from collections.abc import Callable
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import defaultdict, Counter
from itertools import product
import hashlib

class EnhancedARCSolver:
    """نظام محسّن لحل مهام ARC"""
    
    def __init__(self):
        self.pattern_memory = defaultdict(list)
        self.learned_transforms = {}
        self.color_mappings = {}
        
    def solve(self, task_data: Dict) -> np.ndarray:
        """حل المهمة باستخدام استراتيجيات متعددة"""
        if 'train' not in task_data or not task_data['train']:
            return np.zeros((3, 3))
        
        # تحليل المهمة
        analysis = self.analyze_task(task_data)
        
        # جرب استراتيجيات مختلفة
        strategies = [
            self.solve_by_pattern_matching,
            self.solve_by_transformation,
            self.solve_by_color_mapping,
            self.solve_by_grid_operations,
            self.solve_by_symmetry,
            self.solve_by_repetition,
            self.solve_by_object_detection,
            self.solve_by_rule_learning
        ]
        
        best_solution = None
        best_score = 0
        
        for strategy in strategies:
            try:
                solution = strategy(task_data, analysis)
                if solution is not None:
                    score = self.evaluate_solution(solution, task_data)
                    if score > best_score:
                        best_score = score
                        best_solution = solution
                        if score >= 1.0:  # حل مثالي
                            return solution
            except Exception:
                continue
        
        # إذا لم نجد حل جيد، استخدم الذكاء الاصطناعي البسيط
        if best_solution is None:
            best_solution = self.fallback_solution(task_data)
        
        return best_solution
    
    def analyze_task(self, task_data: Dict) -> Dict:
        """تحليل شامل للمهمة"""
        analysis = {
            'input_shapes': [],
            'output_shapes': [],
            'color_changes': [],
            'size_changes': [],
            'patterns': [],
            'transformations': [],
            'objects': [],
            'symmetries': []
        }
        
        for example in task_data['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # أشكال وأحجام
            analysis['input_shapes'].append(input_grid.shape)
            analysis['output_shapes'].append(output_grid.shape)
            
            # تغييرات الحجم
            size_change = (output_grid.shape[0] - input_grid.shape[0],
                          output_grid.shape[1] - input_grid.shape[1])
            analysis['size_changes'].append(size_change)
            
            # تحليل الألوان
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            analysis['color_changes'].append({
                'added': output_colors - input_colors,
                'removed': input_colors - output_colors,
                'common': input_colors & output_colors
            })
            
            # البحث عن الأنماط
            patterns = self.find_patterns(input_grid, output_grid)
            analysis['patterns'].extend(patterns)
            
            # الكشف عن الكائنات
            objects = self.detect_objects(input_grid)
            analysis['objects'].append(objects)
            
            # فحص التناظر
            symmetry = self.check_symmetry(output_grid)
            analysis['symmetries'].append(symmetry)
        
        return analysis
    
    def solve_by_pattern_matching(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بمطابقة الأنماط"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # ابحث عن نمط مشترك في أمثلة التدريب
        common_pattern = self.find_common_pattern(task_data['train'])
        
        if common_pattern:
            return self.apply_pattern(test_input, common_pattern)
        
        return None
    
    def solve_by_transformation(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بالتحولات الهندسية"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # اكتشف التحويل من أمثلة التدريب
        transform = self.learn_transformation(task_data['train'])
        
        if transform:
            return self.apply_transformation(test_input, transform)
        
        return None
    
    def solve_by_color_mapping(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بتحويل الألوان"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # تعلم تحويل الألوان
        color_map = self.learn_color_mapping(task_data['train'])
        
        if color_map:
            result = test_input.copy()
            for old_color, new_color in color_map.items():
                result[result == old_color] = new_color
            return result
        
        return None
    
    def solve_by_grid_operations(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بعمليات الشبكة"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # جرب عمليات مختلفة
        operations = [
            lambda x: np.rot90(x, k=1),
            lambda x: np.rot90(x, k=2),
            lambda x: np.rot90(x, k=3),
            lambda x: np.flip(x, axis=0),
            lambda x: np.flip(x, axis=1),
            lambda x: np.transpose(x),
            lambda x: self.extract_largest_component(x),
            lambda x: self.fill_holes(x),
            lambda x: self.mirror_pattern(x),
            lambda x: self.extend_pattern(x)
        ]
        
        best_op = None
        best_score = 0
        
        for op in operations:
            try:
                # اختبر العملية على أمثلة التدريب
                score = 0
                for example in task_data['train']:
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    result = op(input_grid)
                    if result.shape == output_grid.shape:
                        score += np.mean(result == output_grid)
                
                avg_score = score / len(task_data['train'])
                if avg_score > best_score:
                    best_score = avg_score
                    best_op = op
            except:
                continue
        
        if best_op and best_score > 0.8:
            return best_op(test_input)
        
        return None
    
    def solve_by_symmetry(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بالتناظر"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # فحص نوع التناظر المطلوب
        for example in task_data['train']:
            output_grid = np.array(example['output'])
            
            # تناظر أفقي
            if np.array_equal(output_grid, np.flip(output_grid, axis=0)):
                return self.make_symmetric(test_input, axis=0)
            
            # تناظر عمودي
            if np.array_equal(output_grid, np.flip(output_grid, axis=1)):
                return self.make_symmetric(test_input, axis=1)
            
            # تناظر قطري
            if output_grid.shape[0] == output_grid.shape[1]:
                if np.array_equal(output_grid, output_grid.T):
                    return self.make_diagonal_symmetric(test_input)
        
        return None
    
    def solve_by_repetition(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بالتكرار"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # فحص نمط التكرار
        for example in task_data['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # هل الإخراج تكرار للإدخال؟
            if output_grid.shape[0] == input_grid.shape[0] * 2:
                # تكرار عمودي
                return np.vstack([test_input, test_input])
            elif output_grid.shape[1] == input_grid.shape[1] * 2:
                # تكرار أفقي
                return np.hstack([test_input, test_input])
            elif output_grid.shape[0] == input_grid.shape[0] * 3:
                # تكرار ثلاثي عمودي
                return np.vstack([test_input, test_input, test_input])
        
        return None
    
    def solve_by_object_detection(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بكشف الكائنات"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # كشف الكائنات في الإدخال
        objects = self.detect_objects(test_input)
        
        # تعلم كيفية معالجة الكائنات من الأمثلة
        for example in task_data['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            train_objects = self.detect_objects(input_grid)
            
            # إذا كان عدد الكائنات متطابق
            if len(objects) == len(train_objects):
                # حاول تطبيق نفس التحويل
                if output_grid.shape == input_grid.shape:
                    # ربما يتم تلوين أكبر كائن
                    largest_obj = max(objects, key=lambda o: o['size'])
                    result = test_input.copy()
                    for pos in largest_obj['positions']:
                        result[pos] = 5  # لون مختلف
                    return result
        
        return None
    
    def solve_by_rule_learning(self, task_data: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """حل بتعلم القواعد"""
        if 'test' not in task_data or not task_data['test']:
            return None
        
        test_input = np.array(task_data['test'][0]['input'])
        
        # تعلم القواعد من الأمثلة
        rules = []
        
        for example in task_data['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # قاعدة: إذا كانت الخلية محاطة بلون معين، غيرها
            for i in range(min(input_grid.shape[0], output_grid.shape[0])):
                for j in range(min(input_grid.shape[1], output_grid.shape[1])):
                    if input_grid[i, j] != output_grid[i, j]:
                        # تحليل الجيران
                        neighbors = self.get_neighbors(input_grid, i, j)
                        rule = {
                            'condition': {'center': input_grid[i, j], 'neighbors': neighbors},
                            'action': output_grid[i, j]
                        }
                        rules.append(rule)
        
        # تطبيق القواعد على الاختبار
        if rules:
            result = test_input.copy()
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    neighbors = self.get_neighbors(result, i, j)
                    for rule in rules:
                        if (rule['condition']['center'] == result[i, j] and
                            self.match_neighbors(neighbors, rule['condition']['neighbors'])):
                            result[i, j] = rule['action']
                            break
            return result
        
        return None
    
    def find_patterns(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict]:
        """البحث عن الأنماط"""
        patterns = []
        
        # بحث عن أنماط 2x2, 3x3, etc
        for size in [2, 3, 4]:
            if (input_grid.shape[0] >= size and input_grid.shape[1] >= size and
                output_grid.shape[0] >= size and output_grid.shape[1] >= size):
                
                for i in range(input_grid.shape[0] - size + 1):
                    for j in range(input_grid.shape[1] - size + 1):
                        input_pattern = input_grid[i:i+size, j:j+size]
                        
                        # ابحث عن هذا النمط في الإخراج
                        for oi in range(output_grid.shape[0] - size + 1):
                            for oj in range(output_grid.shape[1] - size + 1):
                                output_pattern = output_grid[oi:oi+size, oj:oj+size]
                                
                                if not np.array_equal(input_pattern, output_pattern):
                                    patterns.append({
                                        'input': input_pattern,
                                        'output': output_pattern,
                                        'position': (i, j),
                                        'size': size
                                    })
        
        return patterns
    
    def detect_objects(self, grid: np.ndarray) -> List[Dict]:
        """كشف الكائنات في الشبكة"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    # بدء كائن جديد
                    obj = self.flood_fill(grid, i, j, visited)
                    if obj['size'] > 0:
                        objects.append(obj)
        
        return objects
    
    def flood_fill(self, grid: np.ndarray, start_i: int, start_j: int, 
                   visited: np.ndarray) -> Dict:
        """ملء الفيضان للعثور على كائن متصل"""
        color = grid[start_i, start_j]
        positions = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or
                visited[i, j] or grid[i, j] != color):
                continue
            
            visited[i, j] = True
            positions.append((i, j))
            
            # أضف الجيران
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return {
            'color': color,
            'positions': positions,
            'size': len(positions),
            'bounds': self.get_bounds(positions) if positions else None
        }
    
    def get_bounds(self, positions: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """الحصول على حدود الكائن"""
        if not positions:
            return (0, 0, 0, 0)
        
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        
        return (min(rows), max(rows), min(cols), max(cols))
    
    def check_symmetry(self, grid: np.ndarray) -> Dict:
        """فحص أنواع التناظر"""
        return {
            'horizontal': np.array_equal(grid, np.flip(grid, axis=0)),
            'vertical': np.array_equal(grid, np.flip(grid, axis=1)),
            'diagonal': np.array_equal(grid, grid.T) if grid.shape[0] == grid.shape[1] else False,
            'rotational_90': np.array_equal(grid, np.rot90(grid, k=2))
        }
    
    def find_common_pattern(self, examples: List[Dict]) -> Optional[Dict]:
        """إيجاد نمط مشترك بين الأمثلة"""
        if not examples:
            return None
        
        # جمع كل التحولات الممكنة
        transformations = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # حاول إيجاد تحويل يربط الإدخال بالإخراج
            transform = self.find_transformation(input_grid, output_grid)
            if transform:
                transformations.append(transform)
        
        # إيجاد التحويل الأكثر شيوعاً
        if transformations:
            # استخدم أول تحويل ناجح
            return transformations[0]
        
        return None
    
    def find_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """إيجاد التحويل بين الإدخال والإخراج"""
        # فحص التحولات البسيطة
        if np.array_equal(output_grid, np.rot90(input_grid, k=1)):
            return {'type': 'rotate', 'k': 1}
        elif np.array_equal(output_grid, np.rot90(input_grid, k=2)):
            return {'type': 'rotate', 'k': 2}
        elif np.array_equal(output_grid, np.rot90(input_grid, k=3)):
            return {'type': 'rotate', 'k': 3}
        elif np.array_equal(output_grid, np.flip(input_grid, axis=0)):
            return {'type': 'flip', 'axis': 0}
        elif np.array_equal(output_grid, np.flip(input_grid, axis=1)):
            return {'type': 'flip', 'axis': 1}
        elif input_grid.shape == output_grid.shape and np.array_equal(output_grid, input_grid.T):
            return {'type': 'transpose'}
        
        # فحص تحويلات أكثر تعقيداً
        if input_grid.shape != output_grid.shape:
            # تغيير الحجم
            return {
                'type': 'resize',
                'from_shape': input_grid.shape,
                'to_shape': output_grid.shape
            }
        
        return None
    
    def apply_pattern(self, grid: np.ndarray, pattern: Dict) -> np.ndarray:
        """تطبيق نمط على الشبكة"""
        if not pattern:
            return grid
        
        result = grid.copy()
        
        if 'type' in pattern:
            if pattern['type'] == 'rotate':
                return np.rot90(grid, k=pattern.get('k', 1))
            elif pattern['type'] == 'flip':
                return np.flip(grid, axis=pattern.get('axis', 0))
            elif pattern['type'] == 'transpose':
                return grid.T
            elif pattern['type'] == 'resize':
                # محاولة تغيير الحجم بطريقة ذكية
                to_shape = pattern['to_shape']
                if to_shape[0] > grid.shape[0] and to_shape[1] > grid.shape[1]:
                    # تكبير
                    result = np.zeros(to_shape, dtype=grid.dtype)
                    result[:grid.shape[0], :grid.shape[1]] = grid
                elif to_shape[0] < grid.shape[0] and to_shape[1] < grid.shape[1]:
                    # تصغير
                    result = grid[:to_shape[0], :to_shape[1]]
                else:
                    result = grid
        
        return result
    
    def apply_transformation(self, grid: np.ndarray, transform: Dict) -> np.ndarray:
        """تطبيق تحويل على الشبكة"""
        return self.apply_pattern(grid, transform)
    
    def learn_transformation(self, examples: List[Dict]) -> Optional[Dict]:
        """تعلم التحويل من الأمثلة"""
        return self.find_common_pattern(examples)
    
    def learn_color_mapping(self, examples: List[Dict]) -> Optional[Dict[int, int]]:
        """تعلم تحويل الألوان"""
        color_maps = []
        
        for example in examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            if input_grid.shape != output_grid.shape:
                continue
            
            # إنشاء خريطة ألوان
            color_map = {}
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    if input_grid[i, j] != output_grid[i, j]:
                        color_map[int(input_grid[i, j])] = int(output_grid[i, j])
            
            if color_map:
                color_maps.append(color_map)
        
        # إيجاد الخريطة الأكثر شيوعاً
        if color_maps:
            return color_maps[0]
        
        return None
    
    def extract_largest_component(self, grid: np.ndarray) -> np.ndarray:
        """استخراج أكبر مكون متصل"""
        objects = self.detect_objects(grid)
        
        if not objects:
            return grid
        
        largest = max(objects, key=lambda o: o['size'])
        result = np.zeros_like(grid)
        
        for pos in largest['positions']:
            result[pos] = grid[pos]
        
        return result
    
    def fill_holes(self, grid: np.ndarray) -> np.ndarray:
        """ملء الثقوب في الشبكة"""
        result = grid.copy()
        
        for i in range(1, grid.shape[0] - 1):
            for j in range(1, grid.shape[1] - 1):
                if grid[i, j] == 0:
                    # فحص الجيران
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    non_zero = [n for n in neighbors if n != 0]
                    
                    if len(non_zero) >= 3:
                        # ملء بأكثر لون شيوعاً
                        result[i, j] = Counter(non_zero).most_common(1)[0][0]
        
        return result
    
    def mirror_pattern(self, grid: np.ndarray) -> np.ndarray:
        """عكس النمط"""
        # جرب أنواع مختلفة من العكس
        h, w = grid.shape
        
        # عكس أفقي
        if h % 2 == 0:
            top_half = grid[:h//2, :]
            result = np.vstack([top_half, np.flip(top_half, axis=0)])
            if result.shape == grid.shape:
                return result
        
        # عكس عمودي
        if w % 2 == 0:
            left_half = grid[:, :w//2]
            result = np.hstack([left_half, np.flip(left_half, axis=1)])
            if result.shape == grid.shape:
                return result
        
        return grid
    
    def extend_pattern(self, grid: np.ndarray) -> np.ndarray:
        """توسيع النمط"""
        # ابحث عن نمط صغير وكرره
        h, w = grid.shape
        
        # جرب أحجام مختلفة للنمط
        for pattern_h in [2, 3, 4]:
            for pattern_w in [2, 3, 4]:
                if h >= pattern_h and w >= pattern_w:
                    pattern = grid[:pattern_h, :pattern_w]
                    
                    # حاول ملء الشبكة بالنمط
                    result = np.zeros_like(grid)
                    for i in range(0, h, pattern_h):
                        for j in range(0, w, pattern_w):
                            end_i = min(i + pattern_h, h)
                            end_j = min(j + pattern_w, w)
                            result[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]
                    
                    # تحقق إذا كان هذا يطابق جزء من الشبكة الأصلية
                    if np.array_equal(result[:pattern_h*2, :pattern_w*2], 
                                     grid[:pattern_h*2, :pattern_w*2]):
                        return result
        
        return grid
    
    def make_symmetric(self, grid: np.ndarray, axis: int) -> np.ndarray:
        """جعل الشبكة متناظرة"""
        if axis == 0:  # تناظر أفقي
            half = grid.shape[0] // 2
            top_half = grid[:half, :]
            return np.vstack([top_half, np.flip(top_half, axis=0)])
        elif axis == 1:  # تناظر عمودي
            half = grid.shape[1] // 2
            left_half = grid[:, :half]
            return np.hstack([left_half, np.flip(left_half, axis=1)])
        
        return grid
    
    def make_diagonal_symmetric(self, grid: np.ndarray) -> np.ndarray:
        """جعل الشبكة متناظرة قطرياً"""
        if grid.shape[0] != grid.shape[1]:
            return grid
        
        result = grid.copy()
        n = grid.shape[0]
        
        for i in range(n):
            for j in range(i+1, n):
                # جعل العناصر متناظرة حول القطر
                result[j, i] = result[i, j]
        
        return result
    
    def get_neighbors(self, grid: np.ndarray, i: int, j: int) -> List[int]:
        """الحصول على قيم الجيران"""
        neighbors = []
        h, w = grid.shape
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbors.append(int(grid[ni, nj]))
                else:
                    neighbors.append(-1)  # خارج الحدود
        
        return neighbors
    
    def match_neighbors(self, neighbors1: List[int], neighbors2: List[int]) -> bool:
        """مطابقة الجيران"""
        if len(neighbors1) != len(neighbors2):
            return False
        
        # مطابقة مرنة - على الأقل 75% تطابق
        matches = sum(1 for n1, n2 in zip(neighbors1, neighbors2) if n1 == n2)
        return matches >= len(neighbors1) * 0.75
    
    def evaluate_solution(self, solution: np.ndarray, task_data: Dict) -> float:
        """تقييم جودة الحل"""
        scores = []
        
        # تقييم على أمثلة التدريب
        for example in task_data['train']:
            output = np.array(example['output'])
            if solution.shape == output.shape:
                score = np.mean(solution == output)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def fallback_solution(self, task_data: Dict) -> np.ndarray:
        """حل احتياطي"""
        if task_data['train']:
            # أرجع أول إخراج من التدريب
            return np.array(task_data['train'][0]['output'])
        
        return np.zeros((3, 3))

# دالة للاستخدام المباشر
def solve_task(task_data: Dict) -> np.ndarray:
    solver = EnhancedARCSolver()
    return solver.solve(task_data)
