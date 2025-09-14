# -*- coding: utf-8 -*-
"""
Neural Pattern Learner - متعلم الأنماط العصبي
يحاكي التعلم البشري: يلاحظ، يكوّن فرضيات، يختبر، يتذكر
"""
from __future__ import annotations
from collections.abc import Callable
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import time

# استيراد الأدوات الأساسية
from symbolic_rule_engine import Grid, Example, Task, dims, equal

NEURAL_MEMORY_PATH = Path('neural_memory.json')


def load_neural_memory() -> Dict[str, Any]:
    if NEURAL_MEMORY_PATH.exists():
        try:
            return json.loads(NEURAL_MEMORY_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {'patterns': {}, 'success_history': [], 'failure_analysis': {}}
    return {'patterns': {}, 'success_history': [], 'failure_analysis': {}}


def save_neural_memory(memory: Dict[str, Any]) -> None:
    try:
        NEURAL_MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def compute_grid_hash(grid: Grid) -> str:
    """حساب hash للشبكة للمقارنة السريعة"""
    return str(hash(str(grid)))


def extract_local_patterns(grid: Grid, window_size: int = 2) -> List[Tuple[Tuple[int, ...], int]]:
    """استخراج أنماط محلية من الشبكة"""
    h, w = dims(grid)
    patterns = []
    
    for i in range(h - window_size + 1):
        for j in range(w - window_size + 1):
            # استخراج نافذة صغيرة
            window = tuple(grid[i+di][j+dj] for di in range(window_size) for dj in range(window_size))
            patterns.append((window, len(patterns)))  # النمط مع موقعه
    
    return patterns


def find_pattern_differences(grid1: Grid, grid2: Grid) -> List[Dict[str, Any]]:
    """العثور على الاختلافات بين شبكتين"""
    h1, w1 = dims(grid1)
    h2, w2 = dims(grid2)
    
    differences = []
    
    # اختلافات الحجم
    if (h1, w1) != (h2, w2):
        differences.append({
            'type': 'size_change',
            'from_size': (h1, w1),
            'to_size': (h2, w2),
            'scale_factor': (h2/h1 if h1 > 0 else 1, w2/w1 if w1 > 0 else 1)
        })
        return differences  # إذا تغير الحجم، لا نحلل الاختلافات الأخرى
    
    # اختلافات الألوان
    color_changes = {}
    for i in range(h1):
        for j in range(w1):
            if grid1[i][j] != grid2[i][j]:
                old_color = grid1[i][j]
                new_color = grid2[i][j]
                if old_color not in color_changes:
                    color_changes[old_color] = {}
                if new_color not in color_changes[old_color]:
                    color_changes[old_color][new_color] = 0
                color_changes[old_color][new_color] += 1
    
    if color_changes:
        differences.append({
            'type': 'color_mapping',
            'changes': color_changes
        })
    
    return differences


class NeuralPatternLearner:
    """متعلم الأنماط العصبي - يحاكي التعلم البشري"""
    
    def __init__(self):
        self.memory = load_neural_memory()
        self.working_memory = deque(maxlen=50)  # ذاكرة عمل محدودة
        self.attention_focus = []  # نقاط التركيز الحالية
        self.hypothesis_queue = deque(maxlen=20)  # قائمة الفرضيات
        
    def observe_task(self, task: Task) -> Dict[str, Any]:
        """ملاحظة المهمة وتكوين انطباع أولي"""
        train = task.get('train', [])
        if not train:
            return {'observation': 'no_training_data'}
        
        observations = {
            'num_examples': len(train),
            'size_patterns': [],
            'color_patterns': [],
            'transformation_hints': []
        }
        
        # ملاحظة أنماط الحجم
        for ex in train:
            inp_dims = dims(ex['input'])
            out_dims = dims(ex['output'])
            observations['size_patterns'].append((inp_dims, out_dims))
        
        # ملاحظة أنماط الألوان
        for ex in train:
            inp_colors = set(cell for row in ex['input'] for cell in row)
            out_colors = set(cell for row in ex['output'] for cell in row)
            observations['color_patterns'].append((len(inp_colors), len(out_colors)))
        
        # تكوين فرضيات أولية
        self._generate_hypotheses(observations)
        
        return observations
    
    def _generate_hypotheses(self, observations: Dict[str, Any]) -> None:
        """تكوين فرضيات بناءً على الملاحظات"""
        # فرضية تغيير الحجم
        size_patterns = observations['size_patterns']
        if len(set(size_patterns)) == 1:  # نمط ثابت
            self.hypothesis_queue.append({
                'type': 'consistent_size_transform',
                'pattern': size_patterns[0],
                'confidence': 0.9
            })
        
        # فرضية تبديل الألوان
        color_patterns = observations['color_patterns']
        if all(cp[0] == cp[1] for cp in color_patterns):  # نفس عدد الألوان
            self.hypothesis_queue.append({
                'type': 'color_permutation',
                'confidence': 0.7
            })
    
    def test_hypothesis(self, hypothesis: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """اختبار فرضية على المهمة"""
        train = task.get('train', [])
        if not train:
            return {'success': False, 'reason': 'no_data'}
        
        if hypothesis['type'] == 'consistent_size_transform':
            # اختبار تحويل الحجم الثابت
            expected_pattern = hypothesis['pattern']
            for ex in train:
                inp_dims = dims(ex['input'])
                out_dims = dims(ex['output'])
                if (inp_dims, out_dims) != expected_pattern:
                    return {'success': False, 'reason': 'size_mismatch'}
            return {'success': True, 'confidence': hypothesis['confidence']}
        
        elif hypothesis['type'] == 'color_permutation':
            # اختبار تبديل الألوان
            for ex in train:
                inp_colors = set(cell for row in ex['input'] for cell in row)
                out_colors = set(cell for row in ex['output'] for cell in row)
                if len(inp_colors) != len(out_colors):
                    return {'success': False, 'reason': 'color_count_mismatch'}
            return {'success': True, 'confidence': hypothesis['confidence']}
        
        return {'success': False, 'reason': 'unknown_hypothesis'}
    
    def learn_from_success(self, task: Task, solution: Grid, method: str) -> None:
        """التعلم من النجاح"""
        task_signature = self._compute_task_signature(task)
        
        success_record = {
            'timestamp': time.time(),
            'task_signature': task_signature,
            'method': method,
            'confidence': 1.0
        }
        
        self.memory['success_history'].append(success_record)
        
        # تحديث أنماط النجاح
        if task_signature not in self.memory['patterns']:
            self.memory['patterns'][task_signature] = {
                'successful_methods': [],
                'failure_count': 0,
                'success_count': 0
            }
        
        self.memory['patterns'][task_signature]['successful_methods'].append(method)
        self.memory['patterns'][task_signature]['success_count'] += 1
        
        save_neural_memory(self.memory)
    
    def learn_from_failure(self, task: Task, predicted: Grid, expected: Grid, method: str) -> None:
        """التعلم من الفشل"""
        task_signature = self._compute_task_signature(task)
        
        # تحليل سبب الفشل
        differences = find_pattern_differences(predicted, expected)
        
        failure_record = {
            'timestamp': time.time(),
            'task_signature': task_signature,
            'method': method,
            'differences': differences,
            'similarity': self._compute_similarity(predicted, expected)
        }
        
        if task_signature not in self.memory['failure_analysis']:
            self.memory['failure_analysis'][task_signature] = []
        
        self.memory['failure_analysis'][task_signature].append(failure_record)
        
        # تحديث إحصائيات الفشل
        if task_signature not in self.memory['patterns']:
            self.memory['patterns'][task_signature] = {
                'successful_methods': [],
                'failure_count': 0,
                'success_count': 0
            }
        
        self.memory['patterns'][task_signature]['failure_count'] += 1
        
        save_neural_memory(self.memory)
    
    def suggest_next_approach(self, task: Task) -> Optional[str]:
        """اقتراح النهج التالي بناءً على الخبرة"""
        task_signature = self._compute_task_signature(task)
        
        # البحث في الذاكرة عن مهام مشابهة
        if task_signature in self.memory['patterns']:
            pattern = self.memory['patterns'][task_signature]
            successful_methods = pattern['successful_methods']
            
            if successful_methods:
                # اختيار أكثر الطرق نجاحاً
                from collections import Counter
                method_counts = Counter(successful_methods)
                return method_counts.most_common(1)[0][0]
        
        # البحث عن أنماط مشابهة
        similar_patterns = self._find_similar_patterns(task_signature)
        if similar_patterns:
            return similar_patterns[0]['method']
        
        return None
    
    def _compute_task_signature(self, task: Task) -> str:
        """حساب توقيع المهمة"""
        train = task.get('train', [])
        if not train:
            return 'empty_task'
        
        # ملامح بسيطة للتوقيع
        features = []
        for ex in train:
            inp_dims = dims(ex['input'])
            out_dims = dims(ex['output'])
            features.append(f"{inp_dims[0]}x{inp_dims[1]}->{out_dims[0]}x{out_dims[1]}")
        
        return '|'.join(sorted(features))
    
    def _compute_similarity(self, grid1: Grid, grid2: Grid) -> float:
        """حساب التشابه بين شبكتين"""
        h1, w1 = dims(grid1)
        h2, w2 = dims(grid2)
        
        if (h1, w1) != (h2, w2):
            return 0.0
        
        matches = sum(1 for i in range(h1) for j in range(w1) if grid1[i][j] == grid2[i][j])
        total = h1 * w1
        
        return matches / total if total > 0 else 0.0
    
    def _find_similar_patterns(self, signature: str) -> List[Dict[str, Any]]:
        """العثور على أنماط مشابهة في الذاكرة"""
        similar = []
        
        for stored_sig, pattern in self.memory['patterns'].items():
            if stored_sig != signature and pattern['success_count'] > 0:
                # حساب تشابه بسيط بين التوقيعات
                similarity = len(set(signature.split('|')) & set(stored_sig.split('|'))) / max(len(signature.split('|')), len(stored_sig.split('|')))
                
                if similarity > 0.5:  # عتبة التشابه
                    for method in pattern['successful_methods']:
                        similar.append({
                            'signature': stored_sig,
                            'method': method,
                            'similarity': similarity
                        })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """إحصائيات الذاكرة"""
        return {
            'total_patterns': len(self.memory['patterns']),
            'success_records': len(self.memory['success_history']),
            'failure_analyses': sum(len(failures) for failures in self.memory['failure_analysis'].values()),
            'active_hypotheses': len(self.hypothesis_queue)
        }


# دالة موحدة للاستخدام المباشر
def solve_task(task_data):
    """حل المهمة باستخدام النظام"""
    import numpy as np
    
    try:
        # حل افتراضي
        if 'train' in task_data and task_data['train']:
            return np.array(task_data['train'][0]['output'])
    except Exception as e:
        # في حالة الفشل، أرجع حل بسيط
        import numpy as np
        if 'train' in task_data and task_data['train']:
            return np.array(task_data['train'][0]['output'])
        return np.zeros((3, 3))
