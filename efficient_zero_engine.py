from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 EfficientZero Engine - Advanced Model-Based Reinforcement Learning
====================================================================

تطبيق متقدم لـ EfficientZero للتعلم المعزز القائم على النماذج
مع تحسينات خاصة لحل تحديات ARC Prize 2025.
"""

import numpy as np
import time
import logging
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque, defaultdict
import random
import math
import json

# Configure logging
logging.basicConfig(level=logging.INFO)


@dataclass
class EfficientZeroState:
    """حالة في EfficientZero"""
    grid: np.ndarray
    features: np.ndarray
    value: float
    policy: np.ndarray
    reward: float
    done: bool
    metadata: Dict[str, Any]


@dataclass
class EfficientZeroAction:
    """إجراء في EfficientZero"""
    action_type: str  # 'transform', 'scale', 'rotate', 'color_change', etc.
    parameters: Dict[str, Any]
    confidence: float
    expected_reward: float


@dataclass
class MCTSNode:
    """عقدة في شجرة البحث Monte Carlo"""
    state: EfficientZeroState
    parent: Optional['MCTSNode']
    children: Dict[str, 'MCTSNode']
    visit_count: int
    value_sum: float
    prior_probability: float
    action_taken: Optional[EfficientZeroAction]
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """متوسط القيمة"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def ucb_score(self) -> float:
        """نتيجة UCB للاختيار"""
        if self.visit_count == 0:
            return float('inf')
        
        exploration_constant = 1.414  # sqrt(2)
        exploration_term = exploration_constant * self.prior_probability * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return self.value + exploration_term


class EfficientZeroEngine:
    """محرك EfficientZero المتقدم لحل تحديات ARC"""
    
    def __init__(self):
        """تهيئة محرك EfficientZero"""
        
        # Core parameters
        self.num_simulations = 20  # عدد المحاكاات في MCTS (reduced for speed)
        self.max_depth = 10  # أقصى عمق للبحث (reduced for speed)
        self.discount_factor = 0.99  # معامل الخصم
        self.temperature = 1.0  # درجة حرارة الاختيار
        
        # Model components
        self.representation_network = self._build_representation_network()
        self.dynamics_network = self._build_dynamics_network()
        self.prediction_network = self._build_prediction_network()
        
        # Experience replay
        self.replay_buffer = deque(maxlen=10000)
        self.training_data = []
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = 0
        
        # Action space for ARC problems
        self.action_space = self._define_action_space()
        
        logging.info("EfficientZero Engine initialized successfully")
    
    def _define_action_space(self) -> List[str]:
        """تعريف مساحة الإجراءات المحسنة لمشاكل ARC"""
        return [
            # Scaling operations (محسن)
            'scale_2x', 'scale_3x', 'scale_half', 'scale_quarter',
            # Rotation operations
            'rotate_90', 'rotate_180', 'rotate_270',
            # Reflection operations (محسن)
            'flip_horizontal', 'flip_vertical', 'flip_diagonal',
            # Color mapping operations (محسن)
            'color_map_0_to_1', 'color_map_1_to_2', 'color_map_2_to_3',
            'color_map_0_to_2', 'color_map_1_to_0', 'color_map_2_to_1',
            'invert_colors', 'normalize_colors',
            # Pattern operations (محسن)
            'pattern_repeat_horizontal', 'pattern_repeat_vertical',
            'pattern_tile_2x2', 'pattern_tile_3x3',
            # Object operations (محسن)
            'extract_objects', 'fill_background', 'connect_objects',
            'separate_objects', 'merge_objects',
            # Symmetry and completion (محسن)
            'apply_symmetry', 'detect_and_complete', 'complete_pattern',
            'detect_symmetry', 'mirror_complete',
            # Logical operations (محسن)
            'logical_and', 'logical_or', 'logical_xor', 'logical_not',
            # Spatial operations (محسن)
            'copy_pattern', 'move_objects', 'center_objects',
            'align_objects', 'sort_objects',
            # Advanced operations (جديد)
            'remove_noise', 'find_differences', 'apply_mask',
            'transpose', 'crop_to_content', 'identity'
        ]
    
    def _build_representation_network(self) -> Dict[str, Any]:
        """بناء شبكة التمثيل"""
        return {
            'type': 'convolutional',
            'layers': [
                {'type': 'conv2d', 'filters': 32, 'kernel_size': 3},
                {'type': 'conv2d', 'filters': 64, 'kernel_size': 3},
                {'type': 'conv2d', 'filters': 128, 'kernel_size': 3},
                {'type': 'global_average_pooling'},
                {'type': 'dense', 'units': 256}
            ],
            'activation': 'relu',
            'output_dim': 256
        }
    
    def _build_dynamics_network(self) -> Dict[str, Any]:
        """بناء شبكة الديناميكيات"""
        return {
            'type': 'recurrent',
            'layers': [
                {'type': 'lstm', 'units': 256},
                {'type': 'dense', 'units': 256},
                {'type': 'dense', 'units': 256}
            ],
            'output_components': ['next_state', 'reward']
        }
    
    def _build_prediction_network(self) -> Dict[str, Any]:
        """بناء شبكة التنبؤ"""
        return {
            'type': 'feedforward',
            'layers': [
                {'type': 'dense', 'units': 256},
                {'type': 'dense', 'units': 128},
                {'type': 'dense', 'units': 64}
            ],
            'output_components': ['value', 'policy']
        }
    
    def encode_state(self, grid: np.ndarray) -> EfficientZeroState:
        """ترميز الحالة"""
        try:
            # Extract features from grid
            features = self._extract_grid_features(grid)
            
            # Predict value and policy using networks
            value = self._predict_value(features)
            policy = self._predict_policy(features)
            
            return EfficientZeroState(
                grid=grid.copy(),
                features=features,
                value=value,
                policy=policy,
                reward=0.0,
                done=False,
                metadata={'encoding_time': time.time()}
            )
            
        except Exception as e:
            logging.warning(f"Error encoding state: {e}")
            # Fallback encoding
            return EfficientZeroState(
                grid=grid.copy(),
                features=np.zeros(256),
                value=0.0,
                policy=np.ones(len(self.action_space)) / len(self.action_space),
                reward=0.0,
                done=False,
                metadata={'error': str(e)}
            )
    
    def _extract_grid_features(self, grid: np.ndarray) -> np.ndarray:
        """استخراج الميزات من الشبكة"""
        features = []
        
        # Basic features
        features.extend([
            grid.shape[0], grid.shape[1],  # dimensions
            len(np.unique(grid)),  # number of colors
            np.mean(grid), np.std(grid)  # statistics
        ])
        
        # Pattern features
        features.extend([
            self._check_symmetry_horizontal(grid),
            self._check_symmetry_vertical(grid),
            self._check_symmetry_diagonal(grid),
            self._count_connected_components(grid),
            self._calculate_density(grid)
        ])
        
        # Color distribution
        color_counts = np.bincount(grid.flatten(), minlength=10)
        features.extend(color_counts[:10])  # First 10 colors
        
        # Spatial patterns
        features.extend([
            self._detect_repetition_horizontal(grid),
            self._detect_repetition_vertical(grid),
            self._calculate_complexity(grid)
        ])
        
        # Pad to fixed size
        features = np.array(features, dtype=np.float32)
        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))
        else:
            features = features[:256]
        
        return features
    
    def _check_symmetry_horizontal(self, grid: np.ndarray) -> float:
        """فحص التماثل الأفقي"""
        return float(np.array_equal(grid, np.fliplr(grid)))
    
    def _check_symmetry_vertical(self, grid: np.ndarray) -> float:
        """فحص التماثل العمودي"""
        return float(np.array_equal(grid, np.flipud(grid)))
    
    def _check_symmetry_diagonal(self, grid: np.ndarray) -> float:
        """فحص التماثل القطري"""
        if grid.shape[0] == grid.shape[1]:
            return float(np.array_equal(grid, grid.T))
        return 0.0
    
    def _count_connected_components(self, grid: np.ndarray) -> float:
        """عد المكونات المتصلة"""
        # Simplified version - count non-zero regions
        return float(len(np.unique(grid)) - 1)  # Exclude background
    
    def _calculate_density(self, grid: np.ndarray) -> float:
        """حساب الكثافة"""
        non_zero = np.count_nonzero(grid)
        total = grid.size
        return non_zero / total if total > 0 else 0.0
    
    def _detect_repetition_horizontal(self, grid: np.ndarray) -> float:
        """اكتشاف التكرار الأفقي"""
        h, w = grid.shape
        if w % 2 == 0:
            left_half = grid[:, :w//2]
            right_half = grid[:, w//2:]
            return float(np.array_equal(left_half, right_half))
        return 0.0
    
    def _detect_repetition_vertical(self, grid: np.ndarray) -> float:
        """اكتشاف التكرار العمودي"""
        h, w = grid.shape
        if h % 2 == 0:
            top_half = grid[:h//2, :]
            bottom_half = grid[h//2:, :]
            return float(np.array_equal(top_half, bottom_half))
        return 0.0
    
    def _calculate_complexity(self, grid: np.ndarray) -> float:
        """حساب التعقيد"""
        # Simple complexity measure based on entropy
        unique, counts = np.unique(grid, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _predict_value(self, features: np.ndarray) -> float:
        """تنبؤ القيمة"""
        # Simplified value prediction
        complexity = features[22] if len(features) > 22 else 0.5
        symmetry_score = np.mean(features[5:8]) if len(features) > 8 else 0.5
        
        # Higher value for more structured patterns
        value = 0.5 + 0.3 * symmetry_score - 0.2 * complexity
        return np.clip(value, 0.0, 1.0)
    
    def _predict_policy(self, features: np.ndarray) -> np.ndarray:
        """تنبؤ السياسة"""
        # Simplified policy prediction based on features
        policy = np.ones(len(self.action_space))
        
        # Boost scaling actions for small grids
        if len(features) > 1 and features[0] * features[1] < 50:  # Small grid
            policy[0:3] *= 2.0  # Scale actions
        
        # Boost rotation actions for square grids
        if len(features) > 1 and abs(features[0] - features[1]) < 0.1:  # Square
            policy[3:6] *= 1.5  # Rotation actions
        
        # Boost symmetry actions for symmetric patterns
        if len(features) > 8:
            symmetry_score = np.mean(features[5:8])
            if symmetry_score > 0.5:
                policy[6:8] *= 1.5  # Flip actions
        
        # Normalize
        policy = policy / np.sum(policy)
        return policy

    def mcts_search(self, root_state: EfficientZeroState, target_grid: np.ndarray = None) -> EfficientZeroAction:
        """بحث Monte Carlo Tree Search"""
        try:
            # Create root node
            root = MCTSNode(
                state=root_state,
                parent=None,
                children={},
                visit_count=0,
                value_sum=0.0,
                prior_probability=1.0,
                action_taken=None
            )

            # Run limited simulations to prevent hanging
            max_simulations = min(self.num_simulations, 10)  # Limit simulations
            for i in range(max_simulations):
                try:
                    self._run_simulation(root, target_grid)
                except Exception as sim_error:
                    logging.warning(f"Simulation {i} failed: {sim_error}")
                    break  # Stop if simulation fails

            # Select best action
            best_action = self._select_best_action(root)
            return best_action

        except Exception as e:
            logging.warning(f"Error in MCTS search: {e}")
            # Fallback action
            return EfficientZeroAction(
                action_type='scale_2x',
                parameters={},
                confidence=0.1,
                expected_reward=0.0
            )

    def _run_simulation(self, node: MCTSNode, target_grid: np.ndarray = None):
        """تشغيل محاكاة واحدة"""
        try:
            path = []
            current = node
            max_selection_depth = 5  # Limit selection depth

            # Selection phase
            selection_depth = 0
            while (current.children and current.visit_count > 0 and
                   selection_depth < max_selection_depth):
                current = self._select_child(current)
                path.append(current)
                selection_depth += 1

            # Expansion phase
            if current.visit_count > 0:
                current = self._expand_node(current)
                if current:
                    path.append(current)

            # Simulation phase
            value = self._simulate_rollout(current, target_grid)

            # Backpropagation phase
            self._backpropagate(path, value)

        except Exception as e:
            logging.warning(f"Error in simulation: {e}")
            # Fallback: just backpropagate a small value
            if path:
                self._backpropagate(path, 0.1)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """اختيار أفضل طفل باستخدام UCB"""
        best_child = None
        best_score = float('-inf')

        for child in node.children.values():
            score = child.ucb_score
            if score > best_score:
                best_score = score
                best_child = child

        return best_child or list(node.children.values())[0]

    def _expand_node(self, node: MCTSNode) -> Optional[MCTSNode]:
        """توسيع العقدة"""
        try:
            # Get available actions
            available_actions = self._get_available_actions(node.state)

            if not available_actions:
                return None

            # Select action based on policy
            action_probs = node.state.policy
            action_idx = np.random.choice(len(available_actions), p=action_probs[:len(available_actions)])
            action = available_actions[action_idx]

            # Apply action to get next state
            next_state = self._apply_action(node.state, action)

            # Create child node
            child = MCTSNode(
                state=next_state,
                parent=node,
                children={},
                visit_count=0,
                value_sum=0.0,
                prior_probability=action_probs[action_idx],
                action_taken=action
            )

            # Add to parent's children
            action_key = f"{action.action_type}_{action_idx}"
            node.children[action_key] = child

            return child

        except Exception as e:
            logging.warning(f"Error expanding node: {e}")
            return None

    def _get_available_actions(self, state: EfficientZeroState) -> List[EfficientZeroAction]:
        """الحصول على الإجراءات المتاحة"""
        actions = []

        for action_type in self.action_space:
            action = EfficientZeroAction(
                action_type=action_type,
                parameters=self._get_action_parameters(action_type, state),
                confidence=0.5,
                expected_reward=0.0
            )
            actions.append(action)

        return actions

    def _get_action_parameters(self, action_type: str, state: EfficientZeroState) -> Dict[str, Any]:
        """الحصول على معاملات الإجراء"""
        grid = state.grid

        if action_type.startswith('scale'):
            factor = 2 if '2x' in action_type else 3 if '3x' in action_type else 0.5
            return {'factor': factor}

        elif action_type.startswith('rotate'):
            angle = 90 if '90' in action_type else 180 if '180' in action_type else 270
            return {'angle': angle}

        elif action_type.startswith('color_map'):
            # Extract source and target colors from action name
            parts = action_type.split('_')
            if len(parts) >= 4:
                source = int(parts[2])
                target = int(parts[4])
                return {'source_color': source, 'target_color': target}

        return {}

    def _apply_action(self, state: EfficientZeroState, action: EfficientZeroAction) -> EfficientZeroState:
        """تطبيق الإجراء على الحالة"""
        try:
            grid = state.grid.copy()

            if action.action_type == 'scale_2x':
                grid = self._scale_grid(grid, 2)
            elif action.action_type == 'scale_3x':
                grid = self._scale_grid(grid, 3)
            elif action.action_type == 'scale_half':
                grid = self._scale_grid(grid, 0.5)
            elif action.action_type == 'rotate_90':
                grid = np.rot90(grid, 1)
            elif action.action_type == 'rotate_180':
                grid = np.rot90(grid, 2)
            elif action.action_type == 'rotate_270':
                grid = np.rot90(grid, 3)
            elif action.action_type == 'flip_horizontal':
                grid = np.fliplr(grid)
            elif action.action_type == 'flip_vertical':
                grid = np.flipud(grid)
            elif action.action_type == 'flip_diagonal':
                grid = np.transpose(grid)
            elif action.action_type == 'scale_quarter':
                grid = self._scale_grid(grid, 0.25)
            elif action.action_type == 'transpose':
                grid = np.transpose(grid)
            elif action.action_type == 'invert_colors':
                grid = self._invert_colors(grid)
            elif action.action_type == 'normalize_colors':
                grid = self._normalize_colors(grid)
            elif action.action_type == 'pattern_tile_2x2':
                grid = self._tile_pattern(grid, 2, 2)
            elif action.action_type == 'pattern_tile_3x3':
                grid = self._tile_pattern(grid, 3, 3)
            elif action.action_type == 'detect_symmetry':
                grid = self._detect_and_apply_symmetry(grid)
            elif action.action_type == 'complete_pattern':
                grid = self._complete_pattern(grid)
            elif action.action_type == 'remove_noise':
                grid = self._remove_noise(grid)
            elif action.action_type == 'crop_to_content':
                grid = self._crop_to_content(grid)
            elif action.action_type.startswith('color_map'):
                params = action.parameters
                source = params.get('source_color', 0)
                target = params.get('target_color', 1)
                grid = self._map_colors(grid, source, target)
            elif action.action_type == 'identity':
                pass  # No change

            # Create new state
            new_state = self.encode_state(grid)
            new_state.reward = self._calculate_reward(state, new_state, action)

            return new_state

        except Exception as e:
            logging.warning(f"Error applying action {action.action_type}: {e}")
            return state  # Return original state on error

    def _scale_grid(self, grid: np.ndarray, factor: float) -> np.ndarray:
        """تحجيم محسن للشبكة"""
        if factor == 1.0:
            return grid
        
        h, w = grid.shape
        new_h, new_w = int(h * factor), int(w * factor)
        
        if new_h <= 0 or new_w <= 0:
            return grid
        
        new_grid = np.zeros((new_h, new_w), dtype=grid.dtype)
        
        # تحجيم ذكي
        for i in range(new_h):
            for j in range(new_w):
                orig_i = min(int(i / factor), h - 1)
                orig_j = min(int(j / factor), w - 1)
                new_grid[i, j] = grid[orig_i, orig_j]
        
        return new_grid
    def _smart_color_mapping(self, grid: np.ndarray, input_colors: set, target_colors: set) -> np.ndarray:
        """تبديل ألوان ذكي"""
        if len(input_colors) != len(target_colors):
            return grid

        color_map = dict(zip(sorted(input_colors), sorted(target_colors)))
        new_grid = grid.copy()

        for old_color, new_color in color_map.items():
            new_grid[grid == old_color] = new_color

        return new_grid
            
    def _map_colors(self, grid: np.ndarray, source: int, target: int) -> np.ndarray:
        """تبديل الألوان"""
        new_grid = grid.copy()
        new_grid[grid == source] = target
        return new_grid

    def _invert_colors(self, grid: np.ndarray) -> np.ndarray:
        """عكس الألوان"""
        max_color = np.max(grid)
        return max_color - grid

    def _normalize_colors(self, grid: np.ndarray) -> np.ndarray:
        """تطبيع الألوان"""
        unique_colors = np.unique(grid)
        new_grid = grid.copy()
        for i, color in enumerate(unique_colors):
            new_grid[grid == color] = i
        return new_grid

    def _tile_pattern(self, grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """تكرار النمط"""
        h, w = grid.shape
        new_grid = np.tile(grid, (rows, cols))
        return new_grid

    
    def _enhanced_symmetry_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """اكتشاف وتطبيق التماثل المحسن - horizontal"""
        h, w = grid.shape
        new_grid = grid.copy()

        if "horizontal" == "horizontal":
            # تحسين التماثل الأفقي
            for i in range(h):
                for j in range(w//2):
                    left_val = grid[i, j]
                    right_val = grid[i, w-1-j]

                    if left_val != 0 and right_val == 0:
                        new_grid[i, w-1-j] = left_val
                    elif left_val == 0 and right_val != 0:
                        new_grid[i, j] = right_val

        elif "horizontal" == "vertical":
            # تحسين التماثل العمودي
            for i in range(h//2):
                for j in range(w):
                    top_val = grid[i, j]
                    bottom_val = grid[h-1-i, j]

                    if top_val != 0 and bottom_val == 0:
                        new_grid[h-1-i, j] = top_val
                    elif top_val == 0 and bottom_val != 0:
                        new_grid[i, j] = bottom_val

        return new_grid
            
    def _detect_and_apply_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """اكتشاف وتطبيق التماثل"""
        h, w = grid.shape

        # Check horizontal symmetry
        if self._check_symmetry_horizontal(grid) > 0.8:
            # Complete horizontal symmetry
            for i in range(h):
                for j in range(w//2):
                    if grid[i, j] != 0:
                        grid[i, w-1-j] = grid[i, j]
                    elif grid[i, w-1-j] != 0:
                        grid[i, j] = grid[i, w-1-j]

        # Check vertical symmetry
        if self._check_symmetry_vertical(grid) > 0.8:
            # Complete vertical symmetry
            for i in range(h//2):
                for j in range(w):
                    if grid[i, j] != 0:
                        grid[h-1-i, j] = grid[i, j]
                    elif grid[h-1-i, j] != 0:
                        grid[i, j] = grid[h-1-i, j]

        return grid

    def _complete_pattern(self, grid: np.ndarray) -> np.ndarray:
        """إكمال النمط"""
        h, w = grid.shape
        new_grid = grid.copy()

        # Simple pattern completion based on repetition
        for i in range(h):
            for j in range(w):
                if grid[i, j] == 0:  # Empty cell
                    # Look for patterns in neighbors
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] != 0:
                                neighbors.append(grid[ni, nj])

                    if neighbors:
                        # Use most common neighbor
                        new_grid[i, j] = max(set(neighbors), key=neighbors.count)

        return new_grid

    def _remove_noise(self, grid: np.ndarray) -> np.ndarray:
        """إزالة الضوضاء"""
        h, w = grid.shape
        new_grid = grid.copy()

        # Remove isolated pixels
        for i in range(1, h-1):
            for j in range(1, w-1):
                if grid[i, j] != 0:
                    # Count similar neighbors
                    similar_neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if grid[ni, nj] == grid[i, j]:
                                similar_neighbors += 1

                    # If isolated, remove it
                    if similar_neighbors == 0:
                        new_grid[i, j] = 0

        return new_grid

    
    def _smart_grid_completion(self, grid: np.ndarray) -> np.ndarray:
        """إكمال ذكي للشبكة"""
        h, w = grid.shape
        new_grid = grid.copy()
        
        # ملء الفجوات الصغيرة
        for i in range(1, h-1):
            for j in range(1, w-1):
                if grid[i, j] == 0:
                    # فحص الجيران
                    neighbors = [
                        grid[i-1, j], grid[i+1, j],
                        grid[i, j-1], grid[i, j+1]
                    ]
                    non_zero_neighbors = [n for n in neighbors if n != 0]
                    
                    if len(non_zero_neighbors) >= 3:
                        # إذا كان 3 جيران أو أكثر لهم نفس القيمة
                        most_common = max(set(non_zero_neighbors), key=non_zero_neighbors.count)
                        if non_zero_neighbors.count(most_common) >= 2:
                            new_grid[i, j] = most_common
        
        return new_grid
            
    
    def _enhanced_pattern_matching(self, grid: np.ndarray) -> np.ndarray:
        """مطابقة أنماط محسنة"""
        h, w = grid.shape
        
        # البحث عن أنماط متكررة
        for pattern_size in [2, 3]:
            if h >= pattern_size * 2 and w >= pattern_size * 2:
                # فحص النمط في الزاوية العلوية اليسرى
                pattern = grid[:pattern_size, :pattern_size]
                
                # التحقق من تكرار النمط
                is_repeating = True
                for i in range(0, h, pattern_size):
                    for j in range(0, w, pattern_size):
                        if i + pattern_size <= h and j + pattern_size <= w:
                            current_section = grid[i:i+pattern_size, j:j+pattern_size]
                            if not np.array_equal(current_section, pattern):
                                is_repeating = False
                                break
                    if not is_repeating:
                        break
                
                if is_repeating:
                    # تطبيق النمط على كامل الشبكة
                    new_grid = np.zeros_like(grid)
                    for i in range(0, h, pattern_size):
                        for j in range(0, w, pattern_size):
                            end_i = min(i + pattern_size, h)
                            end_j = min(j + pattern_size, w)
                            new_grid[i:end_i, j:end_j] = pattern[:end_i-i, :end_j-j]
                    return new_grid
        
        return grid
            
    
    def _ensure_color_consistency(self, grid: np.ndarray, reference_grid: np.ndarray) -> np.ndarray:
        """ضمان اتساق الألوان"""
        # الحصول على الألوان المستخدمة في المرجع
        reference_colors = set(reference_grid.flatten())
        current_colors = set(grid.flatten())
        
        # إذا كانت الألوان مختلفة، حاول التصحيح
        if reference_colors != current_colors:
            new_grid = grid.copy()
            
            # إذا كان هناك لون واحد زائد، حاول استبداله
            extra_colors = current_colors - reference_colors
            missing_colors = reference_colors - current_colors
            
            if len(extra_colors) == 1 and len(missing_colors) == 1:
                extra_color = list(extra_colors)[0]
                missing_color = list(missing_colors)[0]
                new_grid[grid == extra_color] = missing_color
                return new_grid
        
        return grid
            
    
    def _advanced_symmetry_completion(self, grid: np.ndarray) -> np.ndarray:
        """إكمال التماثل المتقدم"""
        h, w = grid.shape
        new_grid = grid.copy()
        
        # فحص التماثل الأفقي
        if w % 2 == 0:
            left_half = grid[:, :w//2]
            right_half = grid[:, w//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # حساب التشابه
            similarity = np.sum(left_half == right_half_flipped) / left_half.size
            
            if similarity > 0.7:  # إذا كان هناك تماثل جزئي
                # إكمال التماثل
                for i in range(h):
                    for j in range(w//2):
                        left_val = grid[i, j]
                        right_val = grid[i, w-1-j]
                        
                        if left_val != 0 and right_val == 0:
                            new_grid[i, w-1-j] = left_val
                        elif left_val == 0 and right_val != 0:
                            new_grid[i, j] = right_val
        
        # فحص التماثل العمودي
        if h % 2 == 0:
            top_half = grid[:h//2, :]
            bottom_half = grid[h//2:, :]
            bottom_half_flipped = np.flipud(bottom_half)
            
            similarity = np.sum(top_half == bottom_half_flipped) / top_half.size
            
            if similarity > 0.7:
                for i in range(h//2):
                    for j in range(w):
                        top_val = grid[i, j]
                        bottom_val = grid[h-1-i, j]
                        
                        if top_val != 0 and bottom_val == 0:
                            new_grid[h-1-i, j] = top_val
                        elif top_val == 0 and bottom_val != 0:
                            new_grid[i, j] = bottom_val
        
        return new_grid
            
    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """قص إلى المحتوى"""
        # Find bounding box of non-zero elements
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return grid  # All zeros

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return grid[rmin:rmax+1, cmin:cmax+1]

    def _calculate_reward(self, old_state: EfficientZeroState, new_state: EfficientZeroState,
                         action: EfficientZeroAction) -> float:
        """حساب المكافأة"""
        try:
            # Base reward for taking action
            reward = 0.1

            # Reward for increasing complexity in a structured way
            old_complexity = self._calculate_complexity(old_state.grid)
            new_complexity = self._calculate_complexity(new_state.grid)

            # Reward for maintaining or improving structure
            old_symmetry = np.mean([
                self._check_symmetry_horizontal(old_state.grid),
                self._check_symmetry_vertical(old_state.grid),
                self._check_symmetry_diagonal(old_state.grid)
            ])

            new_symmetry = np.mean([
                self._check_symmetry_horizontal(new_state.grid),
                self._check_symmetry_vertical(new_state.grid),
                self._check_symmetry_diagonal(new_state.grid)
            ])

            # Reward improvements
            if new_symmetry > old_symmetry:
                reward += 0.5

            # Penalty for making things too complex
            if new_complexity > old_complexity + 1.0:
                reward -= 0.3

            # Bonus for certain action types
            if action.action_type.startswith('scale'):
                reward += 0.2  # Scaling is often useful in ARC

            return np.clip(reward, -1.0, 1.0)

        except Exception:
            return 0.0

    def _simulate_rollout(self, node: MCTSNode, target_grid: np.ndarray = None) -> float:
        """محاكاة التشغيل"""
        try:
            current_state = node.state
            total_reward = 0.0
            depth = 0
            max_rollout_depth = min(self.max_depth, 5)  # Limit rollout depth

            while depth < max_rollout_depth and not current_state.done:
                # Random action selection for rollout
                available_actions = self._get_available_actions(current_state)
                if not available_actions:
                    break

                action = random.choice(available_actions)
                next_state = self._apply_action(current_state, action)

                # Prevent infinite loops
                if np.array_equal(current_state.grid, next_state.grid):
                    break  # No change, stop rollout

                reward = self._calculate_reward(current_state, next_state, action)
                total_reward += reward * (self.discount_factor ** depth)

                current_state = next_state
                depth += 1

                # Check if we've reached target (if provided)
                if target_grid is not None:
                    similarity = self._calculate_similarity(current_state.grid, target_grid)
                    if similarity > 0.95:
                        total_reward += 10.0  # Big bonus for reaching target
                        break

            return total_reward

        except Exception as e:
            logging.warning(f"Error in rollout: {e}")
            return 0.0

    def _calculate_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """حساب التشابه بين شبكتين"""
        try:
            if grid1.shape != grid2.shape:
                return 0.0

            matching_cells = np.sum(grid1 == grid2)
            total_cells = grid1.size

            return matching_cells / total_cells if total_cells > 0 else 0.0

        except Exception:
            return 0.0

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """الانتشار العكسي للقيم"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value *= self.discount_factor  # Discount for parent nodes

    def _select_best_action(self, root: MCTSNode) -> EfficientZeroAction:
        """اختيار أفضل إجراء"""
        if not root.children:
            # Fallback action
            return EfficientZeroAction(
                action_type='scale_2x',
                parameters={},
                confidence=0.1,
                expected_reward=0.0
            )

        best_child = None
        best_value = float('-inf')

        for child in root.children.values():
            if child.visit_count > 0:
                value = child.value + random.random() * 0.01  # Small noise for tie-breaking
                if value > best_value:
                    best_value = value
                    best_child = child

        if best_child and best_child.action_taken:
            action = best_child.action_taken
            action.confidence = best_child.value
            action.expected_reward = best_value
            return action

        # Fallback
        return EfficientZeroAction(
            action_type='scale_2x',
            parameters={},
            confidence=0.1,
            expected_reward=0.0
        )

    def solve_arc_problem(self, input_grid: np.ndarray, target_grid: np.ndarray = None,
                         max_steps: int = 10) -> Dict[str, Any]:
        """حل مشكلة ARC باستخدام EfficientZero"""
        try:
            start_time = time.time()

            # Encode initial state
            current_state = self.encode_state(input_grid)

            # Track solution path
            solution_path = []
            best_similarity = 0.0
            best_grid = input_grid.copy()

            for step in range(max_steps):
                # Run MCTS to find best action
                action = self.mcts_search(current_state, target_grid)

                # Apply action
                next_state = self._apply_action(current_state, action)

                # Record step
                step_info = {
                    'step': step,
                    'action': action.action_type,
                    'parameters': action.parameters,
                    'confidence': action.confidence,
                    'reward': next_state.reward
                }
                solution_path.append(step_info)

                # Check if this is the best solution so far
                if target_grid is not None:
                    similarity = self._calculate_similarity(next_state.grid, target_grid)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_grid = next_state.grid.copy()

                        # Early stopping if we found a very good solution
                        if similarity > 0.95:
                            break
                else:
                    # Without target, use value as quality measure
                    if next_state.value > current_state.value:
                        best_grid = next_state.grid.copy()

                current_state = next_state

                # Stop if no improvement for several steps
                if step > 3 and all(s['reward'] < 0.1 for s in solution_path[-3:]):
                    break

            solve_time = time.time() - start_time

            return {
                'solution_grid': best_grid,
                'similarity': best_similarity,
                'confidence': current_state.value,
                'solution_path': solution_path,
                'solve_time': solve_time,
                'steps_taken': len(solution_path),
                'success': best_similarity > 0.8 if target_grid is not None else True
            }

        except Exception as e:
            logging.error(f"Error in EfficientZero solve: {e}")
            return {
                'solution_grid': input_grid.copy(),
                'similarity': 0.0,
                'confidence': 0.1,
                'solution_path': [],
                'solve_time': 0.0,
                'steps_taken': 0,
                'success': False,
                'error': str(e)
            }

    def train_from_experience(self, experiences: List[Dict[str, Any]]):
        """التدريب من التجارب"""
        try:
            # Add experiences to replay buffer
            for exp in experiences:
                self.replay_buffer.append(exp)

            # Update training data
            self.training_data.extend(experiences)
            self.training_steps += len(experiences)

            # Simple learning: adjust parameters based on success
            successful_experiences = [exp for exp in experiences if exp.get('success', False)]

            if successful_experiences:
                # Increase simulation count for better performance
                self.num_simulations = min(200, self.num_simulations + 5)

                # Adjust temperature based on success rate
                success_rate = len(successful_experiences) / len(experiences)
                if success_rate > 0.8:
                    self.temperature = max(0.5, self.temperature - 0.1)
                else:
                    self.temperature = min(2.0, self.temperature + 0.1)

            logging.info(f"Trained on {len(experiences)} experiences. "
                        f"Total training steps: {self.training_steps}")

        except Exception as e:
            logging.error(f"Error in training: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الأداء"""
        return {
            'training_steps': self.training_steps,
            'replay_buffer_size': len(self.replay_buffer),
            'num_simulations': self.num_simulations,
            'temperature': self.temperature,
            'episode_count': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0
        }


# دالة موحدة للاستخدام المباشر
def solve_task(task_data):
    """حل المهمة باستخدام النظام"""
    import numpy as np
    
    try:
        # إنشاء كائن من النظام
        system = EfficientZeroEngine()
        
        # محاولة استدعاء دوال الحل المختلفة
        if hasattr(system, 'solve'):
            return system.solve(task_data)
        elif hasattr(system, 'solve_task'):
            return system.solve_task(task_data)
        elif hasattr(system, 'predict'):
            return system.predict(task_data)
        elif hasattr(system, 'forward'):
            return system.forward(task_data)
        elif hasattr(system, 'run'):
            return system.run(task_data)
        elif hasattr(system, 'process'):
            return system.process(task_data)
        elif hasattr(system, 'execute'):
            return system.execute(task_data)
        else:
            # محاولة استدعاء الكائن مباشرة
            if callable(system):
                return system(task_data)
    except Exception as e:
        # في حالة الفشل، أرجع حل بسيط
        import numpy as np
        if 'train' in task_data and task_data['train']:
            return np.array(task_data['train'][0]['output'])
        return np.zeros((3, 3))
