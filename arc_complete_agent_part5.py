from __future__ import annotations
# -*- coding: utf-8 -*-
"""
ARC ULTIMATE REVOLUTIONARY INTELLIGENT AGENT - PART 5
=====================================================
🧠  :    -  MuZero
🎯       ARC
Author: Nabil Alagi
: v1.0 -   
: 2025
:        
"""

import os
import sys
import json
import time
import math
import random
from collections.abc import Callable
from typing import Dict, List, Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    from dependency_manager import safe_import_numpy, safe_import_torch
    np = safe_import_numpy()
    torch = safe_import_torch()
    if torch:
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = False
        # Create dummy classes for torch components
        class nn:
            class Module: pass
            class Linear: pass
            class ReLU: pass
            class Sequential: pass
        class F:
            @staticmethod
            def relu(x): return x
            @staticmethod
            def softmax(x, dim=None): return x
        class optim:
            class Adam: pass
except ImportError:
    import numpy as np
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print("Warning: PyTorch not available, deep learning features disabled")

#      
try:
    from arc_complete_agent_part1 import UltraAdvancedGridCalculusEngine, memory_manager
    from arc_complete_agent_part3 import AdvancedStrategyManager
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("⚠️ Components from other parts not found, using simplified versions")

# =============================================================================
# MUZERO CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class MuZeroConfig:
    """    MuZero"""
    
    # Network Architecture Parameters
    hidden_state_size: int = 256
    representation_layers: List[int] = field(default_factory=lambda: [512, 256, 256])
    dynamics_layers: List[int] = field(default_factory=lambda: [256, 256, 256])
    prediction_layers: List[int] = field(default_factory=lambda: [256, 256])
    
    # MCTS Parameters
    num_simulations: int = 50
    max_depth: int = 20
    c_puct: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    
    # Training Parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 10.0
    batch_size: int = 32
    unroll_steps: int = 5
    td_steps: int = 10
    
    # ARC-specific Parameters
    action_space_size: int = 100
    max_grid_size: int = 30
    max_colors: int = 10
    
    # Optimization Parameters
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.1
    
    # Memory and Experience
    replay_buffer_size: int = 10000
    min_replay_buffer_size: int = 1000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    
    # Advanced Features
    use_self_play: bool = True
    use_muzero_reanalyze: bool = True
    use_fresh_learner: bool = False
    model_based_value: bool = True
    
    # Exploration and Exploitation
    known_bounds: Optional[Tuple[float, float]] = None
    max_visits_init: int = 50
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class ResidualBlock(nn.Module):
    """   """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        return F.relu(out)

class AttentionMechanism(nn.Module):
    """    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1] if len(x.shape) > 2 else 1
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output(attention_output)
        
        if seq_len == 1:
            output = output.squeeze(1)
            x = x.squeeze(1)
        
        return self.layer_norm(x + output)

class RepresentationNetwork(nn.Module):
    """  -     """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        #   
        self.conv_layers = nn.ModuleList()
        in_channels = config.max_colors
        
        for i, out_channels in enumerate([32, 64, 128]):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            in_channels = out_channels
        
        #  
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128, 128) for _ in range(3)
        ])
        
        #      
        self.flatten_size = 128 * config.max_grid_size * config.max_grid_size
        
        #   
        self.fc_layers = nn.ModuleList()
        in_features = self.flatten_size
        
        for out_features in config.representation_layers:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            in_features = out_features
        
        #   
        self.output_layer = nn.Linear(config.representation_layers[-1], config.hidden_state_size)
        
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
            
        Args:
            observation:   [batch_size, colors, height, width]
        Returns:
            hidden_state:   [batch_size, hidden_state_size]
        """
        x = observation
        
        #   
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        #   
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        #      
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        #   
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        #  
        hidden_state = self.output_layer(x)
        hidden_state = F.tanh(hidden_state)
        
        return hidden_state

class DynamicsNetwork(nn.Module):
    """  -    """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        #    
        input_dim = config.hidden_state_size + config.action_space_size
        
        self.fusion_layers = nn.ModuleList()
        in_features = input_dim
        
        for out_features in config.dynamics_layers:
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            in_features = out_features
        
        #   
        self.next_state_head = nn.Sequential(
            nn.Linear(config.dynamics_layers[-1], config.hidden_state_size),
            nn.Tanh()
        )
        
        #  
        self.reward_head = nn.Sequential(
            nn.Linear(config.dynamics_layers[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
           
        Args:
            hidden_state:   [batch_size, hidden_state_size]
            action:  [batch_size, action_space_size]
        Returns:
            next_hidden_state:   [batch_size, hidden_state_size]
            reward:  [batch_size, 1]
        """
        #   
        x = torch.cat([hidden_state, action], dim=-1)
        
        #   
        for fusion_layer in self.fusion_layers:
            x = fusion_layer(x)
        
        #  
        next_hidden_state = self.next_state_head(x)
        
        # 
        reward = self.reward_head(x)
        
        return next_hidden_state, reward

class PredictionNetwork(nn.Module):
    """  -   """
    
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        #  
        self.shared_layers = nn.ModuleList()
        in_features = config.hidden_state_size
        
        for out_features in config.prediction_layers:
            self.shared_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
            in_features = out_features
        
        #  
        self.policy_head = nn.Sequential(
            nn.Linear(config.prediction_layers[-1], config.action_space_size),
            nn.LogSoftmax(dim=-1)
        )
        
        #  
        self.value_head = nn.Sequential(
            nn.Linear(config.prediction_layers[-1], 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
          
        Args:
            hidden_state:   [batch_size, hidden_state_size]
        Returns:
            policy_logits:  [batch_size, action_space_size]
            value:  [batch_size, 1]
        """
        x = hidden_state
        
        #   
        for shared_layer in self.shared_layers:
            x = shared_layer(x)
        
        # 
        policy_logits = self.policy_head(x)
        
        # 
        value = self.value_head(x)
        
        return policy_logits, value

# =============================================================================
# MONTE CARLO TREE SEARCH (MCTS)
# =============================================================================

class Node:
    """   MCTS"""
    
    def __init__(self, prior_probability: float = 0.0, parent: Optional['Node'] = None):
        self.parent = parent
        self.children = {}
        
        #  MCTS
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_probability = prior_probability
        self.reward = 0.0
        
        #  
        self.hidden_state = None
        self.is_expanded = False
        
    @property
    def value(self) -> float:
        """  """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_root(self) -> bool:
        """     """
        return self.parent is None
    
    def is_leaf(self) -> bool:
        """    """
        return len(self.children) == 0
    
    def select_action(self, temperature: float = 1.0) -> int:
        """     """
        if not self.children:
            return 0
        
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        
        if temperature == 0:
            #  
            action_idx = np.argmax(visit_counts)
        else:
            #  
            visit_counts = visit_counts ** (1 / temperature)
            probabilities = visit_counts / np.sum(visit_counts)
            action_idx = np.random.choice(len(actions), p=probabilities)
        
        return actions[action_idx]
    
    def add_child(self, action: int, prior_probability: float) -> 'Node':
        """  """
        child = Node(prior_probability=prior_probability, parent=self)
        self.children[action] = child
        return child
    
    def backup(self, value: float):
        """    """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)

class MCTS:
    """ Monte Carlo Tree Search """
    
    def __init__(self, config: MuZeroConfig):
        self.config = config
        
    def select_child(self, node: Node) -> Tuple[int, Node]:
        """     UCB"""
        best_score = -float('inf')
        best_action = 0
        best_child = None
        
        for action, child in node.children.items():
            score = self._ucb_score(node, child)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _ucb_score(self, parent: Node, child: Node) -> float:
        """  UCB """
        if child.visit_count == 0:
            return float('inf')
        
        #  PUCT 
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior_probability
        value_score = child.value
        
        return prior_score + value_score
    
    def expand_node(self, node: Node, hidden_state: torch.Tensor, 
                   policy_logits: torch.Tensor) -> None:
        """   """
        policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
        
        for action in range(len(policy_probs)):
            if policy_probs[action] > 1e-8:
                node.add_child(action, policy_probs[action])
        
        node.hidden_state = hidden_state
        node.is_expanded = True
    
    def add_exploration_noise(self, node: Node) -> None:
        """   """
        if not node.is_root():
            return
        
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        
        for action, noise_val in zip(actions, noise):
            child = node.children[action]
            child.prior_probability = (
                child.prior_probability * (1 - self.config.root_exploration_fraction) + 
                noise_val * self.config.root_exploration_fraction
            )

# =============================================================================
# REPLAY BUFFER
# =============================================================================

@dataclass
class Experience:
    """     """
    observation: torch.Tensor
    actions: List[int]
    rewards: List[float]
    policy_targets: List[torch.Tensor]
    value_targets: List[float]
    priority: float = 1.0

class ReplayBuffer:
    """    """
    
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.buffer = []
        self.priorities = deque(maxlen=config.replay_buffer_size)
        self.max_priority = 1.0
        
    def add(self, experience: Experience) -> None:
        """  """
        if len(self.buffer) >= self.config.replay_buffer_size:
            self.buffer.pop(0)
        
        experience.priority = self.max_priority
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """      """
        if len(self.buffer) < self.config.min_replay_buffer_size:
            return []
        
        #   
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.config.priority_alpha
        probabilities = probabilities / probabilities.sum()
        
        #  
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        experiences = [self.buffer[i] for i in indices]
        
        return experiences
    
    def update_priorities(self, experiences: List[Experience], td_errors: List[float]) -> None:
        """  """
        for experience, td_error in zip(experiences, td_errors):
            priority = abs(td_error) + 1e-6
            experience.priority = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)

# =============================================================================
# ARC ENVIRONMENT
# =============================================================================

class ARCEnvironment:
    """    ARC      ."""

    def __init__(self, strategy_manager: 'AdvancedStrategyManager', input_grid: np.ndarray, target_grid: np.ndarray):
        self.strategy_manager = strategy_manager
        self.initial_grid = input_grid.copy()
        self.target_grid = target_grid.copy()
        self.current_grid = input_grid.copy()
        self.action_names = list(getattr(self.strategy_manager, 'basic_strategies', {}).keys())
        self.max_steps = 10
        self.steps = 0
        self.done = False

    def reset(self) -> np.ndarray:
        self.current_grid = self.initial_grid.copy()
        self.steps = 0
        self.done = False
        return self.current_grid

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self.current_grid, 0.0, True, {'reason': 'already_done'}

        if not self.action_names:
            self.done = True
            return self.current_grid, 0.0, True, {'reason': 'no_actions'}

        name = self.action_names[action_index % len(self.action_names)]
        try:
            next_grid = self.strategy_manager.apply_strategy(name, self.current_grid, {})
        except Exception:
            next_grid = self.current_grid

        # :    
        if next_grid.shape == self.target_grid.shape:
            reward = float(np.mean(next_grid == self.target_grid))
        else:
            reward = 0.0

        self.current_grid = next_grid
        self.steps += 1
        self.done = bool(reward >= 0.999 or self.steps >= self.max_steps)
        return self.current_grid, reward, self.done, {'action': name}

# =============================================================================
# MUZERO AGENT
# =============================================================================

class MuZeroAgent:
    """   MuZero     ARC"""
    
    def __init__(self, config: MuZeroConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing MuZero Agent on {self.device}")
        
        #  
        self.representation_network = RepresentationNetwork(config).to(self.device)
        self.dynamics_network = DynamicsNetwork(config).to(self.device)
        self.prediction_network = PredictionNetwork(config).to(self.device)
        
        # 
        parameters = (list(self.representation_network.parameters()) + 
                     list(self.dynamics_network.parameters()) + 
                     list(self.prediction_network.parameters()))
        
        self.optimizer = optim.Adam(parameters, 
                                   lr=config.learning_rate, 
                                   weight_decay=config.weight_decay)
        
        #   
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # MCTS
        self.mcts = MCTS(config)
        
        #   
        self.replay_buffer = ReplayBuffer(config)
        
        #   (  )
        if COMPONENTS_AVAILABLE:
            self.calculus_engine = UltraAdvancedGridCalculusEngine()
            self.strategy_manager = AdvancedStrategyManager()
        else:
            self.calculus_engine = None
            self.strategy_manager = None
        
        #  
        self.training_stats = {
            'total_steps': 0,
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'total_loss': [],
            'learning_rate': []
        }
        
        print("✅ MuZero Agent initialized successfully!")
    
    def plan_action(self, observation: np.ndarray, context: Dict[str, Any]) -> int:
        """
            MCTS
        Args:
            observation:  
            context:  
        Returns:
            action:   
        """
        #    tensor
        obs_tensor = self._observation_to_tensor(observation)
        
        #    
        with torch.no_grad():
            hidden_state = self.representation_network(obs_tensor)
            policy_logits, value = self.prediction_network(hidden_state)
        
        #   
        root = Node()
        self.mcts.expand_node(root, hidden_state, policy_logits)
        self.mcts.add_exploration_noise(root)
        
        #   MCTS
        for _ in range(self.config.num_simulations):
            self._run_simulation(root)
        
        #      
        action = root.select_action(temperature=self.config.temperature)
        
        #   
        self._store_search_statistics(root, obs_tensor)
        
        return action
    
    def _run_simulation(self, root: Node) -> None:
        """    MCTS"""
        node = root
        search_path = [node]
        actions = []
        
        #  
        while node.is_expanded and not node.is_leaf():
            action, node = self.mcts.select_child(node)
            search_path.append(node)
            actions.append(action)
        
        #   
        if not node.is_expanded:
            #       
            parent = node.parent
            if parent and parent.hidden_state is not None and actions:
                action_tensor = self._action_to_tensor(actions[-1])
                
                with torch.no_grad():
                    hidden_state, reward = self.dynamics_network(
                        parent.hidden_state, action_tensor
                    )
                    policy_logits, value = self.prediction_network(hidden_state)
                
                node.reward = reward.item()
                self.mcts.expand_node(node, hidden_state, policy_logits)
                
                #  
                for node_in_path in reversed(search_path):
                    node_in_path.backup(value.item())
            else:
                #  
                for node_in_path in reversed(search_path):
                    node_in_path.backup(0.0)
        else:
            #   
            value = node.value
            for node_in_path in reversed(search_path):
                node_in_path.backup(value)
    
    def train(self, batch_size: int = None) -> Dict[str, float]:
        """
          
        Args:
            batch_size:  
        Returns:
            training_metrics:  
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        #      
        experiences = self.replay_buffer.sample(batch_size)
        if not experiences:
            return {}
        
        #  
        observations = torch.stack([exp.observation for exp in experiences]).to(self.device)
        
        #  
        policy_loss, value_loss, reward_loss = self._compute_losses(experiences)
        
        #  
        total_loss = (
            self.config.policy_loss_weight * policy_loss +
            self.config.value_loss_weight * value_loss +
            self.config.reward_loss_weight * reward_loss
        )
        
        # 
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for network in [self.representation_network, self.dynamics_network, self.prediction_network]
             for p in network.parameters()],
            self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        
        #  
        self.training_stats['total_steps'] += 1
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['reward_loss'].append(reward_loss.item())
        self.training_stats['total_loss'].append(total_loss.item())
        self.training_stats['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item(),
            'total_loss': total_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def _compute_losses(self, experiences: List[Experience]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
          
        Args:
            experiences:  
        Returns:
            policy_loss:  
            value_loss:  
            reward_loss:  
        """
        batch_size = len(experiences)
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        reward_loss_sum = 0.0
        
        for experience in experiences:
            observation = experience.observation.to(self.device).unsqueeze(0)
            
            #  
            hidden_state = self.representation_network(observation)
            
            #   
            _, initial_value = self.prediction_network(hidden_state)
            value_target = torch.tensor([experience.value_targets[0]], dtype=torch.float32).to(self.device)
            value_loss_sum += F.mse_loss(initial_value.squeeze(), value_target)
            
            #    
            for step in range(min(len(experience.actions), self.config.unroll_steps)):
                action = experience.actions[step]
                action_tensor = self._action_to_tensor(action).to(self.device)
                
                # 
                hidden_state, predicted_reward = self.dynamics_network(hidden_state, action_tensor)
                
                #  
                reward_target = torch.tensor([experience.rewards[step]], dtype=torch.float32).to(self.device)
                reward_loss_sum += F.mse_loss(predicted_reward.squeeze(), reward_target)
                
                # 
                policy_logits, predicted_value = self.prediction_network(hidden_state)
                
                #  
                if step < len(experience.policy_targets):
                    policy_target = experience.policy_targets[step].to(self.device)
                    policy_loss_sum += F.kl_div(
                        F.log_softmax(policy_logits, dim=-1),
                        policy_target,
                        reduction='batchmean'
                    )
                
                #  
                if step + 1 < len(experience.value_targets):
                    value_target = torch.tensor([experience.value_targets[step + 1]], dtype=torch.float32).to(self.device)
                    value_loss_sum += F.mse_loss(predicted_value.squeeze(), value_target)
        
        #  
        num_steps = batch_size * self.config.unroll_steps
        policy_loss = policy_loss_sum / num_steps
        value_loss = value_loss_sum / num_steps
        reward_loss = reward_loss_sum / num_steps
        
        return policy_loss, value_loss, reward_loss
    
    def _observation_to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        """
           tensor
        Args:
            observation:   numpy array
        Returns:
            tensor:   torch tensor
        """
        #       (  )
        height, width = observation.shape
        num_colors = self.config.max_colors
        
        #  tensor  
        tensor = torch.zeros((1, num_colors, self.config.max_grid_size, self.config.max_grid_size))
        
        #     
        for color in range(num_colors):
            mask = (observation == color)
            tensor[0, color, :height, :width] = torch.from_numpy(mask.astype(np.float32))
        
        return tensor.to(self.device)
    
    def _action_to_tensor(self, action: int) -> torch.Tensor:
        """
           tensor one-hot
        Args:
            action:  
        Returns:
            tensor:  one-hot 
        """
        action_tensor = torch.zeros(1, self.config.action_space_size)
        action_tensor[0, action] = 1.0
        return action_tensor.to(self.device)
    
    def _store_search_statistics(self, root: Node, observation: torch.Tensor) -> None:
        """
           
        Args:
            root:  
            observation:  
        """
        #   
        actions = []
        rewards = []
        policy_targets = []
        value_targets = []
        
        #      
        visit_counts = np.array([child.visit_count for action, child in sorted(root.children.items())])
        policy_target = visit_counts / visit_counts.sum()
        policy_targets.append(torch.from_numpy(policy_target).float())
        
        #   
        value_target = root.value
        value_targets.append(value_target)
        
        #    
        experience = Experience(
            observation=observation.squeeze(0),
            actions=actions,
            rewards=rewards,
            policy_targets=policy_targets,
            value_targets=value_targets
        )
        
        self.replay_buffer.add(experience)
    
    def save_model(self, filepath: str) -> None:
        """
           
        Args:
            filepath:  
        """
        state_dict = {
            'representation_network': self.representation_network.state_dict(),
            'dynamics_network': self.dynamics_network.state_dict(),
            'prediction_network': self.prediction_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }
        torch.save(state_dict, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
           
        Args:
            filepath:  
        """
        state_dict = torch.load(filepath, map_location=self.device)
        
        self.representation_network.load_state_dict(state_dict['representation_network'])
        self.dynamics_network.load_state_dict(state_dict['dynamics_network'])
        self.prediction_network.load_state_dict(state_dict['prediction_network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.training_stats = state_dict['training_stats']
        
        print(f"✅ Model loaded from {filepath}")
    
    def evaluate(self, observation: np.ndarray) -> Dict[str, Any]:
        """
          
        Args:
            observation:  
        Returns:
            evaluation:   
        """
        obs_tensor = self._observation_to_tensor(observation)
        
        with torch.no_grad():
            hidden_state = self.representation_network(obs_tensor)
            policy_logits, value = self.prediction_network(hidden_state)
            
            #   
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
            
            #  
            top_actions = np.argsort(policy_probs)[-5:][::-1]
            
        return {
            'value': value.item(),
            'top_actions': top_actions.tolist(),
            'action_probabilities': policy_probs.tolist(),
            'confidence': float(np.max(policy_probs))
        }

# =============================================================================
# INTEGRATION INTERFACE
# =============================================================================

class MuZeroIntegrationInterface:
    """      """
    
    def __init__(self, config: Optional[MuZeroConfig] = None):
        """
          
        Args:
            config:  MuZero
        """
        if config is None:
            config = MuZeroConfig()
        
        self.agent = MuZeroAgent(config)
        self.config = config
        
    def process_task(self, input_grid: np.ndarray, output_grid: Optional[np.ndarray] = None,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
          ARC 
        Args:
            input_grid:  
            output_grid:   ()
            context:  
        Returns:
            result:  
        """
        if context is None:
            context = {}
        
        #  
        action = self.agent.plan_action(input_grid, context)
        
        # 
        evaluation = self.agent.evaluate(input_grid)
        
        #        
        if output_grid is not None:
            self._create_training_experience(input_grid, output_grid, action)
        
        return {
            'selected_action': action,
            'evaluation': evaluation,
            'context': context
        }
    
    def _create_training_experience(self, input_grid: np.ndarray, 
                                   output_grid: np.ndarray, action: int) -> None:
        """
            
        Args:
            input_grid:  
            output_grid:   
            action:  
        """
        #     
        reward = self._calculate_reward(input_grid, output_grid, action)
        
        #  
        obs_tensor = self.agent._observation_to_tensor(input_grid)
        
        #   
        value_target = reward
        
        #    
        policy_target = torch.zeros(self.config.action_space_size)
        policy_target[action] = 1.0
        
        experience = Experience(
            observation=obs_tensor.squeeze(0),
            actions=[action],
            rewards=[reward],
            policy_targets=[policy_target],
            value_targets=[value_target]
        )
        
        self.agent.replay_buffer.add(experience)
    
    def _calculate_reward(self, input_grid: np.ndarray, 
                         output_grid: np.ndarray, action: int) -> float:
        """
         
        Args:
            input_grid:  
            output_grid:   
            action:  
        Returns:
            reward: 
        """
        #         
        #     
        similarity = np.mean(input_grid == output_grid)
        return 2 * similarity - 1  #    [-1, 1]
    
    def train_step(self) -> Dict[str, float]:
        """
          
        Returns:
            metrics:  
        """
        return self.agent.train()
    
    def save(self, filepath: str) -> None:
        """ """
        self.agent.save_model(filepath)
    
    def load(self, filepath: str) -> None:
        """ """
        self.agent.load_model(filepath)

# =============================================================================
# SELF-PLAY TRAINER
# =============================================================================

class SelfPlayTrainer:
    """   """
    
    def __init__(self, agent: MuZeroAgent, config: MuZeroConfig):
        """
          
        Args:
            agent:  MuZero
            config:  
        """
        self.agent = agent
        self.config = config
        self.episode_count = 0
        self.best_reward = -float('inf')
        
    def run_episode(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
             
        Args:
            task_data:  
        Returns:
            episode_stats:  
        """
        input_grid = task_data['input']
        output_grid = task_data.get('output', None)
        
        #  
        action = self.agent.plan_action(input_grid, {})
        
        #  
        if output_grid is not None:
            reward = self._evaluate_solution(input_grid, output_grid, action)
        else:
            reward = 0.0
        
        # 
        if len(self.agent.replay_buffer) >= self.config.min_replay_buffer_size:
            training_metrics = self.agent.train()
        else:
            training_metrics = {}
        
        self.episode_count += 1
        
        #   
        if reward > self.best_reward:
            self.best_reward = reward
        
        return {
            'episode': self.episode_count,
            'reward': reward,
            'best_reward': self.best_reward,
            'training_metrics': training_metrics
        }
    
    def _evaluate_solution(self, input_grid: np.ndarray, 
                          output_grid: np.ndarray, action: int) -> float:
        """
          
        Args:
            input_grid:  
            output_grid:   
            action:  
        Returns:
            reward: 
        """
        #        
        similarity = np.mean(input_grid == output_grid)
        return similarity

    def run_episode_arc(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """      ARC  ReplayBuffer  ."""
        input_grid = task_data['input']
        output_grid = task_data.get('output', None)

        if output_grid is None:
            output_grid = input_grid

        if not hasattr(self.agent, 'strategy_manager') or self.agent.strategy_manager is None:
            return {'episode': self.episode_count, 'reward': 0.0, 'best_reward': self.best_reward, 'training_metrics': {}}

        env = ARCEnvironment(self.agent.strategy_manager, input_grid, output_grid)
        obs = env.reset()

        actions: List[int] = []
        rewards: List[float] = [0.0]
        policy_targets: List[torch.Tensor] = []
        value_targets: List[float] = [0.0]

        for _ in range(env.max_steps):
            action = self.agent.plan_action(obs, {})
            next_obs, reward, done, info = env.step(action)

            policy = torch.zeros(self.agent.config.action_space_size)
            policy[action % self.agent.config.action_space_size] = 1.0
            policy_targets.append(policy)

            actions.append(int(action))
            rewards.append(float(reward))
            value_targets.append(float(reward))

            obs = next_obs
            if done:
                break

        obs_tensor = self.agent._observation_to_tensor(input_grid)

        exp = Experience(
            observation=obs_tensor,
            actions=actions if actions else [0],
            rewards=rewards,
            policy_targets=policy_targets if policy_targets else [torch.ones(self.agent.config.action_space_size)/self.agent.config.action_space_size],
            value_targets=value_targets
        )
        self.agent.replay_buffer.add(exp)

        if len(self.agent.replay_buffer) >= self.config.min_replay_buffer_size:
            training_metrics = self.agent.train()
        else:
            training_metrics = {}

        self.episode_count += 1
        ep_reward = float(rewards[-1]) if rewards else 0.0
        self.best_reward = max(self.best_reward, ep_reward)

        return {
            'episode': self.episode_count,
            'reward': ep_reward,
            'best_reward': self.best_reward,
            'training_metrics': training_metrics
        }

# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def test_muzero_agent():
    """   MuZero"""
    print("=" * 80)
    print("🧪 TESTING MUZERO AGENT")
    print("=" * 80)
    
    #    
    config = MuZeroConfig(
        num_simulations=10,  #    
        batch_size=4,
        replay_buffer_size=100,
        min_replay_buffer_size=10
    )
    
    #  
    agent = MuZeroAgent(config)
    
    #   
    test_grid = np.random.randint(0, 3, (5, 5))
    
    print("\n📊 Test Grid:")
    print(test_grid)
    
    #  
    print("\n🎯 Planning action...")
    action = agent.plan_action(test_grid, {})
    print(f"Selected action: {action}")
    
    #  
    print("\n📈 Evaluating position...")
    evaluation = agent.evaluate(test_grid)
    print(f"Value: {evaluation['value']:.4f}")
    print(f"Top actions: {evaluation['top_actions']}")
    print(f"Confidence: {evaluation['confidence']:.4f}")
    
    #    
    print("\n🎓 Adding training experiences...")
    for _ in range(20):
        dummy_obs = torch.randn(config.max_colors, config.max_grid_size, config.max_grid_size)
        experience = Experience(
            observation=dummy_obs,
            actions=[np.random.randint(0, config.action_space_size)],
            rewards=[np.random.random()],
            policy_targets=[torch.softmax(torch.randn(config.action_space_size), dim=0)],
            value_targets=[np.random.random()]
        )
        agent.replay_buffer.add(experience)
    
    #  
    print("\n🏋️ Training...")
    metrics = agent.train()
    if metrics:
        print(f"Policy loss: {metrics['policy_loss']:.4f}")
        print(f"Value loss: {metrics['value_loss']:.4f}")
        print(f"Reward loss: {metrics['reward_loss']:.4f}")
        print(f"Total loss: {metrics['total_loss']:.4f}")
    
    print("\n✅ All tests completed successfully!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ARC MUZERO AGENT - ADVANCED PLANNING SYSTEM")
    print("=" * 80)
    
    #  
    test_muzero_agent()
    
    print("\n" + "=" * 80)
    print("🎯 MuZero Agent Ready for Integration!")
    print("=" * 80)

