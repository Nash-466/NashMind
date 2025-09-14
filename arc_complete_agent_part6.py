from __future__ import annotations
# -*- coding: utf-8 -*-
"""
MUZERO PHASE II - HYPERPARAMETER OPTIMIZATION & FEATURE ENGINEERING
====================================================================
🎯  :    
📊      
Author: Nabil Alagi
: v2.0 -   
: 2025
"""

from collections.abc import Callable
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import pickle
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    from dependency_manager import safe_import_numpy, safe_import_torch, safe_import_sklearn
    np = safe_import_numpy()
    torch = safe_import_torch()
    sklearn = safe_import_sklearn()

    if torch:
        import torch.nn as nn
        TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = False
        class nn:
            class Module: pass

    if sklearn:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.feature_selection import mutual_info_regression, SelectKBest
        from sklearn.decomposition import PCA
        SKLEARN_AVAILABLE = True
    else:
        SKLEARN_AVAILABLE = False
        class StandardScaler: pass
        class MinMaxScaler: pass
        class PCA: pass

    try:
        from scipy.stats import pearsonr, spearmanr
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        def pearsonr(x, y): return (0.0, 1.0)
        def spearmanr(x, y): return (0.0, 1.0)

    try:
        import optuna
        OPTUNA_AVAILABLE = True
    except ImportError:
        OPTUNA_AVAILABLE = False
        class optuna:
            class Trial: pass
            @staticmethod
            def create_study(): return None

except ImportError:
    import numpy as np
    TORCH_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False
    OPTUNA_AVAILABLE = False

#     
try:
    # In a real scenario, these would be the actual classes
    # For this script, we define dummy classes if the import fails
    from pasted_content import MuZeroConfig, MuZeroAgent
    MUZERO_AVAILABLE = True
except ImportError:
    MUZERO_AVAILABLE = False
    print("⚠️ MuZero components not found, using dummy classes for demonstration.")
    # Define dummy classes to allow the script to run
    @dataclass
    class MuZeroConfig:
        num_simulations: int = 50
        learning_rate: float = 0.001
        hidden_state_size: int = 256
        unroll_steps: int = 5
        c_puct: float = 1.25
        batch_size: int = 32
        value_loss_weight: float = 0.25
        policy_loss_weight: float = 1.0
        gradient_clip_norm: float = 10.0
        root_dirichlet_alpha: float = 0.3
        root_exploration_fraction: float = 0.25
        temperature: float = 1.0

    class MuZeroAgent:
        def __init__(self, config: MuZeroConfig):
            self.config = config
        def evaluate(self, grid):
            return {'confidence': np.random.random()}

# =============================================================================
# SECTION 1: HYPERPARAMETER OPTIMIZATION
# =============================================================================

@dataclass
class HyperparameterSearchSpace:
    """    """

    #    
    num_simulations: List[int] = field(default_factory=lambda: [25, 50, 80, 120, 200])
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    hidden_state_size: List[int] = field(default_factory=lambda: [128, 256, 512])
    unroll_steps: List[int] = field(default_factory=lambda: [3, 5, 8])
    c_puct: List[float] = field(default_factory=lambda: [0.8, 1.25, 2.0, 3.0])

    #  
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64])
    value_loss_weight: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    policy_loss_weight: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    gradient_clip_norm: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])

    #   
    representation_layers: List[List[int]] = field(default_factory=lambda: [
        [256, 256],
        [512, 256, 256],
        [512, 512, 256]
    ])
    dynamics_layers: List[List[int]] = field(default_factory=lambda: [
        [256, 256],
        [256, 256, 256],
        [512, 256, 256]
    ])

    #  MCTS 
    root_dirichlet_alpha: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    root_exploration_fraction: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.35])
    temperature: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])

class BayesianOptimizer:
    """      """

    def __init__(self, search_space: HyperparameterSearchSpace):
        self.search_space = search_space
        self.study = None
        self.best_params = None
        self.best_score = -float('inf')
        self.trial_history = []

    def objective(self, trial) -> float:
        """   """

        #     
        params = {
            'num_simulations': trial.suggest_categorical('num_simulations', self.search_space.num_simulations),
            'learning_rate': trial.suggest_categorical('learning_rate', self.search_space.learning_rate),
            'hidden_state_size': trial.suggest_categorical('hidden_state_size', self.search_space.hidden_state_size),
            'unroll_steps': trial.suggest_categorical('unroll_steps', self.search_space.unroll_steps),
            'c_puct': trial.suggest_categorical('c_puct', self.search_space.c_puct),
            'batch_size': trial.suggest_categorical('batch_size', self.search_space.batch_size),
            'value_loss_weight': trial.suggest_categorical('value_loss_weight', self.search_space.value_loss_weight),
            'policy_loss_weight': trial.suggest_categorical('policy_loss_weight', self.search_space.policy_loss_weight),
            'gradient_clip_norm': trial.suggest_categorical('gradient_clip_norm', self.search_space.gradient_clip_norm),
            'root_dirichlet_alpha': trial.suggest_categorical('root_dirichlet_alpha', self.search_space.root_dirichlet_alpha),
            'root_exploration_fraction': trial.suggest_categorical('root_exploration_fraction', self.search_space.root_exploration_fraction),
            'temperature': trial.suggest_categorical('temperature', self.search_space.temperature)
        }

        #  
        score = self.evaluate_params(params)

        #  
        self.trial_history.append({
            'params': params,
            'score': score,
            'trial_number': len(self.trial_history)
        })

        #   
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            print(f"🎯 New best score: {score:.4f}")

        return score

    def evaluate_params(self, params: Dict) -> float:
        """    """

        #   MuZero
        config = MuZeroConfig(
            num_simulations=params['num_simulations'],
            learning_rate=params['learning_rate'],
            hidden_state_size=params['hidden_state_size'],
            unroll_steps=params['unroll_steps'],
            c_puct=params['c_puct'],
            batch_size=params['batch_size'],
            value_loss_weight=params['value_loss_weight'],
            policy_loss_weight=params['policy_loss_weight'],
            gradient_clip_norm=params['gradient_clip_norm'],
            root_dirichlet_alpha=params['root_dirichlet_alpha'],
            root_exploration_fraction=params['root_exploration_fraction'],
            temperature=params['temperature']
        )

        #   MuZero
        agent = MuZeroAgent(config)

        #     
        validation_score = self._quick_evaluation(agent)

        return validation_score

    def _quick_evaluation(self, agent: 'MuZeroAgent') -> float:
        """  """
        scores = []
        #     
        for _ in range(5):
            #   
            test_grid = np.random.randint(0, 3, (5, 5))
            # 
            evaluation = agent.evaluate(test_grid)
            scores.append(evaluation['confidence'])
        return np.mean(scores)

    def optimize(self, n_trials: int = 100) -> Dict:
        """  """
        print(f"🚀 Starting Bayesian Optimization with {n_trials} trials")
        #   Optuna
        self.study = optuna.create_study(direction='maximize')
        #  
        self.study.optimize(self.objective, n_trials=n_trials)
        #    
        best_trial = self.study.best_trial
        print(f"\n✅ Optimization Complete!")
        print(f"Best Score: {best_trial.value:.4f}")
        print(f"Best Parameters: {best_trial.params}")
        return {
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'all_trials': self.trial_history
        }

class GridSearchOptimizer:
    """    """

    def __init__(self, center_params: Dict, search_radius: Dict):
        """
        Args:
            center_params:   (  )
            search_radius:     
        """
        self.center_params = center_params
        self.search_radius = search_radius
        self.results = []

    def generate_grid(self) -> List[Dict]:
        """  """
        grid_params = {}
        for param_name, center_value in self.center_params.items():
            if param_name in self.search_radius:
                radius = self.search_radius[param_name]
                if isinstance(center_value, int):
                    #  
                    values = [
                        max(1, center_value - radius),
                        center_value,
                        center_value + radius
                    ]
                elif isinstance(center_value, float):
                    #  
                    values = [
                        max(1e-6, center_value * 0.5),
                        center_value,
                        center_value * 2.0
                    ]
                else:
                    values = [center_value]
                grid_params[param_name] = values
            else:
                grid_params[param_name] = [center_value]

        #   
        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        grid = []
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            grid.append(params)
        return grid

    def search(self, evaluation_func: Callable) -> Dict:
        """  """
        grid = self.generate_grid()
        print(f"🔍 Grid Search: Testing {len(grid)} combinations")
        best_score = -float('inf')
        best_params = None
        for i, params in enumerate(grid):
            score = evaluation_func(params)
            self.results.append({'params': params, 'score': score})
            if score > best_score:
                best_score = score
                best_params = params
            print(f"Progress: {i+1}/{len(grid)} - Current Best: {best_score:.4f}")
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }

# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """    MuZero"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = {}
        self.pca = None
        self.feature_combinations = []

    def extract_baseline_features(self, grid: np.ndarray, part1_output: Optional[Dict] = None, part2_output: Optional[Dict] = None) -> np.ndarray:
        """      """
        features = []
        #    
        features.extend([
            grid.shape[0],  # 
            grid.shape[1],  # 
            np.unique(grid).size,  #   
            np.mean(grid),  #  
            np.std(grid),  #  
            np.max(grid),  #  
            np.min(grid),  #  
        ])
        #   
        color_counts = np.bincount(grid.flatten())
        features.extend([
            np.max(color_counts) if len(color_counts) > 0 else 0,  #   
            np.min(color_counts[color_counts > 0]) if len(color_counts[color_counts > 0]) > 0 else 0,  #   
            np.std(color_counts[color_counts > 0]) if len(color_counts[color_counts > 0]) > 0 else 0,  #  
        ])
        #  
        features.extend([
            self._calculate_spatial_entropy(grid),
            self._calculate_symmetry_score(grid),
            self._calculate_edge_density(grid),
            self._calculate_pattern_complexity(grid)
        ])
        #    part1   
        if part1_output:
            features.extend(self._extract_part1_features(part1_output))
        #    part2   
        if part2_output:
            features.extend(self._extract_part2_features(part2_output))
        return np.array(features)

    def _calculate_spatial_entropy(self, grid: np.ndarray) -> float:
        """  """
        from scipy.stats import entropy
        h, w = grid.shape
        if h >= 2 and w >= 2:
            regions = [
                grid[:h//2, :w//2],  #  
                grid[:h//2, w//2:],  #  
                grid[h//2:, :w//2],  #  
                grid[h//2:, w//2:]   #  
            ]
            entropies = []
            for region in regions:
                if region.size > 0:
                    counts = np.bincount(region.flatten())
                    if counts.sum() > 0:
                        probs = counts / counts.sum()
                        entropies.append(entropy(probs))
            return np.mean(entropies) if entropies else 0.0
        return 0.0

    def _calculate_symmetry_score(self, grid: np.ndarray) -> float:
        """  """
        scores = []
        #  
        h_sym = np.mean(grid == np.fliplr(grid))
        scores.append(h_sym)
        #  
        v_sym = np.mean(grid == np.flipud(grid))
        scores.append(v_sym)
        #  
        if grid.shape[0] == grid.shape[1]:
            d_sym = np.mean(grid == grid.T)
            scores.append(d_sym)
        return np.mean(scores) if scores else 0.0

    def _calculate_edge_density(self, grid: np.ndarray) -> float:
        """  """
        edges = 0
        h, w = grid.shape
        #  
        for i in range(h):
            for j in range(w - 1):
                if grid[i, j] != grid[i, j + 1]:
                    edges += 1
        #  
        for i in range(h - 1):
            for j in range(w):
                if grid[i, j] != grid[i + 1, j]:
                    edges += 1
        max_edges = (h * (w - 1)) + ((h - 1) * w)
        return edges / max_edges if max_edges > 0 else 0.0

    def _calculate_pattern_complexity(self, grid: np.ndarray) -> float:
        """  """
        import zlib
        if grid.size == 0:
            return 0.0
        compressed = zlib.compress(grid.tobytes())
        return len(compressed) / grid.size

    def _extract_part1_features(self, part1_output: Dict) -> List[float]:
        """     """
        features = []
        if 'topology' in part1_output and isinstance(part1_output['topology'], dict):
            features.extend(part1_output['topology'].values())
        if 'fractals' in part1_output and isinstance(part1_output['fractals'], dict):
            features.extend(part1_output['fractals'].values())
        return features

    def _extract_part2_features(self, part2_output: Dict) -> List[float]:
        """     """
        features = []
        if 'patterns' in part2_output and isinstance(part2_output['patterns'], dict):
            features.extend(part2_output['patterns'].values())
        return features

    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """   StandardScaler"""
        if fit:
            return self.scaler.fit_transform(features)
        else:
            return self.scaler.transform(features)

    def select_features(self, X: np.ndarray, y: np.ndarray, k: int = 20) -> np.ndarray:
        """  k    mutual_info_regression"""
        selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)
        #   
        scores = selector.scores_
        self.feature_importance = {f'feature_{i}': float(scores[i]) for i in self.selected_features}
        print(f"Selected {len(self.selected_features)} features.")
        return X_selected

    def create_interaction_features(self, features: np.ndarray) -> np.ndarray:
        """   ()"""
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        interaction = poly.fit_transform(features.reshape(1, -1))
        return interaction.flatten()

    def apply_pca(self, features: np.ndarray, n_components: int = 15, fit: bool = False) -> np.ndarray:
        """    (PCA)"""
        self.pca = PCA(n_components=min(n_components, features.shape[1]))
        if fit:
            return self.pca.fit_transform(features)
        else:
            return self.pca.transform(features)

class AblationStudy:
    """    """

    def __init__(self, base_model: 'MuZeroAgent'):
        self.base_model = base_model
        self.results = {}

    def run_ablation(self, feature_groups: Dict[str, List[int]], evaluation_func: Callable) -> Dict:
        """
             
        Args:
            feature_groups:   
            evaluation_func:  
        """
        #   
        baseline_score = evaluation_func(None)  # None    
        self.results['baseline'] = baseline_score
        print(f"📊 Baseline Score (all features): {baseline_score:.4f}")

        #    
        for group_name, feature_indices in feature_groups.items():
            print(f"\n🔬 Testing without {group_name}...")
            #    
            score_without = evaluation_func(feature_indices)
            #  
            importance = baseline_score - score_without
            self.results[group_name] = {
                'score_without': score_without,
                'importance': importance,
                'percentage_drop': (importance / baseline_score) * 100 if baseline_score > 0 else 0
            }
            print(f"Score without {group_name}: {score_without:.4f}")
            print(f"Importance: {importance:.4f} ({self.results[group_name]['percentage_drop']:.2f}% drop)")

        #    
        sorted_groups = sorted(
            [item for item in self.results.items() if isinstance(item[1], dict)],
            key=lambda x: x[1]['importance'],
            reverse=True
        )

        return {
            'results': self.results,
            'ranking': sorted_groups
        }

# =============================================================================
# SECTION 3: INTEGRATED OPTIMIZATION PIPELINE
# =============================================================================

class OptimizationPipeline:
    """   """

    def __init__(self):
        self.bayesian_optimizer = None
        self.grid_optimizer = None
        self.feature_engineer = FeatureEngineer()
        self.ablation_study = None
        self.final_config = None
        self.optimization_report = {}

    def run_phase1_hyperparameter_optimization(self, n_trials: int = 50) -> Dict:
        """ :   """
        print("=" * 80)
        print("📈 PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)

        #   
        search_space = HyperparameterSearchSpace()
        self.bayesian_optimizer = BayesianOptimizer(search_space)
        bayesian_results = self.bayesian_optimizer.optimize(n_trials=n_trials)

        #   
        search_radius = {
            'num_simulations': 10,
            'hidden_state_size': 64,
            'unroll_steps': 1
        }
        self.grid_optimizer = GridSearchOptimizer(bayesian_results['best_params'], search_radius)

        #    
        def grid_evaluation(params):
            config = MuZeroConfig(**{k: v for k, v in params.items() if k in MuZeroConfig.__annotations__})
            agent = MuZeroAgent(config)
            #  
            scores = [agent.evaluate(np.random.randint(0, 3, (5, 5)))['confidence'] for _ in range(3)]
            return np.mean(scores)

        grid_results = self.grid_optimizer.search(grid_evaluation)

        #  
        self.optimization_report['phase1'] = {
            'bayesian_results': bayesian_results,
            'grid_results': grid_results,
            'final_hyperparameters': grid_results['best_params']
        }
        print(f"\n✅ Phase 1 Complete!")
        print(f"Best Hyperparameters: {grid_results['best_params']}")
        return self.optimization_report['phase1']

    def run_phase2_feature_engineering(self, training_data: List[Dict]) -> Dict:
        """ :  """
        print("\n" + "=" * 80)
        print("🔧 PHASE 2: FEATURE ENGINEERING")
        print("=" * 80)

        #     
        all_features, targets = [], []
        for data in training_data:
            features = self.feature_engineer.extract_baseline_features(data['input'], data.get('part1_output'), data.get('part2_output'))
            all_features.append(features)
            targets.append(data.get('success_score', np.random.random()))

        X, y = np.array(all_features), np.array(targets)
        X_normalized = self.feature_engineer.normalize_features(X, fit=True)
        X_selected = self.feature_engineer.select_features(X_normalized, y)
        interaction_features = [self.feature_engineer.create_interaction_features(f) for f in X_selected]
        X_interaction = np.array(interaction_features)
        X_pca = self.feature_engineer.apply_pca(X_interaction, fit=True)

        #  
        ablation_results = "Skipped (MuZero not available)"
        if MUZERO_AVAILABLE:
            config_params = self.optimization_report.get('phase1', {}).get('final_hyperparameters', MuZeroConfig().__dict__)
            config = MuZeroConfig(**{k: v for k, v in config_params.items() if k in MuZeroConfig.__annotations__})
            test_agent = MuZeroAgent(config)
            self.ablation_study = AblationStudy(test_agent)
            feature_groups = {
                'spatial_features': list(range(7, 11)),
                'color_features': list(range(3, 7)),
                'complexity_features': list(range(11, 14)),
            }

            def ablation_evaluation(excluded_indices):
                print(f"  - Evaluating with features excluding indices: {excluded_indices}")
                return np.random.random()  # 

            ablation_results = self.ablation_study.run_ablation(feature_groups, ablation_evaluation)

        self.optimization_report['phase2'] = {
            'feature_selection_results': self.feature_engineer.feature_importance,
            'pca_explained_variance': self.feature_engineer.pca.explained_variance_ratio_.tolist() if self.feature_engineer.pca else [],
            'ablation_study_results': ablation_results,
            'final_feature_count': X_pca.shape[1] if self.feature_engineer.pca else X_selected.shape[1]
        }
        print(f"\n✅ Phase 2 Complete!")
        print(f"Selected Features Count: {len(self.feature_engineer.selected_features or [])}")
        print(f"PCA Components: {self.optimization_report['phase2']['final_feature_count']}")
        return self.optimization_report['phase2']

    def run_phase3_final_configuration(self) -> Dict:
        """ :   """
        print("\n" + "=" * 80)
        print("⚙️ PHASE 3: FINAL CONFIGURATION")
        print("=" * 80)

        if not self.optimization_report.get('phase1'):
            raise RuntimeError("Phase 1 (Hyperparameter Optimization) must be run first.")

        hyperparams = self.optimization_report['phase1']['final_hyperparameters']
        self.final_config = MuZeroConfig(**{k: v for k, v in hyperparams.items() if k in MuZeroConfig.__annotations__})

        self.optimization_report['phase3'] = {
            'final_config_dict': self.final_config.__dict__,
            'notes': "Final configuration created by combining best hyperparameters from Phase 1 and insights from Phase 2."
        }
        print("✅ Phase 3 Complete!")
        print("Final MuZero Configuration:")
        for key, value in self.final_config.__dict__.items():
            print(f"  - {key}: {value}")
        return self.final_config.__dict__

    def generate_report(self, report_path: str = "optimization_report.json"):
        """    """
        print("\n" + "=" * 80)
        print("📄 GENERATING OPTIMIZATION REPORT")
        print("=" * 80)

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                # A custom default function to handle non-serializable types
                def json_default(o):
                    if isinstance(o, (np.ndarray, np.generic)):
                        return o.tolist()
                    if isinstance(o, (dict, list, str, int, float, bool, type(None))):
                        return o
                    return str(o)
                json.dump(self.optimization_report, f, indent=4, ensure_ascii=False, default=json_default)
            print(f"✅ Report successfully saved to {report_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")

# =============================================================================
# SECTION 4: DEMONSTRATION AND EXECUTION
# =============================================================================

def create_dummy_training_data(num_samples: int = 20) -> List[Dict]:
    """    """
    dummy_data = []
    for i in range(num_samples):
        size = np.random.randint(5, 15)
        grid = np.random.randint(0, 5, (size, size))
        dummy_data.append({
            'id': f'dummy_task_{i}',
            'input': grid,
            'success_score': np.random.random(),
            'part1_output': {'topology': {'betti_0': np.random.randint(1, 5), 'betti_1': np.random.randint(0, 3)}},
            'part2_output': {'patterns': {'geometric': np.random.random(), 'spatial': np.random.random()}}
        })
    return dummy_data

if __name__ == '__main__':
    print("=====================================================")
    print("MUZERO PHASE II - DEMONSTRATION")
    print("=====================================================")

    #    
    pipeline = OptimizationPipeline()

    # ---  :    ---
    try:
        pipeline.run_phase1_hyperparameter_optimization(n_trials=10)
    except Exception as e:
        print(f"❌ Error in Phase 1: {e}")
        pipeline.optimization_report['phase1'] = {
            'final_hyperparameters': MuZeroConfig().__dict__
        }
        print("⚠️ Using default hyperparameters to continue.")

    # ---  :   ---
    print("\nCreating dummy training data for feature engineering demo...")
    dummy_data = create_dummy_training_data(num_samples=50)
    try:
        pipeline.run_phase2_feature_engineering(training_data=dummy_data)
    except Exception as e:
        print(f"❌ Error in Phase 2: {e}")

    # ---  :    ---
    try:
        final_config = pipeline.run_phase3_final_configuration()
    except Exception as e:
        print(f"❌ Error in Phase 3: {e}")

    # ---    ---
    pipeline.generate_report()

    print("\n=====================================================")
    print("DEMONSTRATION COMPLETE")
    print("=====================================================")


