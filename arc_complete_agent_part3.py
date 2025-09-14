from __future__ import annotations
import numpy as np
from collections.abc import Callable
from typing import Dict, Any, Tuple


class AdvancedStrategyManager:
    """Strategy manager with basic metadata and safe composition."""

    def __init__(self):
        self.basic_strategies: Dict[str, callable] = {
            'identity': lambda g, ctx: g,
            'flip_horizontal': lambda g, ctx: np.fliplr(g),
            'flip_vertical': lambda g, ctx: np.flipud(g),
            'rotate_90': lambda g, ctx: np.rot90(g, 1),
            'rotate_180': lambda g, ctx: np.rot90(g, 2),
            'rotate_270': lambda g, ctx: np.rot90(g, 3),
            'transpose': lambda g, ctx: g.T,
        }
        self.strategy_meta: Dict[str, Dict[str, Any]] = {
            'identity': {'cost': 0.1, 'effects': ['none']},
            'flip_horizontal': {'cost': 0.5, 'effects': ['mirror_x']},
            'flip_vertical': {'cost': 0.5, 'effects': ['mirror_y']},
            'rotate_90': {'cost': 0.7, 'effects': ['rotate']},
            'rotate_180': {'cost': 0.7, 'effects': ['rotate']},
            'rotate_270': {'cost': 0.7, 'effects': ['rotate']},
            'transpose': {'cost': 0.6, 'effects': ['swap_axes']},
        }
        self.strategy_combinations: Dict[str, Tuple[str, str]] = {}

    def apply_strategy(self, strategy_name: str, grid: np.ndarray, context: Dict) -> np.ndarray:
        """Apply strategy by name; returns original grid on failure."""
        try:
            if strategy_name in self.basic_strategies:
                return self.basic_strategies[strategy_name](grid, context)
            return grid
        except Exception:
            return grid

    def get_cost(self, strategy_name: str) -> float:
        meta = self.strategy_meta.get(strategy_name, {})
        return float(meta.get('cost', 1.0))

    def compose(self, strat1: str, strat2: str) -> str:
        """Create composed strategy strat2(strat1(grid)) and register with metadata."""
        name = f"compose({strat1}+{strat2})"

        def _composed(g, ctx):
            g1 = self.apply_strategy(strat1, g, ctx)
            g2 = self.apply_strategy(strat2, g1, ctx)
            return g2

        self.basic_strategies[name] = lambda g, ctx: _composed(g, ctx)
        self.strategy_meta[name] = {'cost': self.get_cost(strat1)+self.get_cost(strat2), 'effects': ['composed']}
        self.strategy_combinations[name] = (strat1, strat2)
        return name

