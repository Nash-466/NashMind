from __future__ import annotations
import json
import os
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from arc_complete_agent_part2 import UltraComprehensivePatternAnalyzer
except Exception:
    UltraComprehensivePatternAnalyzer = None  # type: ignore

try:
    from arc_complete_agent_part1 import CausalSimulationEngine
except Exception:
    CausalSimulationEngine = None  # type: ignore

try:
    from arc_complete_agent_part5 import ARCEnvironment, MuZeroAgent, MuZeroConfig
except Exception:
    ARCEnvironment = None  # type: ignore
    MuZeroAgent = None  # type: ignore
    MuZeroConfig = None  # type: ignore


class KnowledgeBase:
    """       ."""

    def __init__(self, kb_path: str = "_kb/meta_kb.json"):
        self.kb_path = kb_path
        self._ensure_storage()
        self.data = self._load()

    def _ensure_storage(self) -> None:
        kb_dir = os.path.dirname(self.kb_path)
        if kb_dir and not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)
        if not os.path.exists(self.kb_path):
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save(self) -> None:
        try:
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def compute_signature(task: Dict[str, Any]) -> str:
        #  :       
        try:
            shapes = []
            color_counts = []
            for p in task.get('train', []):
                ig = np.array(p['input']); og = np.array(p['output'])
                shapes.append((tuple(ig.shape), tuple(og.shape)))
                color_counts.append((int(np.unique(ig).size), int(np.unique(og).size)))
            test_shape = tuple(np.array(task['test'][0]['input']).shape)
            signature = {
                'shapes': shapes,
                'color_counts': color_counts,
                'test_shape': test_shape,
            }
            return json.dumps(signature, sort_keys=True)
        except Exception:
            return json.dumps({'fallback': True})

    def retrieve(self, signature: str) -> Optional[Dict[str, Any]]:
        return self.data.get(signature)

    def upsert(self, signature: str, record: Dict[str, Any]) -> None:
        self.data[signature] = record
        self.save()


class HypothesisMarket:
    """      ."""

    SIMPLE_COST = {
        'identity': 1.0,
        'flip_horizontal': 1.0,
        'flip_vertical': 1.0,
        'rotate_90': 1.2,
        'rotate_180': 1.2,
        'rotate_270': 1.2,
        'transpose': 1.2,
    }

    def price(self, hypothesis: Any, time_budget: float = 5.0) -> float:
        try:
            name = hypothesis.rule_name
            conf = float(getattr(hypothesis, 'confidence_score', 0.5))
        except Exception:
            name = str(hypothesis)
            conf = 0.5
        cost = self.SIMPLE_COST.get(name, 2.0)
        time_penalty = 1.0 + max(0.0, cost - 1.0) / max(1.0, time_budget)
        return conf / time_penalty

    def select_portfolio(self, hypotheses: List[Any], k: int = 10, time_budget: float = 5.0) -> List[Any]:
        scored = [(self.price(h, time_budget), h) for h in hypotheses]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:k]]


class StrategyComposer:
    """      ."""

    def __init__(self, strategy_manager: Any):
        self.mgr = strategy_manager

    def propose_compositions(self, depth: int = 2) -> List[str]:
        names = list(getattr(self.mgr, 'basic_strategies', {}).keys())
        #     
        seeds = [('rotate_90', 'flip_horizontal'), ('transpose', 'flip_vertical'), ('flip_horizontal', 'flip_vertical')]
        proposals: List[str] = []
        for a, b in seeds:
            if a in names and b in names:
                composed = self.mgr.compose(a, b)
                proposals.append(composed)
        return proposals

    @staticmethod
    def validate_on_pairs(mgr: Any, strat_name: str, train_pairs: List[Dict[str, np.ndarray]]) -> bool:
        for p in train_pairs:
            ig = p['input']; og = p['output']
            pred = mgr.apply_strategy(strat_name, ig, {})
            if pred.shape != og.shape or not np.array_equal(pred, og):
                return False
        return True


class MetaBrain:
    """           ."""

    def __init__(self, kb_path: str = "_kb/meta_kb.json"):
        self.kb = KnowledgeBase(kb_path)
        self.analyzer = UltraComprehensivePatternAnalyzer() if UltraComprehensivePatternAnalyzer else None
        self.causal = CausalSimulationEngine() if CausalSimulationEngine else None
        self.market = HypothesisMarket()

    def suggest_and_solve(self, task: Dict[str, Any], orchestrator: Any, mode: str = 'fast') -> Optional[np.ndarray]:
        signature = self.kb.compute_signature(task)
        test_input = np.array(task['test'][0]['input'])
        train_pairs = [{'input': np.array(p['input']), 'output': np.array(p['output'])} for p in task['train']]

        # 1)   
        kb_hit = self.kb.retrieve(signature)
        if kb_hit:
            out = self._apply_kb_record(kb_hit, orchestrator, test_input)
            if out is not None:
                return out

        # 2)     
        hypotheses: List[Any] = []
        try:
            for theory in getattr(orchestrator, 'theory_library', []):
                try:
                    hs = theory.propose(train_pairs)
                    if hs:
                        hypotheses.extend(hs)
                except Exception:
                    continue
        except Exception:
            pass

        #  
        selected = self.market.select_portfolio(hypotheses, k=10, time_budget=5.0)
        for h in selected:
            try:
                score = h.confidence_score if hasattr(h, 'confidence_score') else 0.5
                #  
                valid = False
                try:
                    tmatch = next((t for t in getattr(orchestrator, 'theory_library', []) if t.__class__.__name__ == h.theory_name), None)
                    if tmatch is not None:
                        valid = tmatch.validate(train_pairs, h) >= 0.99
                except Exception:
                    valid = False
                if valid:
                    #   
                    grid = test_input
                    pred = h.applicator(grid, h.rule_parameters)
                    if pred is not None:
                        self.kb.upsert(signature, {'strategy_name': h.rule_name, 'source': 'market', 'confidence': float(score)})
                        return pred
            except Exception:
                continue

        # 3)    
        if hasattr(orchestrator, 'strategy_manager') and orchestrator.strategy_manager:
            composer = StrategyComposer(orchestrator.strategy_manager)
            for comp_name in composer.propose_compositions(depth=2):
                try:
                    if StrategyComposer.validate_on_pairs(orchestrator.strategy_manager, comp_name, train_pairs):
                        pred = orchestrator.strategy_manager.apply_strategy(comp_name, test_input, {})
                        self.kb.upsert(signature, {'strategy_name': comp_name, 'source': 'composition'})
                        return pred
                except Exception:
                    continue

        # 4)    
        if self.causal is not None:
            try:
                law = self.causal.find_best_law(train_pairs)
                if law['callable'] is not None:
                    pred = law['callable'](test_input)
                    self.kb.upsert(signature, {'causal_law': law['name'], 'source': 'causal'})
                    return pred
            except Exception:
                pass

        # 5) fallback:  orchestrator  
        try:
            pred = orchestrator.process_single_task(task)
            if pred is not None:
                self.kb.upsert(signature, {'cached_grid': np.array(pred).tolist(), 'source': 'orchestrator'})
                return pred
        except Exception:
            pass

        # 6)    (greedy)    
        if mode == 'deep' and ARCEnvironment is not None and hasattr(orchestrator, 'strategy_manager') and orchestrator.strategy_manager is not None:
            try:
                env = ARCEnvironment(orchestrator.strategy_manager, test_input, np.array(task['test'][0].get('output', test_input)))
                obs = env.reset()
                best_grid = obs
                best_reward = -1.0
                #   
                for _ in range(env.max_steps):
                    rewards = []
                    for i, name in enumerate(env.action_names):
                        nxt = orchestrator.strategy_manager.apply_strategy(name, obs, {})
                        r = float(np.mean(nxt == env.target_grid)) if nxt.shape == env.target_grid.shape else 0.0
                        rewards.append((r, i, nxt))
                    rewards.sort(reverse=True, key=lambda x: x[0])
                    if not rewards:
                        break
                    r, idx, nxt = rewards[0]
                    obs = nxt
                    if r > best_reward:
                        best_reward = r
                        best_grid = nxt
                    if r >= 0.999:
                        break
                self.kb.upsert(signature, {'source': 'greedy_deep', 'reward': float(best_reward)})
                return best_grid
            except Exception:
                pass

        return None

    @staticmethod
    def _apply_kb_record(rec: Dict[str, Any], orchestrator: Any, test_input: np.ndarray) -> Optional[np.ndarray]:
        try:
            if 'cached_grid' in rec:
                return np.array(rec['cached_grid'])
            if 'strategy_name' in rec and hasattr(orchestrator, 'strategy_manager') and orchestrator.strategy_manager:
                return orchestrator.strategy_manager.apply_strategy(rec['strategy_name'], test_input, {})
            if 'causal_law' in rec and CausalSimulationEngine is not None:
                engine = CausalSimulationEngine()
                func = getattr(engine, rec['causal_law'], None)
                if func:
                    return func(test_input)
        except Exception:
            return None
        return None


