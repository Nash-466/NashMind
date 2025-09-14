from __future__ import annotations
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable, NamedTuple
try:
    from collections.abc import Callable as Callable
except Exception:
    pass
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

try:
    from arc_complete_agent_part2 import UltraComprehensivePatternAnalyzer, convert_grid_to_graph
    from arc_complete_agent_part3 import AdvancedStrategyManager
    from arc_complete_agent_part1 import CausalSimulationEngine
    COMPONENTS_AVAILABLE = True
except Exception:
    COMPONENTS_AVAILABLE = False
    class UltraComprehensivePatternAnalyzer: pass
    class AdvancedStrategyManager: pass
    class CausalSimulationEngine: pass
    class CausalSimulationEngine: pass

class Hypothesis(NamedTuple):
    theory_name: str
    rule_name: str
    rule_parameters: Dict[str, Any]
    confidence_score: float
    description: str
    applicator: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]

class Theory(ABC):
    def __init__(self, pattern_analyzer, strategy_manager):
        self.pattern_analyzer = pattern_analyzer
        self.strategy_manager = strategy_manager
        self.name = self.__class__.__name__

    @abstractmethod
    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        ...

    @abstractmethod
    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        ...

class SymmetryTheory(Theory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetry_rules = {
            'identity': lambda g: g,
            'flip_horizontal': np.fliplr,
            'flip_vertical': np.flipud,
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'transpose': np.transpose,
        }

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        evidence_boost = 0.0
        try:
            if self.pattern_analyzer and train_pairs:
                analysis = self.pattern_analyzer.analyze_ultra_comprehensive_patterns(train_pairs[0]['input'])
                evidence_boost = float(getattr(analysis, 'geometric_symmetry', 0.0))
        except Exception:
            evidence_boost = 0.0
        for rule_name, func in self.symmetry_rules.items():
            hypotheses.append(Hypothesis(
                theory_name=self.name,
                rule_name=rule_name,
                rule_parameters={'name': rule_name},
                confidence_score=min(1.0, 0.5 + 0.5*evidence_boost),
                description=f"Apply {rule_name} to full grid",
                applicator=self._apply_symmetry_rule
            ))
        return hypotheses

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        rule_name = hypothesis.rule_parameters['name']
        func = self.symmetry_rules.get(rule_name)
        if func is None:
            return 0.0
        matches = 0
        for p in train_pairs:
            i, o = p['input'], p['output']
            pred = func(i)
            if pred.shape == o.shape and np.array_equal(pred, o):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_symmetry_rule(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        rule_name = params['name']
        rules = {
            'identity': lambda g: g,
            'flip_horizontal': np.fliplr,
            'flip_vertical': np.flipud,
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'transpose': np.transpose,
        }
        if rule_name not in rules:
            return grid
        return rules[rule_name](grid)

class ObjectMovementTheory(Theory):
    """Infer a consistent translation (dx, dy) of foreground pixels between input and output."""

    def _centroid(self, g: np.ndarray) -> Tuple[float, float]:
        pts = np.argwhere(g > 0)
        if pts.size == 0:
            return (0.0, 0.0)
        yx = pts.astype(np.float64)
        cy = float(np.mean(yx[:, 0])); cx = float(np.mean(yx[:, 1]))
        return (cy, cx)

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        shifts: List[Tuple[int, int]] = []
        for p in train_pairs:
            i, o = p['input'], p['output']
            (iy, ix) = self._centroid(i)
            (oy, ox) = self._centroid(o)
            dy = int(round(oy - iy)); dx = int(round(ox - ix))
            shifts.append((dy, dx))
        # consistent shift across pairs
        if not shifts:
            return []
        dy, dx = shifts[0]
        if any((s != (dy, dx)) for s in shifts):
            return []
        return [Hypothesis(
            theory_name=self.name,
            rule_name='translate_fg',
            rule_parameters={'dy': int(dy), 'dx': int(dx)},
            confidence_score=0.65,
            description=f'Translate foreground by (dy={dy}, dx={dx})',
            applicator=self._apply_translate
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            pred = self._apply_translate(p['input'], hypothesis.rule_parameters)
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_translate(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        dy = int(params.get('dy', 0)); dx = int(params.get('dx', 0))
        h, w = grid.shape
        out = np.zeros_like(grid)
        src_y0 = max(0, -dy); src_y1 = min(h, h - dy)
        src_x0 = max(0, -dx); src_x1 = min(w, w - dx)
        dst_y0 = max(0, dy); dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx); dst_x1 = dst_x0 + (src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            src = grid[src_y0:src_y1, src_x0:src_x1]
            out[dst_y0:dst_y1, dst_x0:dst_x1] = np.where(src > 0, src, out[dst_y0:dst_y1, dst_x0:dst_x1])
        return out

class ObjectCentricTheory(Theory):
    """Reason on connected components (object-level) with simple alignment rules."""

    @staticmethod
    def _components(mask: np.ndarray):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []
        for i in range(h):
            for j in range(w):
                if mask[i,j] and not visited[i,j]:
                    q = [(i,j)]; visited[i,j] = True; cells = []
                    while q:
                        y,x = q.pop(0); cells.append((y,x))
                        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            ny, nx = y+dy, x+dx
                            if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not visited[ny,nx]:
                                visited[ny,nx] = True; q.append((ny,nx))
                    comps.append(cells)
        return comps

    @staticmethod
    def _centroid_coords(cells):
        if not cells:
            return (0.0, 0.0)
        ys = [y for y,_ in cells]; xs = [x for _,x in cells]
        return (float(np.mean(ys)), float(np.mean(xs)))

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        shifts_centroid = []
        align_topleft_consistent = True
        for p in train_pairs:
            i = p['input']; o = p['output']
            im = (i>0).astype(np.uint8); om = (o>0).astype(np.uint8)
            # centroid shift
            ci = self._centroid_coords([c for comp in self._components(im) for c in comp])
            co = self._centroid_coords([c for comp in self._components(om) for c in comp])
            dy = int(round(co[0]-ci[0])); dx = int(round(co[1]-ci[1]))
            shifts_centroid.append((dy,dx))
            # topleft align check
            if om.any():
                comps_o = self._components(om)
                ys = [y for comp in comps_o for y,_ in comp]
                xs = [x for comp in comps_o for _,x in comp]
                yo = min(ys) if ys else 0; xo = min(xs) if xs else 0
            else:
                yo, xo = (0,0)
            if not (yo==0 and xo==0):
                align_topleft_consistent = False
        hyps: List[Hypothesis] = []
        if shifts_centroid:
            dy, dx = shifts_centroid[0]
            if all(s==(dy,dx) for s in shifts_centroid):
                hyps.append(Hypothesis(
                    theory_name=self.name,
                    rule_name='align_centroid',
                    rule_parameters={'dy': int(dy), 'dx': int(dx)},
                    confidence_score=0.6,
                    description=f'Align overall centroid by (dy={dy},dx={dx})',
                    applicator=self._apply_align_centroid
                ))
        if align_topleft_consistent:
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='align_topleft',
                rule_parameters={},
                confidence_score=0.55,
                description='Move foreground to top-left corner',
                applicator=self._apply_align_topleft
            ))
        return hyps

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            pred = None
            if hypothesis.rule_name=='align_centroid':
                pred = self._apply_align_centroid(p['input'], hypothesis.rule_parameters)
            elif hypothesis.rule_name=='align_topleft':
                pred = self._apply_align_topleft(p['input'], {})
            if pred is not None and pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_align_centroid(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        dy = int(params.get('dy',0)); dx = int(params.get('dx',0))
        return ObjectMovementTheory._apply_translate(grid, {'dy':dy,'dx':dx})

    @staticmethod
    def _apply_align_topleft(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        g = grid.copy(); mask = (g>0)
        if not mask.any():
            return g
        ys, xs = np.where(mask)
        miny, minx = int(ys.min()), int(xs.min())
        return ObjectMovementTheory._apply_translate(g, {'dy': -miny, 'dx': -minx})

class FillBetweenTheory(Theory):
    """Fill between first and last nonzero per row/column with the row/col color when unique."""

    @staticmethod
    def _fill_between_rows(inp: np.ndarray) -> np.ndarray:
        out = np.zeros_like(inp)
        h, w = inp.shape
        for i in range(h):
            row = inp[i, :]
            nz = np.where(row != 0)[0]
            if nz.size == 0:
                continue
            colors = np.unique(row[row != 0])
            if colors.size != 1:
                continue
            c = int(colors[0])
            out[i, nz.min():nz.max()+1] = c
        return out

    @staticmethod
    def _fill_between_cols(inp: np.ndarray) -> np.ndarray:
        out = np.zeros_like(inp)
        h, w = inp.shape
        for j in range(w):
            col = inp[:, j]
            nz = np.where(col != 0)[0]
            if nz.size == 0:
                continue
            colors = np.unique(col[col != 0])
            if colors.size != 1:
                continue
            c = int(colors[0])
            out[nz.min():nz.max()+1, j] = c
        return out

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        # Check rows
        ok_rows = True
        for p in train_pairs:
            pred = self._fill_between_rows(p['input'])
            if pred.shape != p['output'].shape or not np.array_equal(pred, p['output']):
                ok_rows = False
                break
        if ok_rows:
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='fill_between_rows',
                rule_parameters={},
                confidence_score=0.7,
                description='Fill between first/last nonzero per row with unique row color',
                applicator=lambda g, _: self._fill_between_rows(g)
            ))
        # Check cols
        ok_cols = True
        for p in train_pairs:
            pred = self._fill_between_cols(p['input'])
            if pred.shape != p['output'].shape or not np.array_equal(pred, p['output']):
                ok_cols = False
                break
        if ok_cols:
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='fill_between_cols',
                rule_parameters={},
                confidence_score=0.7,
                description='Fill between first/last nonzero per column with unique column color',
                applicator=lambda g, _: self._fill_between_cols(g)
            ))
        return hyps

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            if hypothesis.rule_name == 'fill_between_rows':
                pred = self._fill_between_rows(p['input'])
            elif hypothesis.rule_name == 'fill_between_cols':
                pred = self._fill_between_cols(p['input'])
            else:
                return 0.0
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

class RowColumnShiftTheory(Theory):
    """Infer consistent cyclic row/column shifts between input and output."""

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        if not train_pairs:
            return []
        dy_dx = self._infer_shift(train_pairs)
        if dy_dx is None:
            return []
        dy, dx = dy_dx
        return [Hypothesis(
            theory_name=self.name,
            rule_name='cyclic_shift',
            rule_parameters={'dy': int(dy), 'dx': int(dx)},
            confidence_score=0.7,
            description=f'Cyclic shift rows by {dy}, cols by {dx}',
            applicator=self._apply_shift
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        dy = int(hypothesis.rule_parameters.get('dy', 0))
        dx = int(hypothesis.rule_parameters.get('dx', 0))
        matches = 0
        for p in train_pairs:
            pred = self._apply_shift(p['input'], {'dy': dy, 'dx': dx})
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_shift(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        dy = int(params.get('dy', 0)); dx = int(params.get('dx', 0))
        return np.roll(np.roll(grid, dy, axis=0), dx, axis=1)

    @staticmethod
    def _infer_shift(train_pairs: List[Dict[str, np.ndarray]]):
        dy_dx = None
        for p in train_pairs:
            i, o = p['input'], p['output']
            if i.shape != o.shape:
                return None
            h, w = i.shape
            found = False
            for dy in range(-h+1, h):
                for dx in range(-w+1, w):
                    if np.array_equal(np.roll(np.roll(i, dy, axis=0), dx, axis=1), o):
                        if dy_dx is None:
                            dy_dx = (dy, dx)
                            found = True
                            break
                        else:
                            if dy_dx == (dy, dx):
                                found = True
                                break
                if found:
                    break
            if not found:
                return None
        return dy_dx

class BorderFillTheory(Theory):
    """Fill the border (frame) with a uniform color inferred from training pairs."""

    @staticmethod
    def _fill_border(grid: np.ndarray, color: int) -> np.ndarray:
        g = grid.copy()
        if g.size == 0:
            return g
        g[0, :] = color
        g[-1, :] = color
        g[:, 0] = color
        g[:, -1] = color
        return g

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        color = None
        for p in train_pairs:
            i, o = p['input'], p['output']
            if i.shape != o.shape:
                return []
            # infer color as the dominant border color in output
            border_vals = np.concatenate([o[0, :], o[-1, :], o[:, 0], o[:, -1]])
            uniq, cnt = np.unique(border_vals, return_counts=True)
            c = int(uniq[np.argmax(cnt)]) if uniq.size else 0
            # check interior unchanged
            if i.shape[0] > 2 and i.shape[1] > 2:
                if not np.array_equal(i[1:-1, 1:-1], o[1:-1, 1:-1]):
                    return []
            # check the border fill reproduces output
            if color is None:
                color = c
            elif color != c:
                return []
            test = self._fill_border(i, color)
            if not np.array_equal(test, o):
                return []
        if color is None:
            return []
        return [Hypothesis(
            theory_name=self.name,
            rule_name='border_fill',
            rule_parameters={'color': int(color)},
            confidence_score=0.7,
            description=f'Fill border with color {color}',
            applicator=lambda g, params: self._fill_border(g, int(params.get('color', 0)))
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        color = int(hypothesis.rule_parameters.get('color', 0))
        matches = 0
        for p in train_pairs:
            i, o = p['input'], p['output']
            pred = self._fill_border(i, color)
            if pred.shape == o.shape and np.array_equal(pred, o):
                matches += 1
        return matches / max(1, len(train_pairs))

class GraphReasoningTheory(Theory):
    """Reason over simple graph abstractions: keep/remove single color globally."""

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        colors_in = set(); colors_out = set()
        for p in train_pairs:
            colors_in |= set(np.unique(p['input']).tolist())
            colors_out |= set(np.unique(p['output']).tolist())
        remove_candidates = list(colors_in - colors_out)
        keep_candidates = list(colors_out)
        hyps: List[Hypothesis] = []
        for c in remove_candidates:
            if c == 0: continue
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='remove_color',
                rule_parameters={'type': 'remove_color', 'color': int(c)},
                confidence_score=0.7,
                description=f'Remove all pixels of color {c}',
                applicator=self._apply_graph_rule
            ))
        for c in keep_candidates:
            if c == 0: continue
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='keep_color',
                rule_parameters={'type': 'keep_color', 'color': int(c)},
                confidence_score=0.6,
                description=f'Keep only pixels of color {c}',
                applicator=self._apply_graph_rule
            ))
        return hyps

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            pred = self._apply_graph_rule(p['input'], hypothesis.rule_parameters)
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_graph_rule(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        t = params.get('type'); c = int(params.get('color', -1))
        g = grid.copy()
        if t == 'remove_color' and c >= 0:
            g = np.where(g == c, 0, g)
        elif t == 'keep_color' and c >= 0:
            g = np.where(g == c, g, 0)
        return g

class CausalInferenceTheory(Theory):
    """Delegate to CausalSimulationEngine to find a simple law explaining pairs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = CausalSimulationEngine() if CausalSimulationEngine else None

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        if not self.engine:
            return []
        law = self.engine.find_best_law(train_pairs)
        if not law or not law.get('callable'):
            return []
        name = law.get('name') or 'causal_law'
        return [Hypothesis(
            theory_name=self.name,
            rule_name=name,
            rule_parameters={'law': name},
            confidence_score=0.65,
            description=f'Apply causal law {name}',
            applicator=self._apply_causal
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        if not self.engine:
            return 0.0
        func = getattr(self.engine, hypothesis.rule_name, None)
        if func is None:
            return 0.0
        matches = 0
        for p in train_pairs:
            pred = func(p['input'])
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    def _apply_causal(self, grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not self.engine:
            return grid
        name = params.get('law'); func = getattr(self.engine, str(name), None)
        return func(grid) if func else grid

class ColorMappingTheory(Theory):
    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        mapping: Dict[int, int] = {}
        for p in train_pairs:
            i, o = p['input'], p['output']
            if i.shape != o.shape:
                return []
            for a,b in zip(i.flatten(), o.flatten()):
                a = int(a); b = int(b)
                if a in mapping and mapping[a] != b:
                    return []
                mapping[a] = b
        if not mapping:
            return []
        return [Hypothesis(
            theory_name=self.name,
            rule_name='color_map',
            rule_parameters={'mapping': mapping},
            confidence_score=0.7,
            description='Apply learned color mapping',
            applicator=self._apply_color_map
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        m = hypothesis.rule_parameters.get('mapping', {})
        if not m:
            return 0.0
        matches = 0
        for p in train_pairs:
            if p['input'].shape != p['output'].shape:
                return 0.0
            pred = self._apply_color_map(p['input'], hypothesis.rule_parameters)
            if np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_color_map(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        mapping: Dict[int,int] = params.get('mapping', {})
        g = grid.copy(); out = g.copy()
        for src,dst in mapping.items():
            out[g==int(src)] = int(dst)
        return out

class TilingAndScalingTheory(Theory):
    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        modes = []; params = []
        for p in train_pairs:
            ih, iw = p['input'].shape; oh, ow = p['output'].shape
            if ih==0 or iw==0:
                return []
            if oh % ih == 0 and ow % iw == 0:
                sy, sx = oh//ih, ow//iw
                if self._is_pixel_scale(p['input'], p['output'], sy, sx):
                    modes.append('pixel_scale'); params.append({'sy': sy, 'sx': sx})
                elif self._is_tiling(p['input'], p['output']):
                    modes.append('tile'); params.append({})
                else:
                    return []
            else:
                return []
        if not modes:
            return []
        if all(m=='pixel_scale' for m in modes):
            sy = params[0]['sy']; sx = params[0]['sx']
            return [Hypothesis(
                theory_name=self.name,
                rule_name='pixel_scale',
                rule_parameters={'sy': int(sy), 'sx': int(sx)},
                confidence_score=0.8,
                description=f'Pixel scale by ({sy},{sx})',
                applicator=self._apply_scale
            )]
        if all(m=='tile' for m in modes):
            return [Hypothesis(
                theory_name=self.name,
                rule_name='tile_fill',
                rule_parameters={},
                confidence_score=0.75,
                description='Tile input to fill target size',
                applicator=self._apply_tile
            )]
        return []

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            i, o = p['input'], p['output']
            if hypothesis.rule_name == 'pixel_scale':
                pred = self._apply_scale(i, hypothesis.rule_parameters)
            elif hypothesis.rule_name == 'tile_fill':
                pred = self._apply_tile(i, {'target_shape': o.shape})
            else:
                return 0.0
            if pred.shape == o.shape and np.array_equal(pred, o):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _is_pixel_scale(inp: np.ndarray, out: np.ndarray, sy: int, sx: int) -> bool:
        block = np.ones((sy, sx), dtype=inp.dtype)
        try:
            pred = np.kron(inp, block)
            return pred.shape == out.shape and np.array_equal(pred, out)
        except Exception:
            return False

    @staticmethod
    def _is_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
        ih, iw = inp.shape; oh, ow = out.shape
        if oh % ih != 0 or ow % iw != 0:
            return False
        ry, rx = oh//ih, ow//iw
        tiled = np.tile(inp, (ry, rx))
        return np.array_equal(tiled, out)

    @staticmethod
    def _apply_scale(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        sy = int(params.get('sy', 1)); sx = int(params.get('sx', 1))
        block = np.ones((sy, sx), dtype=grid.dtype)
        return np.kron(grid, block)

    @staticmethod
    def _apply_tile(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        target = params.get('target_shape')
        if not target:
            return grid
        ih, iw = grid.shape; oh, ow = target
        ry, rx = max(1, oh//ih), max(1, ow//iw)
        return np.tile(grid, (ry, rx))[:oh, :ow]

class GraphReasoningTheory(Theory):
    """Reason over simple graph abstractions: keep/remove single color globally."""

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        # infer candidate colors to remove or keep based on color sets difference
        colors_in = set()
        colors_out = set()
        for p in train_pairs:
            colors_in |= set(np.unique(p['input']).tolist())
            colors_out |= set(np.unique(p['output']).tolist())
        remove_candidates = list(colors_in - colors_out)
        keep_candidates = list(colors_out)
        hyps: List[Hypothesis] = []
        for c in remove_candidates:
            if c == 0: continue
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='remove_color',
                rule_parameters={'type': 'remove_color', 'color': int(c)},
                confidence_score=0.7,
                description=f'Remove all pixels of color {c}',
                applicator=self._apply_graph_rule
            ))
        for c in keep_candidates:
            if c == 0: continue
            hyps.append(Hypothesis(
                theory_name=self.name,
                rule_name='keep_color',
                rule_parameters={'type': 'keep_color', 'color': int(c)},
                confidence_score=0.6,
                description=f'Keep only pixels of color {c}',
                applicator=self._apply_graph_rule
            ))
        return hyps

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        matches = 0
        for p in train_pairs:
            pred = self._apply_graph_rule(p['input'], hypothesis.rule_parameters)
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    @staticmethod
    def _apply_graph_rule(grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        t = params.get('type')
        c = int(params.get('color', -1))
        g = grid.copy()
        if t == 'remove_color' and c >= 0:
            g = np.where(g == c, 0, g)
        elif t == 'keep_color' and c >= 0:
            g = np.where(g == c, g, 0)
        return g

class CausalInferenceTheory(Theory):
    """Delegate to CausalSimulationEngine to find a simple law explaining pairs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = CausalSimulationEngine() if CausalSimulationEngine else None

    def propose(self, train_pairs: List[Dict[str, np.ndarray]]) -> List[Hypothesis]:
        if not self.engine:
            return []
        law = self.engine.find_best_law(train_pairs)
        if not law or not law.get('callable'):
            return []
        name = law.get('name') or 'causal_law'
        return [Hypothesis(
            theory_name=self.name,
            rule_name=name,
            rule_parameters={'law': name},
            confidence_score=0.65,
            description=f'Apply causal law {name}',
            applicator=self._apply_causal
        )]

    def validate(self, train_pairs: List[Dict[str, np.ndarray]], hypothesis: Hypothesis) -> float:
        if not self.engine:
            return 0.0
        matches = 0
        func = getattr(self.engine, hypothesis.rule_name, None)
        if func is None:
            return 0.0
        for p in train_pairs:
            pred = func(p['input'])
            if pred.shape == p['output'].shape and np.array_equal(pred, p['output']):
                matches += 1
        return matches / max(1, len(train_pairs))

    def _apply_causal(self, grid: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not self.engine:
            return grid
        name = params.get('law')
        func = getattr(self.engine, str(name), None)
        return func(grid) if func else grid
class MasterOrchestrator:
    def __init__(self, initial_config: Dict = None):
        self.pattern_analyzer = UltraComprehensivePatternAnalyzer() if COMPONENTS_AVAILABLE else None
        self.strategy_manager = AdvancedStrategyManager() if COMPONENTS_AVAILABLE else None
        self.config = {
            'min_validation_score': 0.99,
        }
        if initial_config:
            self.config.update(initial_config)
        self.theory_library: List[Theory] = [
            SymmetryTheory(self.pattern_analyzer, self.strategy_manager),
            ColorMappingTheory(self.pattern_analyzer, self.strategy_manager),
            TilingAndScalingTheory(self.pattern_analyzer, self.strategy_manager),
            BorderFillTheory(self.pattern_analyzer, self.strategy_manager),
            FillBetweenTheory(self.pattern_analyzer, self.strategy_manager),
            ObjectCentricTheory(self.pattern_analyzer, self.strategy_manager),
            ObjectMovementTheory(self.pattern_analyzer, self.strategy_manager),
            GraphReasoningTheory(self.pattern_analyzer, self.strategy_manager),
            CausalInferenceTheory(self.pattern_analyzer, self.strategy_manager)
        ]

    def process_single_task(self, task: Dict) -> Optional[np.ndarray]:
        train_pairs = [{'input': np.array(p['input']), 'output': np.array(p['output'])} for p in task['train']]
        test_input = np.array(task['test'][0]['input'])
        hyps: List[Hypothesis] = []
        for T in self.theory_library:
            try:
                hyps.extend(T.propose(train_pairs))
            except Exception:
                continue
        # Evaluate and select best by combined priority: validate + confidence - cost_penalty
        best = None
        best_priority = -1.0
        for h in sorted(hyps, key=lambda hh: hh.confidence_score, reverse=True):
            try:
                T = next(t for t in self.theory_library if t.name == h.theory_name)
                v = T.validate(train_pairs, h)
                cost = 1.0
                try:
                    if self.strategy_manager and hasattr(self.strategy_manager, 'get_cost'):
                        cost = self.strategy_manager.get_cost(h.rule_name)
                except Exception:
                    cost = 1.0
                priority = 0.7*float(v) + 0.3*float(h.confidence_score) - 0.05*float(cost)
                if v >= self.config['min_validation_score'] and priority > best_priority:
                    best_priority = priority
                    best = h
            except Exception:
                continue
        if best is not None:
            return best.applicator(test_input, best.rule_parameters)

        # Program search: try short sequences of theory applicators (depth 2-3)
        try:
            top_hyps = sorted(hyps, key=lambda hh: hh.confidence_score, reverse=True)[:10]
            for i in range(len(top_hyps)):
                for j in range(len(top_hyps)):
                    seq = [top_hyps[i], top_hyps[j]]
                    ok = True
                    for p in train_pairs:
                        g = p['input']
                        for h_ in seq:
                            g = h_.applicator(g, h_.rule_parameters)
                        if g.shape != p['output'].shape or not np.array_equal(g, p['output']):
                            ok = False; break
                    if ok:
                        g = test_input
                        for h_ in seq:
                            g = h_.applicator(g, h_.rule_parameters)
                        print('[ProgramSearch] depth2:', [seq[0].rule_name, seq[1].rule_name])
                        return g
            for i in range(min(5, len(top_hyps))):
                for j in range(min(5, len(top_hyps))):
                    for k in range(min(5, len(top_hyps))):
                        seq = [top_hyps[i], top_hyps[j], top_hyps[k]]
                        ok = True
                        for p in train_pairs:
                            g = p['input']
                            for h_ in seq:
                                g = h_.applicator(g, h_.rule_parameters)
                            if g.shape != p['output'].shape or not np.array_equal(g, p['output']):
                                ok = False; break
                        if ok:
                            g = test_input
                            for h_ in seq:
                                g = h_.applicator(g, h_.rule_parameters)
                            print('[ProgramSearch] depth3:', [seq[0].rule_name, seq[1].rule_name, seq[2].rule_name])
                            return g
        except Exception:
            pass
            return best.applicator(test_input, best.rule_parameters)
        # Beam search over composed strategies (depth up to 3)
        try:
            if self.strategy_manager:
                base = ['identity','flip_horizontal','flip_vertical','rotate_90','rotate_180','rotate_270','transpose']
                # depth 1
                for s in base:
                    ok = True
                    for p in train_pairs:
                        pred = self.strategy_manager.apply_strategy(s, p['input'], {})
                        if pred.shape != p['output'].shape or not np.array_equal(pred, p['output']):
                            ok = False; break
                    if ok:
                        return self.strategy_manager.apply_strategy(s, test_input, {})
                # depth 2 compositions (beam)
                beam = []
                for a in base:
                    for b in base:
                        name = self.strategy_manager.compose(a, b)
                        score = 0
                        ok_pair = True
                        for p in train_pairs:
                            pred = self.strategy_manager.apply_strategy(name, p['input'], {})
                            if pred.shape != p['output'].shape:
                                ok_pair = False; break
                            score += int(np.array_equal(pred, p['output']))
                        if ok_pair:
                            beam.append((score, name))
                beam.sort(reverse=True, key=lambda x: x[0])
                for _, name in beam[:10]:
                    all_ok = True
                    for p in train_pairs:
                        pred = self.strategy_manager.apply_strategy(name, p['input'], {})
                        if pred.shape != p['output'].shape or not np.array_equal(pred, p['output']):
                            all_ok = False; break
                    if all_ok:
                        return self.strategy_manager.apply_strategy(name, test_input, {})
                # depth 3 compositions (beam)
                beam3 = []
                for a in base:
                    for b in base:
                        ab = self.strategy_manager.compose(a, b)
                        for c in base:
                            abc = self.strategy_manager.compose(ab, c)
                            score = 0
                            ok_pair = True
                            for p in train_pairs:
                                pred = self.strategy_manager.apply_strategy(abc, p['input'], {})
                                if pred.shape != p['output'].shape:
                                    ok_pair = False; break
                                score += int(np.array_equal(pred, p['output']))
                            if ok_pair:
                                beam3.append((score, abc))
                beam3.sort(reverse=True, key=lambda x: x[0])
                for _, name in beam3[:10]:
                    all_ok = True
                    for p in train_pairs:
                        pred = self.strategy_manager.apply_strategy(name, p['input'], {})
                        if pred.shape != p['output'].shape or not np.array_equal(pred, p['output']):
                            all_ok = False; break
                    if all_ok:
                        return self.strategy_manager.apply_strategy(name, test_input, {})
        except Exception:
            pass
        return None



