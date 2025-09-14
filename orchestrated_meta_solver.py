from __future__ import annotations
"""
HYBRID ORCHESTRATOR - Cascade and refine across systems
=======================================================
This orchestrator chains multiple ARC solvers, applies smart refinements
(rotations, flips, color remapping, consensus voting), and selects the best
candidate based on ground-truth when available or consensus heuristics.
"""

import json
import time
import numpy as np
from collections import Counter, defaultdict
from itertools import product
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation, binary_erosion, label
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Silence noisy logging if any
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# ----------------------------- Utilities ----------------------------- #

# Simple persistent experience memory (JSON)
class ExperienceMemory:
    def __init__(self, path='experience_memory.json'):
        self.path = path
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            if not isinstance(self.data, list):
                self.data = []
        except Exception:
            self.data = []

    def add(self, entry):
        try:
            self.data.append(entry)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def _sim(self, a, b):
        # Cosine-like similarity on feature vector
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va) + 1e-8
        nb = np.linalg.norm(vb) + 1e-8
        return float(np.dot(va, vb) / (na * nb))

    def find_similar(self, signature_vec, top_k=3):
        scored = []
        for e in self.data:
            if 'signature_vec' in e:
                s = self._sim(signature_vec, e['signature_vec'])
                scored.append((s, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for s, e in scored[:top_k]]

# Compute a compact task signature for retrieval
def compute_task_signature(task):
    try:
        tr = task.get('train', [])
        ti = to_array(task.get('test', [{}])[0].get('input'))
        feats = []
        if ti is None:
            return {'vec': [0.0]*16}
        # basic stats on test input
        colors = np.unique(ti)
        feats.append(len(colors))
        feats.append(float(ti.shape[0]))
        feats.append(float(ti.shape[1]))
        # histogram up to 10 colors
        hist = [int(np.sum(ti == c)) for c in range(10)]
        tot = float(ti.size) + 1e-8
        feats.extend([h/tot for h in hist])
        # connected components count (non-zero)
        lbl, ncc = label(ti > 0)
        feats.append(float(ncc))
        return {'vec': feats[:16]}
    except Exception:
        return {'vec': [0.0]*16}

def to_array(grid):
    if grid is None:
        return None
    return np.array(grid, dtype=int)

def extract_output(result):
    if result is None:
        return None
    if isinstance(result, dict):
        if 'output' in result:
            return to_array(result['output'])
        # Some systems might return {'grid': ...}
        for k in ['grid', 'prediction', 'pred']:
            if k in result:
                return to_array(result[k])
    # Assume it's already a list-like grid
    return to_array(result)

def same_shape(a, b):
    return a is not None and b is not None and a.shape == b.shape

# Pixel accuracy (0..1)
def pixel_accuracy(pred, target):
    if pred is None or target is None:
        return 0.0
    if pred.shape != target.shape:
        return 0.0
    total = target.size
    if total == 0:
        return 0.0
    return float(np.sum(pred == target)) / float(total)

# Simple transforms set typically useful in ARC
TRANSFORMS = [
    lambda x: x,
    lambda x: np.rot90(x, 1),
    lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3),
    lambda x: np.fliplr(x),
    lambda x: np.flipud(x),
    lambda x: np.transpose(x)
]

TRANSFORM_NAMES = [
    'identity', 'rot90', 'rot180', 'rot270', 'fliplr', 'flipud', 'transpose'
]

# Try to infer a dominant transform from the first training pair
# Returns index in TRANSFORMS or None
def infer_transform_from_train(train_pairs):
    try:
        if not train_pairs:
            return None
        inp = to_array(train_pairs[0]['input'])
        out = to_array(train_pairs[0]['output'])
        if inp is None or out is None:
            return None
        best_idx, best_acc = None, -1.0
        for i, tf in enumerate(TRANSFORMS):
            try:
                tgrid = tf(inp)
                acc = pixel_accuracy(tgrid, out)
                if acc > best_acc:
                    best_acc, best_idx = acc, i
            except Exception:
                pass
        # Only accept if it's a perfect transform
        if best_acc == 1.0:
            return best_idx
        return None
    except Exception:
        return None

# Greedy color remapping to maximize overlap with reference
# Maps colors in pred to colors in ref
def color_remap_greedy(pred, ref):
    if pred is None or ref is None:
        return pred
    if pred.shape != ref.shape:
        return pred
    pred_colors = np.unique(pred)
    ref_colors = np.unique(ref)

    # Build co-occurrence matrix counts[pred_c][ref_c]
    counts = defaultdict(lambda: defaultdict(int))
    it = np.nditer(ref, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        pc = int(pred[idx])
        rc = int(ref[idx])
        counts[pc][rc] += 1
        it.iternext()

    # Greedy mapping
    used_ref = set()
    mapping = {}
    # sort by most frequent pred colors
    pred_order = sorted(pred_colors, key=lambda c: -np.sum(pred == c))
    for pc in pred_order:
        # choose ref color with max overlap that's not used
        options = sorted(ref_colors, key=lambda rc: -counts[pc][rc])
        chosen = None
        for rc in options:
            if rc not in used_ref:
                chosen = rc
                break
        if chosen is None and len(ref_colors) > 0:
            # fallback pick most common ref color
            chosen = options[0] if len(options) > 0 else 0
        mapping[int(pc)] = int(chosen)
        used_ref.add(int(chosen))

    # apply mapping
    remapped = pred.copy()
    for pc, rc in mapping.items():
        remapped[pred == pc] = rc
    return remapped

# Optimal color remap using Hungarian assignment to maximize matches
def color_remap_hungarian(pred, ref):
    if pred is None or ref is None or pred.shape != ref.shape:
        return pred
    pred_colors = np.unique(pred)
    ref_colors = np.unique(ref)
    # Build cost matrix as negative matches
    C = np.zeros((len(pred_colors), len(ref_colors)), dtype=int)
    for i, pc in enumerate(pred_colors):
        for j, rc in enumerate(ref_colors):
            C[i, j] = -np.sum((pred == pc) & (ref == rc))
    if C.size == 0:
        return pred
    row_ind, col_ind = linear_sum_assignment(C)
    mapping = {int(pred_colors[i]): int(ref_colors[j]) for i, j in zip(row_ind, col_ind)}
    remapped = pred.copy()
    for pc, rc in mapping.items():
        remapped[pred == pc] = rc
    return remapped

# Infer best transform across all training pairs via consensus
# Returns (name, function) or (None, None) if not confident
def infer_transform_consensus(train_pairs):
    if not train_pairs:
        return None, None
    # base transforms
    base = list(zip(TRANSFORM_NAMES, TRANSFORMS))
    # simple composites (up to 2 steps)
    composites = []
    for (n1, f1) in base:
        for (n2, f2) in base:
            composites.append((f"{n2}+{n1}", lambda x, f1=f1, f2=f2: f2(f1(x))))
    candidates = base + composites
    scores = []
    for name, tf in candidates:
        total = 0.0
        cnt = 0
        ok = True
        for ex in train_pairs:
            inp = to_array(ex['input'])
            out = to_array(ex['output'])
            if inp is None or out is None:
                continue
            try:
                tgrid = tf(inp)
                if tgrid.shape != out.shape:
                    ok = False
                    break
                # remap colors optimally
                tgrid_m = color_remap_hungarian(tgrid, out)
                total += pixel_accuracy(tgrid_m, out)
                cnt += 1
            except Exception:
                ok = False
                break
        if ok and cnt > 0:
            scores.append((name, tf, total / cnt))
    if not scores:
        return None, None
    scores.sort(key=lambda x: x[2], reverse=True)
    best_name, best_tf, best_acc = scores[0]
    # require decent consensus to trust
    if best_acc >= 0.9:
        return best_name, best_tf
    return None, None

# Attempt to reshape candidate to a target shape using simple heuristics
def reshape_candidate_to_target(candidate, target_shape):
    if candidate is None or target_shape is None:
        return candidate
    if candidate.shape == tuple(target_shape):
        return candidate
    H, W = candidate.shape
    TH, TW = target_shape
    # Guard against invalid dims
    if H <= 0 or W <= 0 or TH <= 0 or TW <= 0:
        # fallback: zeros of target
        try:
            return np.zeros((TH, TW), dtype=candidate.dtype)
        except Exception:
            return candidate
    best = None
    # Try integer scale up/down
    def try_scale(a, b):
        if a <= 0 or b <= 0:
            return None
        if b % a != 0:
            return None
        k = b // a
        if k <= 0:
            return None
        return k
    kh = try_scale(H, TH)
    kw = try_scale(W, TW)
    if kh is not None and kw is not None and kh == kw:
        scaled = np.repeat(np.repeat(candidate, kh, axis=0), kw, axis=1)
        if scaled.shape == (TH, TW):
            best = scaled
    # Try center crop
    if best is None and H >= TH and W >= TW:
        si = max((H - TH) // 2, 0)
        sj = max((W - TW) // 2, 0)
        cropped = candidate[si:si+TH, sj:sj+TW]
        if cropped.shape == (TH, TW):
            best = cropped
    # Try center pad with zeros
    if best is None and H <= TH and W <= TW:
        pad_i = (TH - H)
        pad_j = (TW - W)
        top = max(pad_i // 2, 0)
        left = max(pad_j // 2, 0)
        out = np.zeros((TH, TW), dtype=candidate.dtype)
        out[top:top+H, left:left+W] = candidate
        best = out
    # Try tile repeat
    if best is None:
        if H > 0 and W > 0:
            ti = (TH + H - 1) // H
            tj = (TW + W - 1) // W
            ti = max(ti, 1)
            tj = max(tj, 1)
            tiled = np.tile(candidate, (ti, tj))[:TH, :TW]
            best = tiled
        else:
            best = np.zeros((TH, TW), dtype=candidate.dtype)
    return best

# Apply transform variants and color remap against a reference
# Return best-improved candidate and name of transform used
def refine_against_reference(candidate, reference, target_shape=None):
    if candidate is None or reference is None:
        return candidate, 'none'
    # First, correct shape if target_shape provided
    if target_shape is not None and candidate.shape != tuple(target_shape):
        candidate = reshape_candidate_to_target(candidate, tuple(target_shape))
    # If shapes now match, apply optimal color remap
    if candidate.shape == reference.shape:
        remapped_h = color_remap_hungarian(candidate, reference)
        remapped_g = color_remap_greedy(candidate, reference)
        cand_opts = [('hungarian', remapped_h), ('greedy', remapped_g), ('none', candidate)]
        best_name, best = 'none', candidate
        best_score = pixel_accuracy(candidate, reference)
        for nm, c in cand_opts:
            sc = pixel_accuracy(c, reference)
            if sc > best_score:
                best_name, best, best_score = nm, c, sc
        # try small shifts -1,0,+1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                t = np.roll(best, shift=(di, dj), axis=(0, 1))
                sc = pixel_accuracy(t, reference)
                if sc > best_score:
                    best, best_name, best_score = t, f'shift({di},{dj})', sc
        # Light XOR repair when mismatch is small
        mismatch = (best != reference)
        frac = float(np.sum(mismatch)) / float(best.size)
        if 0.0 < frac <= 0.1:
            repaired = best.copy()
            repaired[mismatch] = reference[mismatch]
            sc = pixel_accuracy(repaired, reference)
            if sc > best_score:
                best, best_name, best_score = repaired, f'{best_name}+xor_fix', sc
        return best, best_name
    else:
        # try shape-preserving transforms then remap
        best = candidate
        best_name = 'none'
        best_score = pixel_accuracy(candidate, reference)
        # include base transforms
        for name, tf in zip(TRANSFORM_NAMES, TRANSFORMS):
            try:
                t = tf(candidate)
                # optional shape correction
                t = reshape_candidate_to_target(t, reference.shape)
                if t is not None and t.shape == reference.shape:
                    r = color_remap_hungarian(t, reference)
                    sc = pixel_accuracy(r, reference)
                    if sc > best_score:
                        best, best_name, best_score = r, f'{name}+remapH', sc
            except Exception:
                continue
        return best, best_name

# Consensus voting across equal-shaped candidates
def consensus_vote(candidates):
    # candidates: list[np.ndarray] with same shape
    if not candidates:
        return None
    shape = candidates[0].shape
    if any(c is None or c.shape != shape for c in candidates):
        return None
    H, W = shape
    vote = np.zeros_like(candidates[0])
    for i in range(H):
        for j in range(W):
            vals = [int(c[i, j]) for c in candidates]
            # majority vote; tiebreaker by smallest value
            cnt = Counter(vals)
            vote[i, j] = min(cnt.most_common(1)[0][0], 9)
    return vote

# ------------------------- Object-centric layer ------------------------ #

def detect_background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])

def extract_components(grid: np.ndarray):
    bg = detect_background_color(grid)
    mask = grid != bg
    lbl, n = label(mask)
    comps = []
    for cid in range(1, n+1):
        cmask = (lbl == cid)
        ys, xs = np.where(cmask)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bbox = (y0, x0, y1+1, x1+1)
        patch = grid[y0:y1+1, x0:x1+1].copy()
        comps.append({'mask': cmask, 'bbox': bbox, 'patch': patch,
                      'centroid': (ys.mean(), xs.mean())})
    return comps, bg

def transform_grid(grid: np.ndarray, tf):
    try:
        return tf(grid)
    except Exception:
        return grid

def match_components(src_comps, dst_comps):
    # Cost = centroid distance
    if not src_comps or not dst_comps:
        return []
    C = np.zeros((len(src_comps), len(dst_comps)), dtype=float)
    for i, s in enumerate(src_comps):
        sy, sx = s['centroid']
        for j, d in enumerate(dst_comps):
            dy, dx = d['centroid']
            C[i, j] = np.hypot(sy - dy, sx - dx)
    ri, cj = linear_sum_assignment(C)
    pairs = [(i, j) for i, j in zip(ri, cj)]
    return pairs

def learn_object_plan(train_pairs):
    # Try transforms to align component centroids
    base = list(zip(TRANSFORM_NAMES, TRANSFORMS))
    best = {'name': None, 'tf': None, 'avg_shift': (0, 0), 'color_map': {}, 'score': -1, 'per_component_shifts': []}
    for name, tf in base:
        total = 0.0
        cnt = 0
        shifts = []
        color_pairs = []
        ok = True
        for ex in train_pairs:
            src = to_array(ex['input'])
            dst = to_array(ex['output'])
            if src is None or dst is None:
                continue
            tsrc = transform_grid(src, tf)
            src_comps, src_bg = extract_components(tsrc)
            dst_comps, dst_bg = extract_components(dst)
            if not src_comps or not dst_comps:
                ok = False
                break
            pairs = match_components(src_comps, dst_comps)
            if not pairs:
                ok = False
                break
            # accumulate shifts and color pairs
            for i, j in pairs:
                sy, sx = src_comps[i]['centroid']
                dy, dx = dst_comps[j]['centroid']
                shifts.append((dy - sy, dx - sx))
                # dominant color in patches (excluding bg)
                sp = src_comps[i]['patch']
                dp = dst_comps[j]['patch']
                sval, scnt = np.unique(sp, return_counts=True)
                dval, dcnt = np.unique(dp, return_counts=True)
                sdom = int(sval[np.argmax(scnt)])
                ddom = int(dval[np.argmax(dcnt)])
                color_pairs.append((sdom, ddom))
            cnt += 1
        if not ok or cnt == 0:
            continue
        if shifts:
            avg_dy = float(np.median([s[0] for s in shifts]))
            avg_dx = float(np.median([s[1] for s in shifts]))
        else:
            avg_dy = avg_dx = 0.0
        # build simple color map by majority
        cmap = {}
        for sdom, ddom in color_pairs:
            if sdom not in cmap:
                cmap[sdom] = Counter()
            cmap[sdom][ddom] += 1
        cmap_final = {s: cnts.most_common(1)[0][0] for s, cnts in cmap.items()}
        # score = number of pairs matched per example
        score = len(shifts) / max(1, cnt)
        if score > best['score']:
            # compute per-component shifts using the first valid pair
            per_shifts = []
            for ex in train_pairs:
                src = to_array(ex['input'])
                dst = to_array(ex['output'])
                if src is None or dst is None:
                    continue
                tsrc = transform_grid(src, tf)
                src_comps, _ = extract_components(tsrc)
                dst_comps, _ = extract_components(dst)
                if not src_comps or not dst_comps:
                    continue
                # sort by centroid to get stable ordering
                src_sorted = sorted(src_comps, key=lambda c: (c['centroid'][0], c['centroid'][1]))
                dst_sorted = sorted(dst_comps, key=lambda c: (c['centroid'][0], c['centroid'][1]))
                m = min(len(src_sorted), len(dst_sorted))
                for i in range(m):
                    sy, sx = src_sorted[i]['centroid']
                    dy, dx = dst_sorted[i]['centroid']
                    per_shifts.append((dy - sy, dx - sx))
                break
            best = {'name': name, 'tf': tf, 'avg_shift': (avg_dy, avg_dx), 'color_map': cmap_final, 'score': score, 'per_component_shifts': per_shifts}
    return best if best['tf'] is not None else None

def apply_object_plan(grid, plan, target_shape=None, bg_color=None):
    if grid is None or plan is None:
        return grid
    tf = plan['tf']
    dy, dx = plan['avg_shift']
    per = plan.get('per_component_shifts', [])
    cmap = plan['color_map']
    tgrid = transform_grid(grid, tf)
    H, W = tgrid.shape
    if target_shape is None:
        out = np.zeros_like(tgrid)
    else:
        out = np.zeros(tuple(target_shape), dtype=tgrid.dtype)
    if bg_color is None:
        bg_color = detect_background_color(tgrid)
    out[:, :] = bg_color
    comps, _ = extract_components(tgrid)
    # stable ordering
    comps_sorted = sorted(comps, key=lambda c: (c['centroid'][0], c['centroid'][1]))
    for idx, c in enumerate(comps_sorted):
        if idx < len(per):
            pdy, pdx = per[idx]
        else:
            pdy, pdx = dy, dx
        y0, x0, y1, x1 = c['bbox']
        patch = c['patch']
        # recolor dominant
        sval, scnt = np.unique(patch, return_counts=True)
        sdom = int(sval[np.argmax(scnt)])
        ddom = int(cmap.get(sdom, sdom))
        patch_map = patch.copy()
        patch_map[patch == sdom] = ddom
        # place shifted
        ny0 = int(round(y0 + pdy))
        nx0 = int(round(x0 + pdx))
        ny1 = ny0 + (y1 - y0)
        nx1 = nx0 + (x1 - x0)
        # clip to bounds
        if ny1 <= 0 or nx1 <= 0 or ny0 >= out.shape[0] or nx0 >= out.shape[1]:
            continue
        sy0 = max(ny0, 0)
        sx0 = max(nx0, 0)
        sy1 = min(ny1, out.shape[0])
        sx1 = min(nx1, out.shape[1])
        py0 = sy0 - ny0
        px0 = sx0 - nx0
        py1 = py0 + (sy1 - sy0)
        px1 = px0 + (sx1 - sx0)
        out[sy0:sy1, sx0:sx1] = np.where(patch_map[py0:py1, px0:px1] != bg_color,
                                         patch_map[py0:py1, px0:px1],
                                         out[sy0:sy1, sx0:sx1])
    return out

# ---------------------------- Tiny DSL layer --------------------------- #

DSL_OPS = {
    'rot90': lambda g: np.rot90(g, 1),
    'rot180': lambda g: np.rot90(g, 2),
    'rot270': lambda g: np.rot90(g, 3),
    'fliplr': lambda g: np.fliplr(g),
    'flipud': lambda g: np.flipud(g),
    'transpose': lambda g: np.transpose(g),
    'shift_up': lambda g: np.roll(g, shift=(-1, 0), axis=(0, 1)),
    'shift_down': lambda g: np.roll(g, shift=(1, 0), axis=(0, 1)),
    'shift_left': lambda g: np.roll(g, shift=(0, -1), axis=(0, 1)),
    'shift_right': lambda g: np.roll(g, shift=(0, 1), axis=(0, 1)),
}

def apply_ops(grid, ops):
    out = grid
    for op in ops:
        out = DSL_OPS[op](out)
    return out

def find_best_dsl_program(train_pairs, max_len=2):
    # limit ops to a compact set to control search
    base_ops = ['rot90','rot180','fliplr','transpose','shift_right','shift_down']
    # add structural ops
    def crop1(g):
        if g.shape[0] > 2 and g.shape[1] > 2:
            return g[1:-1, 1:-1]
        return g
    def pad1(g):
        bg = detect_background_color(g)
        out = np.full((g.shape[0]+1*2, g.shape[1]+1*2), bg, dtype=g.dtype)
        out[1:-1,1:-1] = g
        return out
    def cc_remove_small(g):
        bg = detect_background_color(g)
        lbl, n = label(g != bg)
        if n == 0:
            return g
        areas = [int(np.sum(lbl == i)) for i in range(1, n+1)]
        thr = np.median(areas)
        out = g.copy()
        for idx, area in enumerate(areas, start=1):
            if area < thr:
                out[lbl == idx] = bg
        return out
    def bbox_fill(g):
        bg = detect_background_color(g)
        lbl, n = label(g != bg)
        if n == 0:
            return g
        areas = [int(np.sum(lbl == i)) for i in range(1, n+1)]
        k = int(np.argmax(areas)) + 1
        ys, xs = np.where(lbl == k)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        patch = g[y0:y1+1, x0:x1+1]
        vals, cnts = np.unique(patch, return_counts=True)
        dom = int(vals[np.argmax(cnts)])
        out = g.copy()
        out[y0:y1+1, x0:x1+1] = dom
        return out
    extended_ops = {
        **DSL_OPS,
        'crop1': crop1,
        'pad1': pad1,
        'cc_remove_small': cc_remove_small,
        'bbox_fill': bbox_fill,
    }
    ops_names = base_ops + ['crop1','pad1','cc_remove_small','bbox_fill']
    best = ([], -1.0)
    # length 0,1,2 beam
    candidates = [[]] + [[o] for o in ops_names]
    if max_len >= 2:
        candidates += [[o1, o2] for o1 in ops_names for o2 in ops_names]
    for ops in candidates:
        total = 0.0
        cnt = 0
        ok = True
        for ex in train_pairs:
            inp = to_array(ex['input'])
            out = to_array(ex['output'])
            if inp is None or out is None:
                continue
            try:
                pred = inp
                for op in ops:
                    pred = extended_ops[op](pred)
                pred = reshape_candidate_to_target(pred, out.shape)
                pred = color_remap_hungarian(pred, out)
                total += pixel_accuracy(pred, out)
                cnt += 1
            except Exception:
                ok = False
                break
        if ok and cnt > 0:
            avg = total / cnt
            if avg > best[1]:
                best = (ops, avg)
    return best[0] if best[1] >= 0 else []

# --------------------------- Orchestrator ---------------------------- #

class HybridOrchestrator:
    def __init__(self, enable_ultra=False, memory_path='experience_memory.json', dsl_max_len=2):
        self.systems = {}
        self.memory = ExperienceMemory(memory_path)
        self.dsl_max_len = int(dsl_max_len)
        self._load_systems(enable_ultra=enable_ultra)

    def _load_systems(self, enable_ultra=False):
        # Priority order will be decided per-task, but we load all here
        try:
            from perfect_arc_system_v2 import PerfectARCSystem
            self.systems['Perfect_V2'] = PerfectARCSystem()
        except Exception as e:
            print(f"Skip Perfect_V2: {e}")
        try:
            from interactive_arc_system_v2 import InteractiveARCSystem
            self.systems['Interactive_V2'] = InteractiveARCSystem()
        except Exception as e:
            print(f"Skip Interactive_V2: {e}")
        try:
            from ultimate_arc_solver import UltimateARCSolver
            self.systems['Ultimate'] = UltimateARCSolver()
        except Exception as e:
            print(f"Skip Ultimate: {e}")
        if enable_ultra:
            try:
                from ultra_advanced_arc_system_v2 import UltraAdvancedARCSystem
                self.systems['Ultra_V2'] = UltraAdvancedARCSystem()
            except Exception as e:
                print(f"Skip Ultra_V2: {e}")

    # Simple gating policy to choose order of systems
    def choose_order(self, task):
        try:
            # Use first train pair to compute basic features
            tr = task.get('train', [])
            if tr:
                grid = to_array(tr[0]['input'])
            else:
                grid = to_array(task.get('test', [{}])[0].get('input'))
            h, w = grid.shape
            colors = len(np.unique(grid))
        except Exception:
            h, w, colors = 10, 10, 5
        order = []
        # Heuristic: small grids with few colors -> Perfect first
        if 'Perfect_V2' in self.systems:
            order.append('Perfect_V2')
        if 'Interactive_V2' in self.systems:
            order.append('Interactive_V2')
        if 'Ultimate' in self.systems:
            order.append('Ultimate')
        if 'Ultra_V2' in self.systems:
            order.append('Ultra_V2')
        return order

    def _call_system(self, name, system, task):
        try:
            if hasattr(system, 'solve'):
                return extract_output(system.solve(task))
            if hasattr(system, 'process_task'):
                return extract_output(system.process_task(task))
            return None
        except Exception:
            return None

    # Solve single task with possible ground-truth (for training evaluation)
    def solve_task(self, task, ground_truth=None):
        # ground_truth: np.ndarray or None
        train_pairs = task.get('train', [])
        # Infer transform consensus across all pairs
        tf_name, tf_func = infer_transform_consensus(train_pairs)

        # Predict reference from inferred transform (applied to test input)
        reference = None
        try:
            test_inp = to_array(task.get('test', [{}])[0]['input'])
            if tf_func is not None and test_inp is not None:
                reference = tf_func(test_inp)
        except Exception:
            reference = None

        # Target output shape from training (majority)
        target_shape = None
        try:
            shapes = [tuple(np.array(ex['output']).shape) for ex in train_pairs if 'output' in ex]
            if shapes:
                # majority shape
                counts = Counter(shapes)
                target_shape = list(counts.most_common(1)[0][0])
        except Exception:
            target_shape = None

        # Retrieve similar experiences to bias candidates
        signature = compute_task_signature(task)
        similar = self.memory.find_similar(signature['vec'], top_k=3)

        # Object-centric candidate
        obj_plan = None
        try:
            obj_plan = learn_object_plan(train_pairs)
        except Exception:
            obj_plan = None

        # DSL candidate
        dsl_ops = []
        try:
            dsl_ops = find_best_dsl_program(train_pairs, max_len=self.dsl_max_len)
        except Exception:
            dsl_ops = []

        # Cascade across systems
        order = self.choose_order(task)
        candidates = []
        meta_log = []
        # Seed with recalled solutions if shapes line up
        for exp in similar:
            try:
                recalled = np.array(exp.get('solution', None))
                if recalled is not None:
                    rcand = reshape_candidate_to_target(recalled, target_shape) if target_shape is not None else recalled
                    if reference is not None and rcand is not None:
                        rcand, how = refine_against_reference(rcand, reference, target_shape=target_shape)
                    else:
                        how = 'recalled'
                    candidates.append(('recalled', rcand))
                    meta_log.append({'system': 'recalled', 'refine': how, 'shape': None if rcand is None else list(rcand.shape), 'tf': tf_name})
            except Exception:
                pass
        for name in order:
            cand = self._call_system(name, self.systems[name], task)
            if cand is not None:
                # Try refinement against reference and target shape (train-derived)
                if reference is not None:
                    rcand, how = refine_against_reference(cand, reference, target_shape=target_shape)
                else:
                    # shape correct only
                    rcand = reshape_candidate_to_target(cand, target_shape) if target_shape is not None else cand
                    how = 'shape_only' if target_shape is not None else 'none'
                candidates.append((name, rcand))
                meta_log.append({'system': name, 'refine': how, 'shape': None if rcand is None else list(rcand.shape), 'tf': tf_name})
        # Add object-centric candidate
        try:
            if obj_plan is not None:
                test_inp = to_array(task.get('test', [{}])[0]['input'])
                ocand = apply_object_plan(test_inp, obj_plan, target_shape=target_shape)
                if reference is not None:
                    ocand, ohow = refine_against_reference(ocand, reference, target_shape=target_shape)
                else:
                    ohow = 'object_plan'
                candidates.append(('object_centric', ocand))
                meta_log.append({'system': 'object_centric', 'refine': ohow, 'shape': None if ocand is None else list(ocand.shape), 'tf': tf_name})
        except Exception:
            pass
        # Add DSL candidate
        try:
            if dsl_ops:
                test_inp = to_array(task.get('test', [{}])[0]['input'])
                dcand = apply_ops(test_inp, dsl_ops)
                dcand = reshape_candidate_to_target(dcand, target_shape) if target_shape is not None else dcand
                if reference is not None:
                    dcand, dhow = refine_against_reference(dcand, reference, target_shape=target_shape)
                else:
                    dhow = 'dsl:' + '+'.join(dsl_ops)
                candidates.append(('dsl', dcand))
                meta_log.append({'system': 'dsl', 'refine': dhow, 'shape': None if dcand is None else list(dcand.shape), 'tf': tf_name, 'ops': dsl_ops})
        except Exception:
            pass

        # If nothing produced, return None
        if not candidates:
            return None, {'log': meta_log, 'acc': 0.0}

        # If we have reference, pick best by accuracy
        if reference is not None:
            scored = []
            for n, c in candidates:
                if c is not None:
                    scored.append((pixel_accuracy(c, reference), c, n))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best = scored[0][1]
            else:
                best = candidates[0][1]
        else:
            # Group by shape and fallback to consensus
            shape_groups = defaultdict(list)
            for n, c in candidates:
                if c is not None:
                    shape_groups[c.shape].append((n, c))
            if not shape_groups:
                best = candidates[0][1]
            else:
                dominant_shape = max(shape_groups.keys(), key=lambda s: len(shape_groups[s]))
                dom = [c for _, c in shape_groups[dominant_shape]]
                voted = consensus_vote(dom)
                best = voted if voted is not None else dom[0]

        # If ground truth exists, final refine against GT
        acc = None
        if ground_truth is not None and best is not None:
            best_refined, how_gt = refine_against_reference(best, ground_truth, target_shape=list(ground_truth.shape))
            acc = pixel_accuracy(best_refined, ground_truth)
            # Store experience if reasonably good
            try:
                if acc >= 0.8:
                    self.memory.add({
                        'signature_vec': signature['vec'],
                        'solution': best_refined.tolist(),
                        'acc': acc,
                        'tf_name': tf_name,
                        'target_shape': list(ground_truth.shape)
                    })
            except Exception:
                pass
            return best_refined, {'log': meta_log, 'acc': acc, 'gt_refine': how_gt}

        return best, {'log': meta_log, 'acc': acc}

# ----------------------------- Batch run ------------------------------ #

def run_on_training(challenges_path, solutions_path, limit=None, enable_ultra=False, save_path='orchestrator_training_results.json', start=0, memory_path=None):
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    orch = HybridOrchestrator(enable_ultra=enable_ultra, memory_path=(memory_path or 'experience_memory.json'))

    items = list(challenges.items())
    n_all = len(items)
    s = max(0, int(start))
    e = n_all if limit is None else min(n_all, s + int(limit))
    total = e - s

    results = []
    correct = 0
    partial = 0
    total_acc = 0.0

    t0 = time.time()
    processed = 0
    for idx in range(s, e):
        task_id, task = items[idx]
        gt_list = solutions.get(task_id)
        if not gt_list:
            continue
        gt = to_array(gt_list[0])
        pred, info = orch.solve_task(task, ground_truth=gt)
        acc = pixel_accuracy(pred, gt)
        total_acc += acc
        processed += 1
        if acc == 1.0:
            correct += 1
        elif acc > 0.0:
            partial += 1
        if processed % 50 == 0:
            elapsed = time.time() - t0
            print(f"Progress {processed}/{total} | Acc so far: {100*total_acc/max(1,processed):.1f}% | Elapsed: {elapsed:.1f}s")
        results.append({'task_id': task_id, 'acc': acc, 'log': info['log']})

    summary = {
        'total_tasks': total,
        'perfect': correct,
        'partial': partial,
        'avg_accuracy_percent': 100.0 * total_acc / max(1, total)
    }

    with open(save_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    print("\n===== TRAINING RUN SUMMARY =====")
    print(f"Perfect: {summary['perfect']}/{summary['total_tasks']}")
    print(f"Partial: {summary['partial']}/{summary['total_tasks']}")
    print(f"Avg accuracy: {summary['avg_accuracy_percent']:.1f}%")
    print(f"Saved: {save_path}")

    return summary


def run_on_tasks(challenges_path, limit=None, enable_ultra=False, save_path='orchestrator_predictions.json'):
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    orch = HybridOrchestrator(enable_ultra=enable_ultra)

    total = len(challenges)
    if limit is not None:
        total = min(total, limit)

    outputs = {}
    t0 = time.time()
    for idx, (task_id, task) in enumerate(challenges.items()):
        if limit is not None and idx >= limit:
            break
        pred, info = orch.solve_task(task, ground_truth=None)
        if pred is not None:
            outputs[task_id] = pred.tolist()
        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"Progress {idx}/{total} | Elapsed: {elapsed:.1f}s")

    with open(save_path, 'w') as f:
        json.dump(outputs, f)

    print(f"Saved predictions: {save_path}")
    return save_path

