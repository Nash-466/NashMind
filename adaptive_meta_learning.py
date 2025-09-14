# -*- coding: utf-8 -*-
"""
Adaptive Meta-Learner
- يبني ذاكرة معرفية رمزية خفيفة من أمثلة التدريب (توقيعات -> استراتيجيات)
- يولّد بايبلاين تحويلات حتمية بسيطة (rotate/mirror/transpose/upscale/color_map)
- يتعلم لحظيًا: يحفظ الاستراتيجية الناجحة، ويُحسنها من التغذية الراجعة عند الفشل القريب

ملاحظة: يعتمد على symbolic_rule_engine لوظائف التحويل الأساسية.
"""
from __future__ import annotations
from collections.abc import Callable
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

# استيراد الأدوات الرمزية
from symbolic_rule_engine import (
    Grid, Example, Task, dims, equal,
    learn_color_map, learn_mirror_rule, learn_rotation_or_transpose, learn_nn_upscale,
    apply_color_map, mirror_h, mirror_v, rotate90, rotate180, rotate270, transpose, upscale
)

MEM_PATH = Path('meta_memory.json')


def load_memory() -> Dict[str, Any]:
    if MEM_PATH.exists():
        try:
            return json.loads(MEM_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_memory(mem: Dict[str, Any]) -> None:
    try:
        MEM_PATH.write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def build_signature(train: List[Example]) -> str:
    # ملامح بسيطة قابلة للتعميم
    mr = learn_mirror_rule(train)
    rot = learn_rotation_or_transpose(train)
    up = learn_nn_upscale(train)
    mp = learn_color_map(train)
    same_dims = all(dims(ex['input']) == dims(ex['output']) for ex in train)
    parts = [
        f"same={int(same_dims)}",
        f"mir={mr or 'N'}",
        f"rot={rot or 'N'}",
        f"up={(str(up['sx'])+'x'+str(up['sy'])) if up else '1x1'}",
        f"cm={int(bool(mp))}"
    ]
    return '|'.join(parts)


def apply_pipeline(inp: Grid, pipeline: List[Dict[str, Any]]) -> Grid:
    out = inp
    for step in pipeline:
        op = step.get('op')
        if op == 'mirror':
            out = mirror_h(out) if step.get('arg') == 'H' else mirror_v(out)
        elif op == 'rotate':
            a = step.get('arg')
            if a == 'R90':
                out = rotate90(out)
            elif a == 'R180':
                out = rotate180(out)
            elif a == 'R270':
                out = rotate270(out)
        elif op == 'transpose':
            out = transpose(out)
        elif op == 'upscale':
            out = upscale(out, int(step['sx']), int(step['sy']))
        elif op == 'color_map':
            out = apply_color_map(out, {int(k): int(v) for k, v in step['map'].items()})
        elif op == 'solid_fill':
            h, w = dims(out)
            c = int(step['color'])
            out = [[c for _ in range(w)] for __ in range(h)]
    return out


def try_fit_pipeline(train: List[Example]) -> Optional[List[Dict[str, Any]]]:
    """
    بحث صغير عن بايبلاينات قصيرة (حتى طول 2) من عمليات رمزية،
    مع إمكانية إضافة color_map تصحيحية في النهاية إذا كانت متسقة عبر جميع الأمثلة.
    """
    # مرشّحات جاهزة من التعلم المباشر
    mr = learn_mirror_rule(train)
    rot = learn_rotation_or_transpose(train)
    up = learn_nn_upscale(train)
    mp_direct = learn_color_map(train)

    # 0) حلول مباشرة أحادية العملية
    if mr:
        pipe = [{'op': 'mirror', 'arg': mr}]
        if all(equal(apply_pipeline(ex['input'], pipe), ex['output']) for ex in train):
            return pipe
    if rot:
        pipe = [{'op': 'transpose'}] if rot == 'T' else [{'op': 'rotate', 'arg': rot}]
        if all(equal(apply_pipeline(ex['input'], pipe), ex['output']) for ex in train):
            return pipe
    if up:
        pipe = [{'op': 'upscale', 'sx': int(up['sx']), 'sy': int(up['sy'])}]
        if all(equal(apply_pipeline(ex['input'], pipe), ex['output']) for ex in train):
            return pipe
    if mp_direct:
        pipe = [{'op': 'color_map', 'map': mp_direct}]
        if all(equal(apply_pipeline(ex['input'], pipe), ex['output']) for ex in train):
            return pipe

    # 1) بحث مركب محدود
    # توليد مجموعة المرشحين (بدون معاملات حرة باستثناء derived ones)
    candidate_ops: List[Dict[str, Any]] = []
    if mr:
        candidate_ops.append({'op': 'mirror', 'arg': mr})
    if rot:
        candidate_ops.append({'op': 'transpose'} if rot == 'T' else {'op': 'rotate', 'arg': rot})
    if up:
        candidate_ops.append({'op': 'upscale', 'sx': int(up['sx']), 'sy': int(up['sy'])})

    # دائماً يمكن تجربة color_map كخطوة أخيرة إن اتسقت
    def derive_map(a: Grid, b: Grid) -> Optional[Dict[int, int]]:
        h1, w1 = dims(a)
        h2, w2 = dims(b)
        if (h1, w1) != (h2, w2):
            return None
        mp: Dict[int, int] = {}
        for i in range(h1):
            for j in range(w1):
                va, vb = a[i][j], b[i][j]
                if va in mp and mp[va] != vb:
                    return None
                mp[va] = vb
        return mp if mp else None

    def check_pipe(pipe: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        # تحقق مباشر
        if all(equal(apply_pipeline(ex['input'], pipe), ex['output']) for ex in train):
            return pipe
        # جرّب إضافة color_map إن أمكن
        maps: List[Dict[int, int]] = []
        for ex in train:
            mid = apply_pipeline(ex['input'], pipe)
            mp = derive_map(mid, ex['output'])
            if not mp:
                return None
            maps.append(mp)
        # تقاطع/اتساق الخرائط
        final_map: Dict[int, int] = {}
        for mp in maps:
            for k, v in mp.items():
                if k in final_map and final_map[k] != v:
                    return None
                final_map[k] = v
        pipe2 = pipe + [{'op': 'color_map', 'map': final_map}]
        if all(equal(apply_pipeline(ex['input'], pipe2), ex['output']) for ex in train):
            return pipe2
        return None

    # طول 1
    for op in candidate_ops:
        res = check_pipe([op])
        if res:
            return res

    # طول 2 (كل التركيبات)
    for i in range(len(candidate_ops)):
        for j in range(len(candidate_ops)):
            res = check_pipe([candidate_ops[i], candidate_ops[j]])
            if res:
                return res

    # محاولات مركبة بسيطة مع mp_direct
    if mp_direct:
        for op in candidate_ops:
            res = check_pipe([op, {'op': 'color_map', 'map': mp_direct}])
            if res:
                return res

    return None


class AdaptiveMetaLearner:
    def __init__(self):
        self.mem: Dict[str, Any] = load_memory()

    def _remember(self, sig: str, pipeline: List[Dict[str, Any]]):
        if not pipeline:
            return
        entry = {'pipeline': pipeline}
        self.mem[sig] = entry
        save_memory(self.mem)

    def _get(self, sig: str) -> Optional[List[Dict[str, Any]]]:
        item = self.mem.get(sig)
        if not item:
            return None
        return item.get('pipeline')

    def _derive_color_correction(self, pred: Grid, expected: Grid) -> Optional[Dict[int, int]]:
        h1, w1 = dims(pred)
        h2, w2 = dims(expected)
        if (h1, w1) != (h2, w2):
            return None
        mp: Dict[int, int] = {}
        for i in range(h1):
            for j in range(w1):
                a = pred[i][j]
                b = expected[i][j]
                if a in mp and mp[a] != b:
                    return None
                mp[a] = b
        return mp if mp else None

    def refine_from_feedback(self, task: Task, pred: Grid, expected: Grid) -> bool:
        # يزيد color_map التصحيحي في نهاية البايبلاين إن كان متسقًا
        train = task.get('train', [])
        sig = build_signature(train)
        pipe = self._get(sig)
        if pipe is None:
            return False
        corr = self._derive_color_correction(pred, expected)
        if not corr:
            return False
        new_pipe = pipe + [{'op': 'color_map', 'map': corr}]
        # لا نضيف إلا إذا لم تكن نفس الخطوة موجودة مسبقًا
        self._remember(sig, new_pipe)
        return True

    def solve(self, task: Task) -> List[Optional[Grid]]:
        train = task.get('train', [])
        test = task.get('test', [])
        sig = build_signature(train)

        pipeline = self._get(sig)
        if pipeline is None:
            pipeline = try_fit_pipeline(train)
            if pipeline:
                self._remember(sig, pipeline)

        outs: List[Optional[Grid]] = []
        for ex in test:
            if pipeline is None:
                outs.append(None)
            else:
                outs.append(apply_pipeline(ex['input'], pipeline))
        return outs

