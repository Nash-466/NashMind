import json
from pathlib import Path
import numpy as np

# Ensure root import
import sys
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from unified_solver_wrapper import UnifiedSolverWrapper

# Minimal dummy ARC-like task
task = {
    'train': [
        {
            'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'output': [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        }
    ],
    'test': [
        {'input': [[1, 1, 1], [2, 2, 2], [3, 3, 3]]}
    ]
}

w = UnifiedSolverWrapper()
results = []

for sysdef in w.systems:
    name = sysdef['name']
    kind = sysdef.get('type', 'unknown')
    ok = True
    shape = None
    err = None
    try:
        sol = sysdef['solve'](task)
        if isinstance(sol, list) and sol:
            sol = sol[0]
        if isinstance(sol, np.ndarray):
            shape = tuple(sol.shape)
        else:
            ok = False
            err = f"unexpected result type: {type(sol).__name__}"
    except Exception as e:
        ok = False
        err = str(e)
    results.append({
        'name': name,
        'type': kind,
        'ok': ok,
        'shape': shape,
        'error': err,
    })

outdir = Path('reports')
outdir.mkdir(parents=True, exist_ok=True)
Path(outdir / 'smoke_all_systems.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))

