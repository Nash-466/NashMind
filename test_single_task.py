from __future__ import annotations
#!/usr/bin/env python3
import run_training_evaluation as rte
import traceback
import time

print("Testing single task...")
start = time.time()

try:
    r = rte.run_evaluation(max_problems=1)
    elapsed = time.time() - start
    print(f"SUCCESS! Time: {elapsed:.2f}s")
    print(f"Results: {r}")
except Exception as e:
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.2f}s: {str(e)}")
    traceback.print_exc()
