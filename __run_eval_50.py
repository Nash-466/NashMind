from __future__ import annotations
import json
import time

try:
    import run_training_evaluation as rte
except Exception as e:
    print("IMPORT_ERROR:", e)
    raise


def main() -> None:
    start = time.time()
    results = rte.run_evaluation(max_problems=50)
    out_name = f"training_eval_50_{int(start)}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(out_name)


if __name__ == "__main__":
    main()


