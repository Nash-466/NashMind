import json
import platform
import sys
from pathlib import Path

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": None,
    "cuda_available": False,
    "cuda_device_count": 0,
    "gpu_backend": "cpu",
    "systems": [],
}

try:
    import torch  # type: ignore
    info["torch"] = getattr(torch, "__version__", None)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if info["cuda_available"]:
            info["gpu_backend"] = "nvidia-cuda"
    except Exception:
        pass
    # Try DirectML probe
    if not info["cuda_available"]:
        try:
            import torch_directml  # type: ignore
            dml = torch_directml.device()
            # Create a tiny tensor to verify
            _ = torch.zeros((1,), device=dml)
            info["gpu_backend"] = "directml"
        except Exception:
            pass
except Exception:
    pass

try:
    # Ensure project root on sys.path
    import sys as _sys
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    from unified_solver_wrapper import UnifiedSolverWrapper
    w = UnifiedSolverWrapper()
    info["systems"] = [{"name": s["name"], "type": s.get("type", "unknown")} for s in w.systems]
except Exception as e:
    info["systems_error"] = str(e)

outdir = Path("reports")
outdir.mkdir(parents=True, exist_ok=True)
Path(outdir / "system_diagnostics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2))
