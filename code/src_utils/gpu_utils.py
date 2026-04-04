from __future__ import annotations

import gc
from typing import Dict


def gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def get_gpu_stats() -> Dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": 0.0,
                "total_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
            }

        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return {
            "available": 1.0,
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
        }
    except Exception:
        return {
            "available": 0.0,
            "total_gb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
        }


def clear_gpu_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
