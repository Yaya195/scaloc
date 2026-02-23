import torch


def resolve_device(preferred: str = "auto") -> str:
    """
    Resolve runtime device string.

    Rules:
      - "auto": use CUDA when available, else CPU
      - "cuda": use CUDA when available, else fall back to CPU with notice
      - "cpu": always CPU
    """
    pref = (preferred or "auto").lower()

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if pref == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[device] CUDA requested but not available, falling back to CPU.")
        return "cpu"

    if pref == "cpu":
        return "cpu"

    print(f"[device] Unknown device '{preferred}', using auto selection.")
    return "cuda" if torch.cuda.is_available() else "cpu"
