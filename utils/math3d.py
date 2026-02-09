import numpy as np

def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """
    x: (T,3) -> v: (T,3)
    末端用前向/后向差分，中间用中心差分
    """
    v = np.zeros_like(x)
    if len(x) < 2:
        return v
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    if len(x) > 2:
        v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    return v

def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= max_norm or n < 1e-8:
        return v
    return v * (max_norm / n)
