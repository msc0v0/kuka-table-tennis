import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EasyParams:
    spawn_x: float
    spawn_y_min: float
    spawn_y_max: float
    spawn_z_min: float
    spawn_z_max: float
    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    vz_min: float
    vz_max: float

class EasyRandomShotSampler:
    def __init__(self, params: EasyParams, seed: int = 0):
        self.p = params
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([
            self.p.spawn_x,
            self.rng.uniform(self.p.spawn_y_min, self.p.spawn_y_max),
            self.rng.uniform(self.p.spawn_z_min, self.p.spawn_z_max),
        ], dtype=np.float32)
        vel = np.array([
            self.rng.uniform(self.p.vx_min, self.p.vx_max),
            self.rng.uniform(self.p.vy_min, self.p.vy_max),
            self.rng.uniform(self.p.vz_min, self.p.vz_max),
        ], dtype=np.float32)
        return pos, vel

class NPZShotSampler:
    def __init__(self, npz_path: str, seed: int = 0):
        d = np.load(npz_path)
        self.pos = d["pos"].astype(np.float32)
        self.vel = d["vel"].astype(np.float32)
        assert len(self.pos) == len(self.vel) and len(self.pos) > 0
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        i = int(self.rng.integers(0, len(self.pos)))
        return self.pos[i].copy(), self.vel[i].copy()
