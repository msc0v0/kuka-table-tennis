import numpy as np

def build_obs(
    qpos9: np.ndarray,
    qvel9: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    racket_pos: np.ndarray,
    racket_vel: np.ndarray,
    prev_actions: np.ndarray,  # (H, A)
    include_relative: bool = True,
) -> np.ndarray:
    feats = [qpos9, qvel9, ball_pos, ball_vel]
    if include_relative:
        feats.append(ball_pos - racket_pos)
        feats.append(ball_vel - racket_vel)
    feats.append(prev_actions.flatten())
    return np.concatenate(feats).astype(np.float32)
