import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LatteReconLayout:
    ball_row: int = -1
    hit_row: Optional[int] = None
    meta_row: int = 0

class LatteReconReader:
    def __init__(self, layout: Optional[LatteReconLayout] = None, verbose: bool = True):
        self.layout = layout or LatteReconLayout()
        self.verbose = verbose

    def load(self, recon_path: str) -> Dict[str, Any]:
        recon = np.load(recon_path)
        assert recon.ndim == 3 and recon.shape[2] == 3, f"Unexpected shape: {recon.shape}"
        T, K, _ = recon.shape

        guessed_hit_row = self._guess_hit_row(recon)
        guessed_ball_row = self._guess_ball_row(recon)

        hit_row = self.layout.hit_row if self.layout.hit_row is not None else guessed_hit_row
        ball_row = self.layout.ball_row if self.layout.ball_row is not None else guessed_ball_row

        fps, nframes, nusable = self._try_parse_meta(recon)

        if self.verbose:
            print(f"[Recon] path={recon_path}")
            print(f"[Recon] shape={recon.shape}, fps={fps}, nframes={nframes}, nusable={nusable}")
            print(f"[Recon] guessed_hit_row={guessed_hit_row}, using_hit_row={hit_row}")
            print(f"[Recon] guessed_ball_row={guessed_ball_row}, using_ball_row={ball_row}")

        ball = recon[:, ball_row, :].astype(np.float32)
        hit = None
        if hit_row is not None:
            hit = recon[:, hit_row, :].astype(np.float32)
        return {
            "recon": recon,
            "ball": ball,
            "hit": hit,
            "fps": fps,
            "nframes": nframes,
            "nusable": nusable,
            "ball_row": ball_row,
            "hit_row": hit_row,
        }

    def _try_parse_meta(self, recon: np.ndarray):
        fps = None
        nframes = None
        nusable = None
        try:
            meta = recon[0, self.layout.meta_row, :]
            if meta[0] > 1 and meta[0] < 240:
                fps = float(meta[0])
                nframes = int(meta[1]) if meta[1] > 0 else None
                nusable = int(meta[2]) if meta[2] > 0 else None
        except Exception:
            pass
        return fps, nframes, nusable

    def _guess_hit_row(self, recon: np.ndarray) -> Optional[int]:
        T, K, _ = recon.shape
        best_row = None
        best_score = -1.0
        for r in range(K):
            x = recon[:, r, :]
            close01 = np.mean((np.abs(x - 0) < 1e-3) | (np.abs(x - 1) < 1e-3))
            in01 = np.mean((x >= -1e-3) & (x <= 1 + 1e-3))
            score = close01 * 0.7 + in01 * 0.3
            if score > best_score:
                best_score = score
                best_row = r
        if best_score < 0.85:
            return None
        return int(best_row)

    def _guess_ball_row(self, recon: np.ndarray) -> int:
        T, K, _ = recon.shape
        scores = []
        for r in range(K):
            x = recon[:, r, :]
            if not np.isfinite(x).all():
                scores.append(-1e9)
                continue
            var = float(np.mean(np.var(x, axis=0)))
            max_abs = float(np.max(np.abs(x)))
            penalty = 0.0
            if max_abs > 10.0:
                penalty += (max_abs - 10.0) * 2.0
            scores.append(var - penalty)
        return int(np.argmax(scores))
