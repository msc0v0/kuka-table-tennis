import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from data.latte_recon_reader import LatteReconReader, LatteReconLayout
from utils.math3d import finite_diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ball_row", type=int, default=-999)
    ap.add_argument("--hit_row", type=int, default=-999)
    ap.add_argument("--after_hit_frames", type=int, default=3)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--vy_abs_max", type=float, default=2.0)
    ap.add_argument("--speed_min", type=float, default=2.0)
    ap.add_argument("--speed_max", type=float, default=12.0)
    args = ap.parse_args()

    layout = LatteReconLayout()
    if args.ball_row != -999:
        layout.ball_row = args.ball_row
    if args.hit_row != -999:
        layout.hit_row = args.hit_row

    reader = LatteReconReader(layout=layout, verbose=False)
    recon_paths = glob(os.path.join(args.root, "**", "*_recon.npy"), recursive=True)
    assert len(recon_paths) > 0, f"No recon found under: {args.root}"

    pos_list = []
    vel_list = []

    for p in tqdm(recon_paths, desc="scan recon"):
        out = reader.load(p)
        ball = out["ball"]
        fps = out["fps"] if out["fps"] is not None else args.fps
        dt = 1.0 / float(fps)
        vel = finite_diff(ball, dt)

        if out["hit"] is not None:
            hit = out["hit"]
            racket_hit = (hit[:, 0] > 0.5) | (hit[:, 1] > 0.5)
            hit_idx = np.where(racket_hit)[0]
            if len(hit_idx) == 0:
                continue
            for hi in hit_idx:
                t0 = hi + args.after_hit_frames
                if t0 < 0 or t0 >= len(ball):
                    continue
                p0 = ball[t0]
                v0 = vel[t0]
                s = float(np.linalg.norm(v0))
                if s < args.speed_min or s > args.speed_max:
                    continue
                if abs(v0[1]) > args.vy_abs_max:
                    continue
                pos_list.append(p0)
                vel_list.append(v0)
        else:
            T = len(ball)
            for t0 in [int(T*0.2), int(T*0.4), int(T*0.6)]:
                p0 = ball[t0]
                v0 = vel[t0]
                s = float(np.linalg.norm(v0))
                if s < args.speed_min or s > args.speed_max:
                    continue
                if abs(v0[1]) > args.vy_abs_max:
                    continue
                pos_list.append(p0)
                vel_list.append(v0)

    pos = np.asarray(pos_list, dtype=np.float32)
    vel = np.asarray(vel_list, dtype=np.float32)
    print(f"[OK] shots={len(pos)} -> {args.out}")
    np.savez_compressed(args.out, pos=pos, vel=vel)

if __name__ == "__main__":
    main()
