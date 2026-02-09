import argparse
import numpy as np
from data.shot_samplers import EasyParams, EasyRandomShotSampler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = EasyParams(
        spawn_x=2.8,
        spawn_y_min=-0.4,
        spawn_y_max=0.4,
        spawn_z_min=0.85,
        spawn_z_max=1.15,
        vx_min=-8.0,
        vx_max=-4.0,
        vy_min=-1.5,
        vy_max=1.5,
        vz_min=-2.0,
        vz_max=1.0,
    )
    sampler = EasyRandomShotSampler(params, seed=args.seed)

    pos = np.zeros((args.n, 3), dtype=np.float32)
    vel = np.zeros((args.n, 3), dtype=np.float32)
    for i in range(args.n):
        p, v = sampler.sample()
        pos[i] = p
        vel[i] = v

    np.savez_compressed(args.out, pos=pos, vel=vel)
    print(f"[OK] saved {args.n} shots -> {args.out}")

if __name__ == "__main__":
    main()
