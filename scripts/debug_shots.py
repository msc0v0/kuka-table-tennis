import os
import argparse
import json
import numpy as np
from tqdm import tqdm

import yaml
from data.shot_samplers import EasyParams, EasyRandomShotSampler, NPZShotSampler

# 你自己的 env 路径按实际改
from envs.kuka_table_tennis_env import KukaTableTennisEnv


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_debug(env_cfg: str, out_dir: str, n: int = 200, seed: int = 0, steps_per_ep: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_yaml(env_cfg)

    # sampler
    mode = cfg["sampler"]["mode"]
    if mode == "easy_random":
        ep = cfg["sampler"]["easy"]
        sampler = EasyRandomShotSampler(EasyParams(**ep), seed=seed)
    elif mode == "latte_npz":
        npz_path = cfg["sampler"]["latte"]["npz_path"]
        sampler = NPZShotSampler(npz_path, seed=seed)
    else:
        raise ValueError(mode)

    env = KukaTableTennisEnv(cfg=cfg, shot_sampler=sampler, seed=seed, render_mode=None)

    # 从 env 里读关键几何参数（不依赖你猜坐标系）
    meta = {
        "net_x": float(getattr(env, "net_x", cfg["table"]["net_x"])),
        "net_height": float(getattr(env, "net_h", cfg["table"]["net_height"])),
        "table_center_x": float(getattr(env, "table_center_x", cfg["table"]["table_center_x"])),
        "table_half_length": float(getattr(env, "table_half_len", cfg["table"]["table_half_length"])),
        "table_half_width": float(getattr(env, "table_half_w", cfg["table"]["table_half_width"])),
        "table_z": float(getattr(env, "table_z", cfg["table"]["table_height_z"] + cfg["table"]["table_top_thickness"])),
        "dt": float(getattr(env, "dt", 0.0)),
    }
    with open(os.path.join(out_dir, "env_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    records = []
    no_action = np.zeros(env.action_space.shape, dtype=np.float32)

    for ep_i in tqdm(range(n), desc="debug episodes"):
        obs, _ = env.reset()

        # 初始球状态
        # env._get_ball_state() 里如果没有，就从 info 或直接读 qpos/qvel
        try:
            ball_pos0, ball_vel0 = env._get_ball_state()
        except Exception:
            ball_pos0 = env.data.qpos[-7:-4].copy()
            ball_vel0 = env.data.qvel[-6:-3].copy()

        first_table_contact = None
        first_table_side = None
        hit_any = False
        success_any = False

        terminated = False
        truncated = False

        for t in range(steps_per_ep):
            obs, rew, terminated, truncated, info = env.step(no_action)

            hit = bool(info.get("hit", info.get("hit_done", False)))
            landed = bool(info.get("landed", False))
            success = bool(info.get("success", False))

            if hit:
                hit_any = True

            if success:
                success_any = True

            # 记录第一次落台点与半场（如果 env 没提供，就从 ball_pos 取）
            if landed and first_table_contact is None:
                bp = info.get("ball_pos", None)
                if bp is None:
                    bp = env.data.qpos[-7:-4].copy()
                bp = np.asarray(bp, dtype=np.float32)
                first_table_contact = bp.tolist()

                # 按 env 的判定函数来决定半场（不猜方向）
                if hasattr(env, "_is_opponent_side"):
                    first_table_side = "opponent" if env._is_opponent_side(float(bp[0])) else "self"
                else:
                    first_table_side = "opponent" if float(bp[0]) > meta["net_x"] else "self"

            if terminated or truncated:
                break

        records.append({
            "episode": ep_i,
            "ball_pos0": ball_pos0.tolist(),
            "ball_vel0": ball_vel0.tolist(),
            "hit_any": hit_any,
            "success_any": success_any,
            "first_table_contact": first_table_contact,
            "first_table_side": first_table_side,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "no_hit_but_success": (not hit_any) and success_any,
        })

    # 汇总
    no_hit_success = sum(1 for r in records if r["no_hit_but_success"])
    hit_rate = sum(1 for r in records if r["hit_any"]) / len(records)
    success_rate = sum(1 for r in records if r["success_any"]) / len(records)
    land_rate = sum(1 for r in records if r["first_table_contact"] is not None) / len(records)

    summary = {
        "n": len(records),
        "hit_rate": hit_rate,
        "success_rate": success_rate,
        "land_rate": land_rate,
        "no_hit_but_success_count": no_hit_success,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(os.path.join(out_dir, "episodes.npz"), records=np.array(records, dtype=object))

    print("\n=== Debug Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir}/env_meta.json, summary.json, episodes.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", type=str, default="configs/env.yaml")
    ap.add_argument("--out_dir", type=str, default="debug_out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    run_debug(args.env_cfg, args.out_dir, args.n, args.seed, args.steps)
