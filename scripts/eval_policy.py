import argparse
import yaml
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.kuka_table_tennis_env import KukaTableTennisEnv
from data.shot_samplers import EasyParams, EasyRandomShotSampler, NPZShotSampler

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_cfg", type=str, default="configs/env.yaml")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--vecnorm", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    cfg = load_yaml(args.env_cfg)

    mode = cfg["sampler"]["mode"]
    if mode == "easy_random":
        ep = cfg["sampler"]["easy"]
        sampler = EasyRandomShotSampler(EasyParams(**ep), seed=0)
    else:
        sampler = NPZShotSampler(cfg["sampler"]["latte"]["npz_path"], seed=0)

    def _make():
        return KukaTableTennisEnv(cfg=cfg, shot_sampler=sampler, seed=0, render_mode=None)

    env = DummyVecEnv([_make])
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(args.model)

    print(f"\n{'='*60}")
nvcc -V - {args.episodes} episodes")
    print(f"{'='*60}\n")

    succ = 0
    hit = 0
    clear_net = 0
    land = 0
    total_reward = 0.0

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_hit = False
        ep_clear = False
        ep_land = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            
            info0 = info[0]
            if info0.get("hit", False):
                ep_hit = True
            if info0.get("cleared_net", False):
                ep_clear = True
            if info0.get("landed", False):
                ep_land = True
            if info0.get("success", False):
                succ += 1
        
        if ep_hit:
            hit += 1
        if ep_clear:
            clear_net += 1
        if ep_land:
            land += 1
        total_reward += ep_reward
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{args.episodes} - "
                  f"Success: {succ}/{ep+1} ({100*succ/(ep+1):.1f}%)")

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"Episodes:        {args.episodes}")
    print(f"Hit Rate:        {100*hit/args.episodes:6.2f}%  ({hit}/{args.episodes})")
    print(f"Clear Net Rate:  {100*clear_net/args.episodes:6.2f}%  ({clear_net}/{args.episodes})")
    print(f"Land Rate:       {100*land/args.episodes:6.2f}%  ({land}/{args.episodes})")
    print(f"Success Rate:    {100*succ/args.episodes:6.2f}%  ({succ}/{args.episodes})")
    print(f"Avg Reward:      {total_reward/args.episodes:8.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
