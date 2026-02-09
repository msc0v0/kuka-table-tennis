import yaml
from envs.kuka_table_tennis_env import KukaTableTennisEnv
from data.shot_samplers import EasyParams, EasyRandomShotSampler
import numpy as np

with open('configs/env.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

ep = cfg['sampler']['easy']
sampler = EasyRandomShotSampler(EasyParams(**ep), seed=42)
env = KukaTableTennisEnv(cfg=cfg, shot_sampler=sampler, seed=42)

print("测试3个episode，看球落在哪一侧:")
for ep_num in range(3):
    print(f"\n=== Episode {ep_num+1} ===")
    obs, info = env.reset()
    
    for step in range(100):
        action = np.zeros(9)
        obs, reward, term, trunc, info = env.step(action)
        ball_pos = info['ball_pos']
        
        if info['landed'] or term or trunc:
            side = "对方侧" if ball_pos[0] > 1.5 else "机器人侧"
            print(f"Step {step}: X={ball_pos[0]:.2f}, Z={ball_pos[2]:.2f} ({side})")
            print(f"  hit={info['hit']}, cleared_net={info['cleared_net']}, landed={info['landed']}, success={info['success']}")
            break
