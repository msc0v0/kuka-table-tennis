"""快速测试环境是否正常工作"""
import yaml
from envs.kuka_table_tennis_env import KukaTableTennisEnv
from data.shot_samplers import EasyParams, EasyRandomShotSampler

# 加载配置
with open("configs/env.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# 创建采样器
ep = cfg["sampler"]["easy"]
sampler = EasyRandomShotSampler(EasyParams(**ep), seed=0)

# 创建环境
print("正在创建环境...")
try:
    env = KukaTableTennisEnv(cfg=cfg, shot_sampler=sampler, seed=0)
    print(f"✓ 环境创建成功！")
    print(f"  观测空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")

    # 测试 reset
    print("\n测试 reset...")
    obs, info = env.reset()
    print(f"✓ Reset 成功！观测形状: {obs.shape}")

    # 测试几
    print("\n测试前5步...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, hit={info['hit']}, cleared_net={info['cleared_net']}, success={info['success']}")
        if terminated or truncated:
            print(f"  Episode 结束")
            break

nvcc ")-V
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
