#!/usr/bin/env python3
"""
 Step B: 快速检查最近 100 局的终止原因分布
"""
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, '.')
from envs import KukaTableTennisEnv

def check_term_reasons(n_episodes=100):
    """运行 n_episodes 局，统计终止原因"""
    env = KukaTableTennisEnv(
        render_mode=None,
        max_steps=200,
        use_stage1=True,
    )
    
    term_reasons = []
    hit_count = 0
    
    print(f"🔍 运 {n_episodes} 局测试...")
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        reason = info.get('term_reason', 'unknown')
        term_reasons.append(reason)
        
        if reason == 'hit':
            hit_count += 1
        
        if (ep + 1) % 20 == 0:
            print(f"  进度: {ep+1}/{n_episodes}")
    
    env.close()
    
    # 统计
    counter = Counter(term_reasons)
    print("\n" + "="*60)
    print("📊 终止原因分布 (最近 %d 局):" % n_episodes)
    print("="*60)
    for reason, count in counter.most_common():
        pct = 100.0 * count / n_episodes
        print(f"  {reason:12s}: {count:4d} 局 ({pct:5.1f}%)")
    print("="*60)
    print(f"✅ Hit Rate: {100.0 * hit_count / n_episodes:.1f}%")
    print("="*60)
    
    return counter

if __name__ == '__main__':
    check_term_reasons(n_episodes=100)
