"""
Hit-friendly sampler: 保证球经过击球窗口
✅ 改进 1) 围绕球拍位置的 curriculum sampler
"""
import numpy as np

def sample_hit_friendly(
    rng: np.random.Generator,
    net_x: float,
    table_center_x: float,
    table_half_length: float,
    table_half_width: float,
    table_z: float,
    self_side_is_x_less: bool,
    racket_pos: np.ndarray = None,
    curriculum_radius: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一个保证经过击球窗口的来球
    
    改进策略：
    1. 如果提供了 racket_pos，击球点围绕球拍位置采样（curriculum）
    2. 否则使用原来的桌面窗口采样（向后兼容）
    3. 从对面生成初始位置，反推速度让球在 t_hit 到达 p_hit
    """
    x_min = table_center_x - table_half_length
    x_max = table_center_x + table_half_length
    
    # 击球点采样：围绕球拍 vs 桌面窗口
    if racket_pos is not None:
        # 围绕球拍位置采样（curriculum learning）
        offset = rng.uniform(-1, 1, size=3)
        offset_norm = np.linalg.norm(offset)
        if offset_norm > 1e-6:
            offset = offset / offset_norm * rng.uniform(0, curriculum_radius)
        else:
            offset = np.zeros(3)
        
        p_hit = racket_pos + offset
        # 确保击球点在合理范围内
        p_hit[0] = np.clip(p_hit[0], x_min + 0.2, x_max - 0.2)
        p_hit[1] = np.clip(p_hit[1], -0.5, 0.5)
        p_hit[2] = np.clip(p_hit[2], 0.85, 1.15)
        
        x_hit, y_hit, z_hit = p_hit[0], p_hit[1], p_hit[2]
    else:
        # 原来的桌面窗口采样（向后兼容）
        if self_side_is_x_less:
            x_hit = rng.uniform(x_min + 0.3, net_x - 0.2)
        else:
            x_hit = rng.uniform(net_x + 0.2, x_max - 0.3)
        
        y_hit = rng.uniform(-0.35, 0.35)
        z_hit = rng.uniform(0.85, 1.15)
    
    # 初始位置：从对面上方出生
    if self_side_is_x_less:
        x0 = rng.uniform(net_x + 0.2, x_max - 0.1)
    else:
        x0 = rng.uniform(x_min + 0.1, net_x - 0.2)
    
    y0 = rng.uniform(-0.3, 0.3)
    z0 = rng.uniform(1.2, 1.6)
    
    # ✅ Step A 测试：固定飞行时间
    t = rng.uniform(1.0, 1.2)
    
    # 重力
    g = -9.81
    
    # 反推初速度
    vx = (x_hit - x0) / t
    vy = (y_hit - y0) / t
    vz = (z_hit - z0 - 0.5 * g * t * t) / t
    
    pos = np.array([x0, y0, z0], dtype=np.float32)
    vel = np.array([vx, vy, vz], dtype=np.float32)
    p_hit = np.array([x_hit, y_hit, z_hit], dtype=np.float32)
    
    return pos, vel, p_hit  # ✅ 2) 同时返回目标击球点
