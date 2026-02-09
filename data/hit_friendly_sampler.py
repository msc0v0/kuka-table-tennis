"""
Hit-friendly sampler: 保证球经过击球窗口
    main()
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
) -> tuple[np.ndarray, np.ndarray]:
    """
#    生成一个保

    
    策略：
    1. 先确定一个"击球点" p_hit 在球
    2. 确定击球时间 t_hit
    3. 从对面生成初始位置，反推速度让球在 t_hit 到达 p_hit
    """
    x_min = table_center_x - table_half_length
    x_max = table_center_x + table_half_length
    
    # 击球窗口：更靠近机械臂侧，高度在易击打范围
    if self_side_is_x_less:
        x_hit = rng.uniform(x_min + 0.3, net_x - 0.2)
        x0 = rng.uniform(net_x + 0.2, x_max - 0.1)
        dir_sign = -1.0
    else:
        x_hit = rng.uniform(net_x + 0.2, x_max - 0.3)
        x0 = rng.uniform(x_min + 0.1, net_x - 0.2)
        dir_sign = 1.0
    
    # 击球点的y/z坐标
    y_hit = rng.uniform(-0.35, 0.35)
    z_hit = rng.uniform(0.85, 1.15)
    
    # 初始位置（对面上方）
    y0 = rng.uniform(-0.3, 0.3)
    z0 = rng.uniform(1.2, 1.6)
    
    # 飞行时间
    t = rng.uniform(0.5, 0.9)
    
    # 重力
    g = -9.81
    
    # 反推初速度
    vx = (x_hit - x0) / t
    vy = (y_hit - y0) / t
    vz = (z_hit - z0 - 0.5 * g * t * t) / t
    
    pos = np.array([x0, y0, z0], dtype=np.float32)
    vel = np.array([vx, vy, vz], dtype=np.float32)
    
    return pos, vel
