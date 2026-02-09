import numpy as np

def sample_incoming(
    rng: np.random.Generator,
    net_x: float,
    table_center_x: float,
    table_half_length: float,
    table_half_width: float,
    table_z: float,
    self_side_is_x_less: bool,
    spawn_margin: float = 0.25,
    spawn_y_range: float = 0.2,
    spawn_z_min: float = 1.2,
    spawn_z_max: float = 1.6,
    t_min: float = 0.38,
    t_max: float = 0.55,
    g: float = -9.81,
):
    """
    采样“来球”：
      1) 球从对方半台上方出生
      2) 第一落台在我方半台（桌面范围内）
    忽略空气阻力（先训通最重要）
    """
    x_min = table_center_x - table_half_length
    x_max = table_center_x + table_half_length

    # 目标第一落点：我方半台
    if self_side_is_x_less:
        x_land = rng.uniform(x_min + 0.10, net_x - 0.15)
        # 出生点：对方半台 (x > net_x)
        x0 = rng.uniform(net_x + spawn_margin, min(x_max - 0.05, net_x + 0.9))
    else:
        x_land = rng.uniform(net_x + 0.15, x_max - 0.10)
        # 出生点：对方半台 (x < net_x)
        x0 = rng.uniform(max(x_min + 0.05, net_x - 0.9), net_x - spawn_margin)

    y0 = rng.uniform(-spawn_y_range, spawn_y_range)
    z0 = rng.uniform(spawn_z_min, spawn_z_max)

    y_land = rng.uniform(-table_half_width * 0.4, table_half_width * 0.4)
    z_land = table_z  # 近似球心落到桌面高度（你也可以之后加 ball radius）

    t = rng.uniform(t_min, t_max)

    vx = (x_land - x0) / t
    vy = (y_land - y0) / t
    vz = (z_land - z0 - 0.5 * g * t * t) / t

    pos = np.array([x0, y0, z0], dtype=np.float32)
    vel = np.array([vx, vy, vz], dtype=np.float32)
    return pos, vel
