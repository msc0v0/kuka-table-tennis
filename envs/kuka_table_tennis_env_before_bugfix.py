import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

from envs.obs import build_obs
from data.incoming_sampler import sample_incoming

# MuJoCo XML 嵌入
XML_STRING = """
<mujoco model="table_tennis">
    <include file="assets/iiwa14_gantry.xml"/>
    <compiler angle="radian" />
    <option timestep="0.01" gravity="0 0 -9.81" />
    <worldbody>
        <!-- Ground -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="vis" pos="100 0 1.26" quat="0 0.7068252 0 0.7073883">
            <geom name="cylinder" type="cylinder" pos="0.2 0 0" size="0.10 0.015" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
            <geom name="handle" type="cylinder" pos="0.05 0 0" size="0.02 0.05" quat="0 0.7068252 0 0.7073883" rgba="0 0 1 0.3" contype="0" conaffinity="0"/>
        </body>
        <!-- Table -->
        <body name="table" pos="1.5 0 0.64">
            <geom name="table_top" type="box" size="1.37 0.7625 0.02" rgba="0 0 1 1" friction="0.2 0.2 0.1" solref="0.04 0.1" solimp="0.9 0.999 0.001" />
        </body>
        <body name="net" pos="1.5 0 0.7" euler="0 0 0">
            <geom name="net_geom" type="box" size="0.01 0.7625 0.08" rgba="1 1 1 1" friction="0 0 0" contype="0" conaffinity="0" />
        </body>
        <!-- Ball -->
        <body name="ball" pos="2 -0.7 1">
            <freejoint name="haha"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1" 
                  friction="0.001 0.001 0.001" solref="0.04 0.05" solimp="0.9 0.999 0.001" />
        </body>
    </worldbody>
</mujoco>
"""

class KukaTableTennisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: dict, shot_sampler=None, seed: int = 0, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.sampler = shot_sampler  # 保留兼容性，但不使用
        self.rng = np.random.default_rng(seed)

        xml_path = cfg["mj"].get("xml_path", None)
        if xml_path:
            self.model = mj.MjModel.from_xml_path(xml_path)
        else:
            self.model = mj.MjModel.from_xml_string(XML_STRING)

        self.data = mj.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.act_dim = 9
        self.action_space = spaces.Box(low=-0.15, high=0.15, shape=(self.act_dim,), dtype=np.float32)

        H = int(cfg["obs"]["history"])
        self.H = H
        self.prev_actions = np.zeros((H, self.act_dim), dtype=np.float32)

        include_relative = bool(cfg["obs"]["include_relative"])
        base = 9 + 9 + 3 + 3
        rel = 6 if include_relative else 0
        obs_dim = base + rel + H * self.act_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.ball_body = self.data.body("ball")
        self.racket_body = self.data.body("tennis_racket")

        tcfg = cfg["table"]
        self.table_center_x = float(tcfg["table_center_x"])
        self.table_half_len = float(tcfg["table_half_length"])
        self.table_half_w = float(tcfg["table_half_width"])
        self.table_z = float(tcfg["table_height_z"] + tcfg["table_top_thickness"])
        self.net_x = float(tcfg["net_x"])
        self.net_h = float(tcfg["net_height"])

        self.max_steps = int(cfg["episode"]["max_steps"])

    def _reset_ball(self, pos3: np.ndarray, vel3: np.ndarray):
        self.data.qpos[-7:-4] = pos3
        self.data.qvel[-6:-3] = vel3

    def _get_ball_state(self):
        pos = np.array(self.data.qpos[-7:-4], dtype=np.float32)
        vel = np.array(self.data.qvel[-6:-3], dtype=np.float32)
        return pos, vel

    def _detect_contacts(self):
        hit_racket = False
        hit_table = False
        hit_net = False

        ncon = int(self.data.ncon)
        if ncon <= 0:
            return hit_racket, hit_table, hit_net

        for i in range(ncon):
            c = self.data.contact[i]
            g1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, c.geom2)
            if g1 is None or g2 is None:
                continue

            pair = {g1, g2}
            if "ball_geom" in pair and "racket" in pair:
                hit_racket = True
            if "ball_geom" in pair and "table_top" in pair:
                hit_table = True
            if "ball_geom" in pair and "net_geom" in pair:
                hit_net = True

        return hit_racket, hit_table, hit_net

    def _is_self_side(self, x: float) -> bool:
        return x < self.net_x if self.self_side_is_x_less else x > self.net_x

    def _is_opponent_side(self, x: float) -> bool:
        return not self._is_self_side(x)

    def _is_on_table_xy(self, x: float, y: float) -> bool:
        x_min = self.table_center_x - self.table_half_len
        x_max = self.table_center_x + self.table_half_len
        return (x_min <= x <= x_max) and (abs(y) <= self.table_half_w)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)

        # 重置所有状
        self.step_count = 0
        self.hit_done = False
        self.cleared_net = False
        self.landed_any = False
        self.incoming_bounced = False
        self.landed_opponent = False
        self.landed_opponent_valid = False  # 是否落在对方桌面范围内
        self.success = False
        self.prev_ball_pos = None
        self.prev_actions[...] = 0.0

        # 先 forward 一次，确定球拍位置，判断己方半台
        mj.mj_forward(self.model, self.data)
        racket_x = float(self.racket_body.xpos[0])
        self.self_side_is_x_less = (racket_x < self.net_x)

        # 使用 incoming_sampler 采样来球
        pos0, vel0 = sample_incoming(
            rng=self.rng,
            net_x=self.net_x,
            table_center_x=self.table_center_x,
            table_half_length=self.table_half_len,
            table_half_width=self.table_half_w,
            table_z=self.table_z,
            self_side_is_x_less=self.self_side_is_x_less,
        )
        self._reset_ball(pos0, vel0)

        mj.mj_forward(self.model, self.data)
        ball_pos, ball_vel = self._get_ball_state()
        self.prev_ball_pos = ball_pos.copy()

        racket_pos = np.array(self.racket_body.xpos, dtype=np.float32)
        racket_vel = np.array(self.racket_body.cvel[:3], dtype=np.float32)

        obs = build_obs(
            qpos9=self.data.qpos[:9].copy(),
            qvel9=self.data.qvel[:9].copy(),
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            racket_pos=racket_pos,
            racket_vel=racket_vel,
            prev_actions=self.prev_actions,
            include_relative=bool(self.cfg["obs"]["include_relative"]),
        )
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.prev_actions[:-1] = self.prev_actions[1:]
        self.prev_actions[-1] = action

        self.data.qpos[0] += 3.0 * float(action[0]) / 60.0
        self.data.qpos[1] += 3.0 * float(action[1]) / 60.0
        self.data.qpos[0] = np.clip(self.data.qpos[0], -1.0, 0.0)
        self.data.qpos[1] = np.clip(self.data.qpos[1], -1.0, 1.0)

        self.data.ctrl[:] = (action[2:] + self.data.qpos[2:9]).astype(np.float32)

        mj.mj_step(self.model, self.data)

        self.step_count += 1

        ball_pos, ball_vel = self._get_ball_state()
        racket_pos = np.array(self.racket_body.xpos, dtype=np.float32)
        racket_vel = np.array(self.racket_body.cvel[:3], dtype=np.float32)

        hit_racket, hit_table, hit_net = self._detect_contacts()

        # 1. 触球事件
        if hit_racket and (not self.hit_done):
            self.hit_done = True

        ep_hit = False 
        if self.hit_done and (not self.cleared_net) and (self.prev_ball_pos is not None):
            x0, z0 = self.prev_ball_pos[0], self.prev_ball_pos[2]
            x1, z1 = ball_pos[0], ball_pos[2]
            # 从己方到对方穿越 net_x，且高度高于球网
            if self._is_self_side(x0) and self._is_opponent_side(x1) and max(z0, z1) > self.net_h:
                self.cleared_net = True

        # 3. 落台事件
        if hit_table and (not self.landed_any):
            self.landed_any = True
            if not self.hit_done:
                # 来球第一次落台（应该在己方）
                self.incoming_bounced = True
            else:
                # 击球后第一次落台
                if self._is_opponent_side(ball_pos[0]) and self._is_on_table_xy(ball_pos[0], ball_pos[1]):
                    self.landed_opponent = True
                self.landed_opponent_valid = True

        # 4. 成功判定：必须 hit_done + cleared_net + landed_opponent
        if self.hit_done and self.cleared_net and self.landed_opponent:
            self.success = True

        # 奖励计算
        rcfg = self.cfg["reward"]
        reward = 0.0

        reward += float(rcfg["alive"])
        reward -= float(rcfg["action_l2"]) * float(np.sum(action * action))

        if not self.hit_done:
            dist = float(np.linalg.norm(ball_pos - racket_pos))
            reward -= float(rcfg["racket_ball_dist"]) * dist

        if hit_racket and (not self.hit_done):
            reward += float(rcfg["hit"])

        if hit_net:
            reward += float(rcfg["net_fault"])

        if self.cleared_net and not hasattr(self, '_cleared_once'):
            self._cleared_once = True
            reward += float(rcfg["clear_net"])

        if self.landed_opponent and not hasattr(self, '_landed_once'):
            self._landed_once = True
            reward += float(rcfg["land_opponent"])

        if self.hit_done:
            reward += float(rcfg["forward_vx"]) * float(ball_vel[0])

        # 终止条件
        terminated = False
        truncated = False

        if self.success:
            terminated = True

        if hit_net:
            terminated = True

        if self.landed_any and self.hit_done and (not self.landed_opponent):
            # 击球后落台但不在对方
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        obs = build_obs(
            qpos9=self.data.qpos[:9].copy(),
            qvel9=self.data.qvel[:9].copy(),
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            racket_pos=racket_pos,
            racket_vel=racket_vel,
            prev_actions=self.prev_actions,
            include_relative=bool(self.cfg["obs"]["include_relative"]),
        )

        info = {
            "hit": self.hit_done,
            "cleared_net": self.cleared_net,
            "landed": self.landed_any,
            "success": self.success,
            "ball_pos": ball_pos.copy(),
            "landed_opponent": self.landed_opponent_valid,
        }

        self.prev_ball_pos = ball_pos.copy()

        return obs, float(reward), terminated, truncated, info
