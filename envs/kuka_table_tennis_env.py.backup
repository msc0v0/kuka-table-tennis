import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

from envs.obs import build_obs
from data.hit_friendly_sampler import sample_hit_friendly

# =========================
# MuJoCo XML 嵌入
# =========================
XML_STRING = """
<mujoco model="table_tennis">
    <include file="assets/iiwa14_gantry.xml"/>
    <compiler angle="radian" />
    <option timestep="0.01" gravity="0 0 -9.81" />
    <worldbody>
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

        <body name="table" pos="1.5 0 0.64">
            <geom name="table_top" type="box" size="1.37 0.7625 0.02"
                  rgba="0 0 1 1" friction="0.2 0.2 0.1"
                  solref="0.04 0.1" solimp="0.9 0.999 0.001" />
        </body>

        <body name="net" pos="1.5 0 0.7" euler="0 0 0">
            <geom name="net_geom" type="box" size="0.01 0.7625 0.08"
                  rgba="1 1 1 1" friction="0 0 0" contype="0" conaffinity="0" />
        </body>

        <body name="ball" pos="2 -0.7 1">
            <freejoint name="ball_free"/>
            <geom name="ball_geom" type="sphere" size="0.02" mass="0.0027" rgba="1 0.5 0 1"
                  friction="0.001 0.001 0.001" solref="0.04 0.05" solimp="0.9 0.999 0.001" />
        </body>
    </worldbody>
</mujoco>
"""

# 防止球飞太远浪费 rollout
OOB_X_MIN, OOB_X_MAX = -2.0, 5.0
OOB_Y_MAX = 2.0
OOB_Z_MIN, OOB_Z_MAX = 0.10, 3.0


class KukaTableTennisEnv(gym.Env):
    """
    Stage1: 只训练“碰到球拍 (hit)”
    - episode 在 hit_event 时立即终止
    - reward 强化靠近球、相对速度对齐、hit 大奖励
    - 不统计过网/落台/成功，避免日志混乱
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, cfg: dict, shot_sampler=None, seed: int = 0, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        xml_path = cfg["mj"].get("xml_path", None)
        if xml_path:
            self.model = mj.MjModel.from_xml_path(xml_path)
        else:
            self.model = mj.MjModel.from_xml_string(XML_STRING)

        self.data = mj.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        # action: 2 gantry + 7 joints
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

        self.max_steps = int(cfg["episode"]["max_steps"])

        self._init_episode_state()

    def _init_episode_state(self):
        self.step_count = 0
        self.hit_done = False
        self.prev_ball_pos = None
        self.prev_dist = None
        self.self_side_is_x_less = True

    def _reset_ball(self, pos3: np.ndarray, vel3: np.ndarray):
        self.data.qpos[-7:-4] = pos3
        self.data.qvel[-6:-3] = vel3

    def _get_ball_state(self):
        pos = np.array(self.data.qpos[-7:-4], dtype=np.float32)
        vel = np.array(self.data.qvel[-6:-3], dtype=np.float32)
        return pos, vel

    @staticmethod
    def _is_racket_geom(name: str) -> bool:
        n = name.lower()
        return ("racket" in n) or ("paddle" in n) or ("bat" in n)

    def _detect_hit_racket(self) -> bool:
        ncon = int(self.data.ncon)
        if ncon <= 0:
            return False
        for i in range(ncon):
            c = self.data.contact[i]
            g1 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, c.geom2)
            if g1 is None or g2 is None:
                continue
            if ("ball_geom" in (g1, g2)) and (self._is_racket_geom(g1) or self._is_racket_geom(g2)):
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)
        self._init_episode_state()
        self.prev_actions[...] = 0.0

        # forward 一次：获取球拍位置，判断己方在哪侧（给 sampler 用）
        mj.mj_forward(self.model, self.data)
        racket_x = float(self.racket_body.xpos[0])
        self.self_side_is_x_less = (racket_x < self.net_x)

        # 采样“必经球拍附近”的来球（你自己保证这个 sampler 真的 hit-friendly）
        pos0, vel0 = sample_hit_friendly(
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

        # history
        self.prev_actions[:-1] = self.prev_actions[1:]
        self.prev_actions[-1] = action

        # apply gantry
        self.data.qpos[0] += 3.0 * float(action[0]) / 60.0
        self.data.qpos[1] += 3.0 * float(action[1]) / 60.0
        self.data.qpos[0] = np.clip(self.data.qpos[0], -1.0, 0.0)
        self.data.qpos[1] = np.clip(self.data.qpos[1], -1.0, 1.0)

        # joints: delta-pos
        self.data.ctrl[:] = (action[2:] + self.data.qpos[2:9]).astype(np.float32)

        mj.mj_step(self.model, self.data)
        self.step_count += 1

        ball_pos, ball_vel = self._get_ball_state()
        racket_pos = np.array(self.racket_body.xpos, dtype=np.float32)
        racket_vel = np.array(self.racket_body.cvel[:3], dtype=np.float32)

        hit_racket = self._detect_hit_racket()
        hit_event = False
        if hit_racket and (not self.hit_done):
            self.hit_done = True
            hit_event = True

        # -------- reward: Stage1 only --------
        rcfg = self.cfg["reward"]
        reward = 0.0

        # alive & action regularization
        reward += float(rcfg.get("alive", 0.0))
        reward -= float(rcfg.get("action_l2", 0.0)) * float(np.sum(action * action))

        # distance shaping (strong!)
        dist = float(np.linalg.norm(ball_pos - racket_pos))
        reward -= float(rcfg.get("racket_ball_dist", 0.08)) * dist

        # progress shaping (helps a lot)
        if self.prev_dist is not None:
            reward += float(rcfg.get("racket_ball_prog", 1.0)) * (self.prev_dist - dist)
        self.prev_dist = dist

        # velocity alignment: encourage moving along line-of-sight to ball
        rel = (ball_pos - racket_pos)
        rel_dir = rel / (float(np.linalg.norm(rel)) + 1e-6)
        reward += float(rcfg.get("racket_vel_align", 0.10)) * float(np.dot(racket_vel, rel_dir))

        # hit bonus (big)
        if hit_event:
            reward += float(rcfg.get("hit", 10.0))

        # -------- termination: Stage1 --------
        terminated = False
        truncated = False

        if hit_event:
            terminated = True  # 关键：永远不会被后面覆盖

        # out-of-bounds / too long
        x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        if (x < OOB_X_MIN) or (x > OOB_X_MAX) or (abs(y) > OOB_Y_MAX) or (z < OOB_Z_MIN) or (z > OOB_Z_MAX):
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

        # 兼容 logger：只看 hit，其它全部固定 False，避免“清网>命中”这种统计鬼影
        info = {
            "hit": self.hit_done,
            "hit_event": hit_event,
            "cleared_net": False,
            "landed": False,
            "landed_opponent": False,
            "success": False,
            "ball_pos": ball_pos.copy(),
        }
        self.prev_ball_pos = ball_pos.copy()
        return obs, float(reward), terminated, truncated, info
