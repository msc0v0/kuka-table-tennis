import argparse
import yaml
import numpy as np
from pathlib import Path
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.kuka_table_tennis_env import KukaTableTennisEnv


class DetailedInfoLogger(BaseCallback):
    """
    以 episode 事件统计（对 stage1 更稳定）：
      - hit_rate: 本局是否出现过 hit_event=True（首次触球事件）
      - 同时统计 term_reason 分布（hit/oob/timeout/unknown）
      - 打印 avg_dist_to_target / avg_ep_len / avg_curriculum_radius 等诊断项
    """
    def __init__(self, log_freq=1000, window_episodes=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self.window = int(window_episodes)

        # per-episode history
        self.episode_hits = []
        self.episode_clears = []
        self.episode_lands = []
        self.episode_land_opponents = []
        self.episode_successes = []

        self.episode_term_reasons = []      # str
        self.episode_lengths = []           # int
        self.episode_avg_dists = []         # float
        self.episode_avg_radius = []        # float

        # current episode accumulators
        self.current_ep_hit = False
        self.current_ep_clear = False
        self.current_ep_land = False
        self.current_ep_land_opponent = False
        self.current_ep_success = False

        self._cur_len = 0
        self._cur_dist_sum = 0.0
        self._cur_dist_n = 0
        self._cur_rad_sum = 0.0
        self._cur_rad_n = 0
        self._cur_term_reason = None

    def _reset_current_episode_acc(self):
        self.current_ep_hit = False
        self.current_ep_clear = False
        self.current_ep_land = False
        self.current_ep_land_opponent = False
        self.current_ep_success = False

        self._cur_len = 0
        self._cur_dist_sum = 0.0
        self._cur_dist_n = 0
        self._cur_rad_sum = 0.0
        self._cur_rad_n = 0
        self._cur_term_reason = None

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # SB3 对 VecEnv：每次 step 都是 n_envs 条 info/done
        for info, done in zip(infos, dones):
            # per-step accumulators
            self._cur_len += 1

            # dist_to_target（如果 env 提供）
            if isinstance(info, dict) and ("dist_to_target" in info):
                try:
                    d = float(info["dist_to_target"])
                    if np.isfinite(d):
                        self._cur_dist_sum += d
                        self._cur_dist_n += 1
                except Exception:
                    pass

            # curriculum radius（如果 env 提供）
            if isinstance(info, dict) and ("curriculum_radius" in info):
                try:
                    r = float(info["curriculum_radius"])
                    if np.isfinite(r):
                        self._cur_rad_sum += r
                        self._cur_rad_n += 1
                except Exception:
                    pass

            # HIT（episode-level）
            if isinstance(info, dict):
                if "hit_event" in info:
                    self.current_ep_hit = self.current_ep_hit or bool(info["hit_event"])
                elif "hit" in info:
                    self.current_ep_hit = self.current_ep_hit or bool(info["hit"])

                # 其它（stage1 env 通常固定 False，但保留兼容）
                if "cleared_net" in info:
                    self.current_ep_clear = self.current_ep_clear or bool(info["cleared_net"])

                if "landed_event" in info:
                    self.current_ep_land = self.current_ep_land or bool(info["landed_event"])
                elif "landed" in info:
                    self.current_ep_land = self.current_ep_land or bool(info["landed"])

                if ("landed_side" in info) and ("landed_on_table" in info):
                    landed_op = (info["landed_side"] == "opponent") and bool(info["landed_on_table"])
                    self.current_ep_land_opponent = self.current_ep_land_opponent or landed_op
                elif "landed_opponent" in info:
                    self.current_ep_land_opponent = self.current_ep_land_opponent or bool(info["landed_opponent"])

                if "success" in info:
                    self.current_ep_success = self.current_ep_success or bool(info["success"])

                # term_reason（通常只有在 done 的最后一步才有意义，但这里先缓存）
                if "term_reason" in info and info["term_reason"] is not None:
                    self._cur_term_reason = str(info["term_reason"])

            # episode ended
            if done:
                self.episode_hits.append(1.0 if self.current_ep_hit else 0.0)
                self.episode_clears.append(1.0 if self.current_ep_clear else 0.0)
                self.episode_lands.append(1.0 if self.current_ep_land else 0.0)
                self.episode_land_opponents.append(1.0 if self.current_ep_land_opponent else 0.0)
                self.episode_successes.append(1.0 if self.current_ep_success else 0.0)

                # term reason
                tr = self._cur_term_reason if self._cur_term_reason is not None else "unknown"
                self.episode_term_reasons.append(tr)

                # episode length
                self.episode_lengths.append(int(self._cur_len))

                # episode avg dist
                if self._cur_dist_n > 0:
                    self.episode_avg_dists.append(float(self._cur_dist_sum / self._cur_dist_n))
                else:
                    self.episode_avg_dists.append(float("nan"))

                # episode avg curriculum radius
                if self._cur_rad_n > 0:
                    self.episode_avg_radius.append(float(self._cur_rad_sum / self._cur_rad_n))
                else:
                    self.episode_avg_radius.append(float("nan"))

                # reset current accumulators
                self._reset_current_episode_acc()

        # periodic print/log
        if (self.num_timesteps % self.log_freq == 0) and (len(self.episode_hits) > 0):
            W = self.window
            recent_hits = self.episode_hits[-W:]
            recent_clears = self.episode_clears[-W:]
            recent_lands = self.episode_lands[-W:]
            recent_land_opponents = self.episode_land_opponents[-W:]
            recent_successes = self.episode_successes[-W:]

            hit_rate = float(np.mean(recent_hits)) if recent_hits else 0.0
            clear_rate = float(np.mean(recent_clears)) if recent_clears else 0.0
            land_rate = float(np.mean(recent_lands)) if recent_lands else 0.0
            land_opponent_rate = float(np.mean(recent_land_opponents)) if recent_land_opponents else 0.0
            success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0

            # term reason distribution
            recent_reasons = self.episode_term_reasons[-W:]
            reason_counts = Counter(recent_reasons)

            # episode len / dist / radius diagnostics
            recent_lens = self.episode_lengths[-W:]
            avg_ep_len = float(np.mean(recent_lens)) if recent_lens else float("nan")

            recent_dists = [d for d in self.episode_avg_dists[-W:] if np.isfinite(d)]
            avg_dist = float(np.mean(recent_dists)) if len(recent_dists) > 0 else float("nan")

            recent_rads = [r for r in self.episode_avg_radius[-W:] if np.isfinite(r)]
            avg_rad = float(np.mean(recent_rads)) if len(recent_rads) > 0 else float("nan")

            print(f"\n[Step {self.num_timesteps}] Detailed Stats (last {min(W, len(self.episode_hits))} episodes):")
            print(f"  Hit Rate: {hit_rate*100:.1f}%")
            print(f"  Clear Net Rate: {clear_rate*100:.1f}%")
            print(f"  Land Rate: {land_rate*100:.1f}%")
            print(f"  Land Opponent Rate: {land_opponent_rate*100:.1f}%")
            print(f"  Success Rate: {success_rate*100:.1f}%")
            print(f"  Avg Episode Length: {avg_ep_len:.1f} steps")
            if np.isfinite(avg_dist):
                print(f"  Avg Dist-To-Target: {avg_dist:.4f} m")
            if np.isfinite(avg_rad):
                print(f"  Avg Curriculum Radius: {avg_rad:.4f} m")

            # pretty term reason print
            # 确保 hit/oob/timeout/unknown 都显示
            for k in ["hit", "oob", "timeout", "unknown"]:
                if k not in reason_counts:
                    reason_counts[k] = 0
            total = len(recent_reasons)
            print("  Termination Reasons:")
            for k in ["hit", "oob", "timeout", "unknown"]:
                v = reason_counts[k]
                print(f"    {k:8s}: {v:4d}  ({(v / max(total,1))*100:5.1f}%)")

            # TB logger
            self.logger.record("rollout/hit_rate", hit_rate)
            self.logger.record("rollout/clear_net_rate", clear_rate)
            self.logger.record("rollout/land_rate", land_rate)
            self.logger.record("rollout/land_opponent_rate", land_opponent_rate)
            self.logger.record("rollout/success_rate", success_rate)
            self.logger.record("rollout/avg_episode_len", avg_ep_len)
            if np.isfinite(avg_dist):
                self.logger.record("rollout/avg_dist_to_target", avg_dist)
            if np.isfinite(avg_rad):
                self.logger.record("rollout/avg_curriculum_radius", avg_rad)

            # termination reason rates
            self.logger.record("rollout/term_hit_rate", reason_counts["hit"] / max(total, 1))
            self.logger.record("rollout/term_oob_rate", reason_counts["oob"] / max(total, 1))
            self.logger.record("rollout/term_timeout_rate", reason_counts["timeout"] / max(total, 1))
            self.logger.record("rollout/term_unknown_rate", reason_counts["unknown"] / max(total, 1))

        return True


def make_env(env_cfg, seed):
    def _init():
        env = KukaTableTennisEnv(cfg=env_cfg, shot_sampler=None, seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, required=True)
    parser.add_argument("--ppo_cfg", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="logs/ppo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.env_cfg, "r") as f:
        env_cfg = yaml.safe_load(f)
    with open(args.ppo_cfg, "r") as f:
        ppo_cfg = yaml.safe_load(f)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(ppo_cfg.get("n_envs", 16))
    env_fns = [make_env(env_cfg, args.seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Stage1：事件奖励更稳定，建议 norm_reward=False
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = ppo_cfg.get("policy_kwargs", {})

    # target_kl：你可以直接删掉让它完全不 early stop
    target_kl = ppo_cfg.get("target_kl", 0.10)
    if target_kl is not None:
        target_kl = float(target_kl)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        n_steps=ppo_cfg.get("n_steps", 2048),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        ent_coef=ppo_cfg.get("ent_coef", 0.01),
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(save_dir / "tensorboard"),
        target_kl=target_kl,
    )

    new_logger = configure(str(save_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(ppo_cfg.get("save_freq", 50_000)),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_model",
    )

    detailed_logger = DetailedInfoLogger(
        log_freq=int(ppo_cfg.get("log_freq", 1000)),
        window_episodes=int(ppo_cfg.get("window_episodes", 100)),
    )

    total_timesteps = int(ppo_cfg.get("total_timesteps", 2_000_000))

    print(f"\n{'='*60}")
    print("Starting PPO Training (Stage1 Hit)")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {save_dir}")
    print(f"Seed: {args.seed}")
    print(f"target_kl: {target_kl}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, detailed_logger],
        progress_bar=True,
    )

    model.save(str(save_dir / "final_model"))
    vec_env.save(str(save_dir / "final_vec_normalize.pkl"))

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Model saved to: {save_dir / 'final_model'}")
    print(f"VecNormalize saved to: {save_dir / 'final_vec_normalize.pkl'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
