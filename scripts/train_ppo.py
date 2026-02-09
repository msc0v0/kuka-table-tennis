import argparse
import yaml
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.kuka_table_tennis_env import KukaTableTennisEnv


class DetailedInfoLogger(BaseCallback):
    """
    以 episode 事件统计（对 stage1 更稳定）：
      - hit_rate: 本局是否出现过 hit_event=True（首次触球事件）
      - 其余指标在 Stage1 env 里会固定为 False（避免统计鬼影）
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

        self.episode_hits = []
        self.episode_clears = []
        self.episode_lands = []
        self.episode_land_opponents = []
        self.episode_successes = []

        self.current_ep_hit = False
        self.current_ep_clear = False
        self.current_ep_land = False
        self.current_ep_land_opponent = False
        self.current_ep_success = False

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            # HIT
            if "hit_event" in info:
                self.current_ep_hit = self.current_ep_hit or bool(info["hit_event"])
            elif "hit" in info:
                self.current_ep_hit = self.current_ep_hit or bool(info["hit"])

            # 其它（stage1 env 里一般都固定 False，但保留兼容）
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

            if done:
                self.episode_hits.append(1.0 if self.current_ep_hit else 0.0)
                self.episode_clears.append(1.0 if self.current_ep_clear else 0.0)
                self.episode_lands.append(1.0 if self.current_ep_land else 0.0)
                self.episode_land_opponents.append(1.0 if self.current_ep_land_opponent else 0.0)
                self.episode_successes.append(1.0 if self.current_ep_success else 0.0)

                self.current_ep_hit = False
                self.current_ep_clear = False
                self.current_ep_land = False
                self.current_ep_land_opponent = False
                self.current_ep_success = False

        if self.num_timesteps % self.log_freq == 0 and len(self.episode_hits) > 0:
            recent_hits = self.episode_hits[-100:]
            recent_clears = self.episode_clears[-100:]
            recent_lands = self.episode_lands[-100:]
            recent_land_opponents = self.episode_land_opponents[-100:]
            recent_successes = self.episode_successes[-100:]

            hit_rate = float(np.mean(recent_hits)) if recent_hits else 0.0
            clear_rate = float(np.mean(recent_clears)) if recent_clears else 0.0
            land_rate = float(np.mean(recent_lands)) if recent_lands else 0.0
            land_opponent_rate = float(np.mean(recent_land_opponents)) if recent_land_opponents else 0.0
            success_rate = float(np.mean(recent_successes)) if recent_successes else 0.0

            print(f"\n[Step {self.num_timesteps}] Detailed Stats (last 100 episodes):")
            print(f"  Hit Rate: {hit_rate*100:.1f}%")
            print(f"  Clear Net Rate: {clear_rate*100:.1f}%")
            print(f"  Land Rate: {land_rate*100:.1f}%")
            print(f"  Land Opponent Rate: {land_opponent_rate*100:.1f}%")
            print(f"  Success Rate: {success_rate*100:.1f}%")

            self.logger.record("rollout/hit_rate", hit_rate)
            self.logger.record("rollout/clear_net_rate", clear_rate)
            self.logger.record("rollout/land_rate", land_rate)
            self.logger.record("rollout/land_opponent_rate", land_opponent_rate)
            self.logger.record("rollout/success_rate", success_rate)

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

    n_envs = int(ppo_cfg.get("n_envs", 16))  # Stage1 建议 16
    env_fns = [make_env(env_cfg, args.seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # Stage1：事件奖励更稳定，建议 norm_reward=False
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    policy_kwargs = ppo_cfg.get("policy_kwargs", {})
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
        ent_coef=ppo_cfg.get("ent_coef", 0.01),     # Stage1 给一点探索
        vf_coef=ppo_cfg.get("vf_coef", 0.5),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(save_dir / "tensorboard"),

        # 关键：别用 0.02 这么小，太容易 early stop
        # 你也可以直接删掉这一行（完全关闭 KL 保险丝）
        target_kl=float(ppo_cfg.get("target_kl", 0.10)),
    )

    new_logger = configure(str(save_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(ppo_cfg.get("save_freq", 50_000)),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_model",
    )
    detailed_logger = DetailedInfoLogger(log_freq=int(ppo_cfg.get("log_freq", 1000)))

    total_timesteps = int(ppo_cfg.get("total_timesteps", 2_000_000))

    print(f"\n{'='*60}")
    print(f"Starting PPO Training (Stage1 Hit)")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {save_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, detailed_logger],
        progress_bar=True,
    )

    model.save(str(save_dir / "final_model"))
    vec_env.save(str(save_dir / "final_vec_normalize.pkl"))

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Model saved to: {save_dir / 'final_model'}")
    print(f"VecNormalize saved to: {save_dir / 'final_vec_normalize.pkl'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
