from pathlib import Path
from stable_baselines3 import PPO
from env import AnymalFlatEnv
import wandb

def main():
    base_dir=Path(__file__).resolve().parent
    config={
        "lateral_speed_range": (-0.3, 0.3),
        "yaw_rate_range": (-0.5, 0.5),
        "n_steps":1024,
        "batch_size":64,
        "learning_rate":3e-4,
        "seed":42,
        "total_timesteps":2048000,
    }
    with wandb.init(
        project="anymal-c-flat-ground",
        name="ppo-baseline-seed42",
        config=config,
        sync_tensorboard=True,
        dir=str(base_dir),
    )as run:
        env = AnymalFlatEnv(
            lateral_speed_range=config["lateral_speed_range"],
            yaw_rate_range=config["yaw_rate_range"],
        )
        try:
            model=PPO(
                policy="MlpPolicy",
                env=env,
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                seed=config["seed"],
                tensorboard_log=str(base_dir / "runs" / run.id),
                verbose=1,
            )
            model.learn(total_timesteps=config["total_timesteps"])

            checkpoint_dir=base_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint=checkpoint_dir / f"ppo_baseline_seed42_{run.id}.zip"
            model.save(str(checkpoint))
            print(f"策略已保存到{checkpoint}")
        finally:
            env.close()

if __name__=="__main__":
    main()
