from pathlib import Path
from time import perf_counter, sleep

import mujoco.viewer
from stable_baselines3 import PPO

from env import AnymalFlatEnv


def main():
    checkpoint = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "ppo_anymal_flat.zip"
    )
    policy = PPO.load(str(checkpoint), device="cpu")
    env = AnymalFlatEnv()

    try:
        obs, _ = env.reset()
        control_dt = env.model.opt.timestep * env.frame_skip

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                start = perf_counter()

                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.sync()

                if terminated or truncated:
                    result = "跌倒" if terminated else "跑满一回合"
                    print(f"{result}，仿真时间 {info['time']:.2f} 秒")
                    sleep(0.5)
                    obs, _ = env.reset()

                sleep(max(0, control_dt - (perf_counter() - start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()