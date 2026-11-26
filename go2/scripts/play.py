import argparse
from json import load
import os
import numpy as np
import mujoco
import mujoco.viewer
import sys
import time
from go2.envs.commands import CommandRanges
from go2.envs.go2_env import Go2Env
import pickle
from pathlib import Path
from go2.envs.gaits import GAITS_TYPES
from go2.scripts.evaluate import load_policy

GO2_ROOT=Path(__file__).resolve().parents[1]
PROJ_ROOT=Path(__file__).resolve().parents[2]
GAIT_KEYS={
    ord("1"):"trot",
    ord("2"):"pace",
    ord("3"):"bound",
    ord("4"):"walk",
    ord("5"):"pronk",
}

# 每次按键时对应的控制改变多少
VEL_STEP = 0.1
YAW_STEP = 0.2
FREQ_STEP = 0.1
HEIGHT_STEP = 0.01


class Teleop:
    def __init__(self,env:Go2Env):
        self.env=env
        self.vx=0.0
        self.vy=0.0
        self.yaw=0.0
        self.gait="trot"
        self.freq=2.0
        self.height=0.30
        self.reset_requested=False

    def apply(self):
        self.env.set_command(
            lin_vel_x=self.vx,
            lin_vel_y=self.vy,
            ang_vel_yaw=self.yaw,
            gait="stand" if self._is_stopped() else self.gait,
            gait_frequency=self.freq,
            base_height=self.height,
        )

    def _is_stopped(self):
        return abs(self.vx)<0.05 and abs(self.vy)<0.05 and abs(self.yaw)<0.05

    def describe(self):
        return(
            "vx=%+.2f  vy=%+.2f  yaw=%+.2f  gait=%-6s  freq=%.1f  h=%.2f" % (self.vx,self.vy,self.yaw,"stand" if self._is_stopped() else self.gait,self.freq,self.height)
        )

    def on_key(self,keycode):
        if keycode == ord("W"):
            self.vx = min(self.vx + VEL_STEP, 2.0)
        elif keycode == ord("S"):
            self.vx = max(self.vx - VEL_STEP, -1.5)
        elif keycode == ord("A"):
            self.vy = min(self.vy + VEL_STEP, 1.0)
        elif keycode == ord("D"):
            self.vy = max(self.vy - VEL_STEP, -1.0)
        elif keycode == ord("Q"):
            self.yaw = min(self.yaw + YAW_STEP, 2.0)
        elif keycode == ord("E"):
            self.yaw = max(self.yaw - YAW_STEP, -2.0)
        elif keycode == ord(" "):
            self.vx = self.vy = self.yaw = 0.0
        elif keycode in GAIT_KEYS:
            self.gait = GAIT_KEYS[keycode]
        elif keycode == ord("["):
            self.freq = max(self.freq - FREQ_STEP, 0.8)
        elif keycode == ord("]"):
            self.freq = min(self.freq + FREQ_STEP, 4.0)
        elif keycode == ord("-"):
            self.height = max(self.height - HEIGHT_STEP, 0.22)
        elif keycode == ord("="):
            self.height = min(self.height + HEIGHT_STEP, 0.38)
        elif keycode == ord("R"):
            self.reset_requested = True
        elif keycode == ord("P"):
            print(self.describe())
            return
        else:
            return
        self.apply()
        print("\r" + self.describe() + "   ", end="", flush=True)

    def load_policy(model_path:str,vecnormalize_path:str,device="cpu"):
        if model_path.endswith(".npz"):
            from go2.envs.brax_policy import BraxPolicy

            policy=BraxPolicy.load(model_path)
            print("load brax policy from %s (%s)" % (model_path,",".join("%s=%s" % kv for kv in sorted(policy.meta.items()))))
            return policy
        from stable_baselines3 import PPO,SAC,TD3

        algo=PPO
        for name,cls in (("sac",SAC),("td3",TD3),("ppo",PPO)):
            if name in os.path.basename(model_path).lower():
                algo=cls
                break
        model=algo.load(model_path,device=device)

        obs_rms,clip_obs,epsilon=None,10.0,1e-8
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            with open(vecnormalize_path,"rb") as f:
                vec=pickle.load(f)
            obs_rms,clip_obs,epsilon=vec.obs_rms,vec.clip_obs,vec.epsilon
        else:
            print("No VecNormalize statistic is provided thus the results will be meaningless unless the policy was trained without normalisation")

        def predict(obs):
            x=obs
            if obs_rms is not None:
                x=np.clip((obs-obs_rms.mean)/np.sqrt(obs_rms.var+epsilon),-clip_obs,clip_obs).astype(np.float32)
            action,_=model.predict(x,deterministic=True)# 策略不随机采样动作而是选择动作分布的中心值
            return action
        return predict

def main():
    p=argparse.ArgumentParser(description="Manipulate the model in mujoco viewer")
    p.add_argument("--model",default=GO2_ROOT/"models"/"go2_ppo_final.zip")
    p.add_argument("--vec-normalize",default=None)
    p.add_argument("--xml",default=PROJ_ROOT/"mujoco_menagerie"/"unitree_go2"/"scene.xml")
    p.add_argument("--random", action="store_true")
    p.add_argument("--realtime", type=float, default=1.0)
    args=p.parse_args()

    env=Go2Env(
        xml_path=args.xml,
        randomize_dynamics=False,
        push_enable=False,
        max_episode_steps=10**9,
        command_ranges=CommandRanges(),
        command_resample_interval_s=0.0
    )
    if args.random:
        def predict(_obs):
            return env.action_space.sample()
    else:
        vecnorm=args.vec_normalize
        if vecnorm is None:
            guess = os.path.splitext(args.model)[0].replace("_final", "")
            guess += "_vecnormalize.pkl"
            vecnorm = guess if os.path.exists(guess) else None
        predict=load_policy(args.model,vecnorm)

    teleop=Teleop(env)
    obs,_=env.reset(seed=0)
    teleop.apply()
    print("Available gaits: " + ", ".join(GAITS_TYPES))
    print(teleop.describe())

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=teleop.on_key,
    ) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING #跟踪模式
        viewer.cam.trackbodyid=env.base_body_id
        viewer.cam.distance=2.5
        viewer.cam.elevation=-15

        while viewer.is_running():
            tic=time.perf_counter()
            if teleop.reset_requested:
                obs,_=env.reset()
                teleop.apply()
                teleop.reset_requested=False
            action=predict(obs)
            obs,_,terminated,_,_=env.step(action)
            if terminated:
                print("fell over ---resetting")
                obs,_=env.reset()
                teleop.apply()
            viewer.sync()
            elapsed = time.perf_counter() - tic
            sleep = env.dt / max(args.realtime, 1e-6) - elapsed
            if sleep > 0:
                time.sleep(sleep)
    env.close()

if __name__=="__main__":
    main()
