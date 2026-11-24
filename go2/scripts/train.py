


import argparse
from ast import arg, parse
from asyncio import trsock
from email.policy import Policy
from multiprocessing import dummy
from multiprocessing.spawn import import_main_path
import trace
from typing import final
from jax.core import nonempty_axis_env_DO_NOT_USE
from sympy import false, is_monotonic
from torch import norm
from wandb.docker import parse_repository_tag
import yaml
from stable_baselines3 import PPO
from go2.envs.commands import ranges_from_config
from go2.envs.go2_env import Go2Env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from go2.callbacks.curriculum import CommandCurriculumCallback
from go2.callbacks.logging import RewardTermLoggerCallback,EpisodeStatesCallback
from pathlib import Path
import os

GO2_ROOT=Path(__file__).resolve().parents[1]




def build_env_kwargs(config:dict):
    env_cfg=config.get("environment",{}) or {}
    cmd_cfg=config.get("commands",{}) or {}
    reward_cfg=(config.get("reward",{}) or {}).get("weights")
    initial=ranges_from_config(cmd_cfg.get("initial"))
    env_cfg.update(
        command_ranges=initial,
        stand_probability=cmd_cfg.get("stand_probability",0.10),
        command_resample_interval_s=cmd_cfg.get("resample_interval_s",5.0),
        gaits=tuple(cmd_cfg.get("gaits",["trot"])),
        max_speed_per_hz=cmd_cfg.get("max_speed_per_hz",0.40),
        max_speed=cmd_cfg.get("max_speed",1.0),
        feasibility_margin=cmd_cfg.get("feasibility_margin",0.85),
        reward_weights=reward_cfg,
    )
    return env_cfg

def make_env(env_kwargs:dict,rank:int,seed=0):
    def _init():
        env=Go2Env(**env_kwargs)
        env=Monitor(env,info_keywords=("tracking_lin_err","tracking_ang_err"))
        env.reset(seed=seed+rank)
        env.action_space.seed(seed+rank)
        return env
    return _init

def _guess_vecnormalize_path(model_path):
    directory,filename=os.path.split(model_path)
    stem=os.path.splitext(filename)[0]
    for "_" in stem:
        head,_,tail=stem.partition("_")
        candidate=os.path.join(directory,head+"_vecnormalize_"+tail+".pkl")
        if os.path.exists(candidate):
            return candidate
    for name in os.listdir(directory or "."):
        if name.startswith("vecnornalize") or "vecnormalize" in name:
            if name.endswith(".pkl"):
                return os.path.join(directory,name)
    return None


def main():
    parser=argparse.ArgumentParser(description="Training Go2 Quadruped with PPO")
    parser.add_argument("--config",type=str,default=GO2_ROOT/"configs"/"training_config.yaml")
    parser.add_argument("--resume",type=str,default=None,help="需要一个模型的.zip文件和对应的VecNormalize文件")
    parser.add_argument("--vecnormalise",type=str,default=None,help="需要VecNormalize的.pkl文件初始化模型")
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--device",type=str,default="auto")
    parser.add_argument("--n-env",type=int,default=None)
    parser.add_argument("--vec-env",choices=("auto","subproc","dummy"),default="auto",help="")
    parser.add_argument("--timesteps",type=int,default=None)
    parser.add_argument("--run-name",type=str,default="go2_ppo")
    args=parser.parse_args()

    with open(args.config) as f:
        config=yaml.safe_load(f)

    env_cfg=config["environment"]
    cmd_cfg=config["commands"]
    curriculum_cfg=config["curriculum"]
    reward_cfg=config["reward"]
    train_cfg=config["training"]
    log_cfg=config["logging"]

    seed=args.seed if args.seed is not None else train_cfg.get("seed",0)
    n_envs=args.n_envs if args.n_envs is not None else train_cfg.get("n_envs",16)
    total_timesteps=args.timesteps if args.timesteps is not None else train_cfg.get("total_timesteps")

    model_dir=log_cfg["model_dir"]
    for sub in ("","checkpoints","best"):
        os.makedirs(os.path.join(model_dir,sub),exist_ok=True)
    os.makedirs(log_cfg["tensorboard_log"],exist_ok=True)

    env_kwargs=build_env_kwargs(config)

    #评估环境参数
    eval_kwargs=dict(env_kwargs)
    eval_kwargs.update(randomize_dynamics=False,push_enable=False)

    if args.vec_env=="subproc":
        vec_cls=SubprocVecEnv
    elif args.vec_env=="dummy":
        vec_cls=DummyVecEnv
    else:
        vec_cls=SubprocVecEnv if n_envs>1 else DummyVecEnv
    env=vec_cls([make_env(env_kwargs,i,seed) for i in range(n_envs)])
    env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.0)

    eval_env=DummyVecEnv([make_env(eval_kwargs,0,seed+10_000)])
    eval_env=VecNormalize(eval_env,norm_obs=True,norm_reward=False,clip_obs=10.0)
    eval_env.training=False

    checkpoint_cb=CheckpointCallback(
        save_freq=max(1,log_cfg["save_freq"]//n_envs),
        save_path=os.path.join(model_dir,"checkpoints"),
        name_prefix=args.run_name,
        save_vecnormalize=True
    )

    eval_cb=EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir,"best"),
        log_path=os.path.join(log_cfg["log_dir"],"eval"),
        eval_freq=max(1,log_cfg["eval_freq"]//n_envs),
        n_eval_episodes=log_cfg["n_eval_episodes"],
        deterministic=True,
    )

    callbacks=[checkpoint_cb,eval_cb,RewardTermLoggerCallback(),EpisodeStatesCallback()]

    if curriculum_cfg.get("enable",True):
        callbacks.append(
            CommandCurriculumCallback(
                initial_ranges=ranges_from_config(cmd_cfg.get("initial")),
                final_ranges=ranges_from_config(cmd_cfg.get("final")),
                threshold=curriculum_cfg.get("threshold",0.85),
                decay_threshold=curriculum_cfg.get("decay_threshold",0.55),
                step=curriculum_cfg.get("step",0.05),
            )
        )

    policy_kwargs=dict(train_cfg.get("policy_kwargs",{}) or {})
    arch=policy_kwargs.get("net_arch")

    if isinstance(arch,list):
        policy_kwargs["net_arch"]=dict(pi=list(arch),vf=list(arch))

    lr=train_cfg.get("learning_rate")

    if train_cfg.get("lr_schedule","constant")=="linear":
        lr=get_linear_fn(start=float(lr),end=0.0,end_fraction=1.0)

    if args.resume:
        print("Resuming from "+args.resume)
        model=PPO.load(args.resume,env=env,device=args.device)

        stats_path=args.vecnormalize or _guess_vecnormalize_path(args.resume)
        if stats_path and os.path.exists(stats_path):
            print("Restoring VecNormalize statistics from "+stats_path)
            env=VecNormalize.load(stats_path,env.venv)
            env.training=True
            env.norm_reward=True
            model.set_env(env)
        else:
            raise SystemError("Could not find the VecNormalize statistics for this checkpoint.")
    else:
        model = PPO(
            policy=train_cfg["policy"],
            env=env,
            learning_rate=lr,
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            target_kl=train_cfg.get("target_kl"),
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_cfg["tensorboard_log"],
            seed=seed,
            verbose=1,
            device=args.device,
        )
    print("Training PPO for %s timesteps with %d environments," \
    "rollouts %d transitions, %d minibatches x %d epochs" 
    % (
        f"{total_timesteps:,}",
        n_envs,
        n_envs*train_cfg["n_steps"],
        max(1,n_envs*train_cfg["n_steps"]//train_cfg["batch_size"]),
        train_cfg["n_epochs"],
    ))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=args.run_name,
            progress_bar=True,
            reset_num_timesteps=not args.resume,
        )
    finally:
        model.save(os.path.join(model_dir,args.run_name+"_final"))
        env.save(os.path.join(model_dir,args.run_name+"_vecnornalize.pkl"))
        print("Saved model and normalisation statistics to "+model_dir)


if __name__=="__main__":
    main()
    


    
