import argparse
import os
import pickle
import sys
from typing import final
from matplotlib import axis
from pathlib import Path
import numpy as np
from go2.envs.gaits import GAITS_TYPES, gait_params,desired_contact
from go2.envs.go2_env import Go2Env
from go2.envs.gaits import FOOT_ORDER
GO2_ROOT=Path(__file__).resolve().parents[1]
PROJ_ROOT=Path(__file__).resolve().parents[2]

FOOT_LABELS = ["FL (front left)", "FR (front right)",
               "RL (rear left)", "RR (rear right)"]
FOOT_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover - plotting is optional
    HAS_MPL = False

# 占空比
def duty_factor(contacts):
    return contacts.mean(axis=0)

# 触地点索引
def touchdown_indices(foot_contacts):
    return np.where(np.diff(foot_contacts.astype(int)) > 0)[0] + 1

# 触地频率
def stride_frequencies(contacts,dt):
    out=np.zeros(contacts.shape[1])
    for i in range(contacts.shape[1]):
        td=touchdown_indices(contacts[:,i])
        if len(td)>2:
            out[i]=1.0/(np.mean(np.diff(td))*dt)
    return out

def phase_offsets(contacts,dt):
    fl=touchdown_indices(contacts[:,0])
    if len(fl)<2:
        return np.full(contacts.shape[1],np.nan)
    period=float(np.mean(np.diff(fl)))
    offsets=np.zeros(contacts.shape[1])
    for i in range(contacts.shape[1]):
        td=touchdown_indices(contacts[:,i])
        if len(td)==0:
            offsets[i]=np.nan
            continue
        rel=[]
        for t in fl[:-1]:
            following=td[td>=t]
            if len(following):
                rel.append(((following[0]-t)%period)/period)
        offsets[i]=float(np.median(rel)) if rel else np.nan
    return offsets

def schedule_match(contacts,phases,gait):
    offsets,duty=gait_params(gait)
    desired=np.array([desired_contact(p,offsets,duty) for p in phases])
    return float(np.mean(contacts==desired)),desired

def load_policy(model_path:str,vecnormalize_path:str,device="cpu"):
    if model_path.endswith(".npz"):
        from go2.envs.brax_policy import BraxPolicy
        policy=BraxPolicy.load(model_path)
        print("loading brax policy from %s (%s)" % (model_path,",".join("%s=%s" % kv for kv in sorted(policy.meta.items()))))
        return policy
    from stable_baselines3 import PPO,SAC,TD3
    algo=PPO
    for name, cls in (("sac",SAC),("td3",TD3),("ppo",PPO)):
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
        print("No VecNormalize statistic is provided thus the results will be meaningless")

    def predict(obs):
        x=obs
        if obs_rms is not None:
            x=np.clip((obs-obs_rms.mean)/np.sqrt(obs_rms.var+epsilon),-clip_obs,clip_obs).astype(np.float32)
        action,_=model.predict(x,deterministic=True)
        return action
    return predict

def record(env:Go2Env,predict,command,gait,frequency,steps,settles=100,seed=0):
    obs,_=env.reset(seed=seed)
    env.set_command(lin_vel_x=command[0],lin_vel_y=command[1],ang_vel_yaw=command[2],gait=gait,gait_frequency=frequency)
    contacts,phases,vels=[],[],[]
    for i in range(steps+settles):
        action=predict(obs)
        obs,_,terminated,_,info=env.step(action)
        if i>settles:
            contacts.append(info["contacts"])
            phases.append(info["gait_phase"])
            vels.append(info["lin_vel_b"][:2])
        if terminated:
            break
    return np.array(contacts),np.array(phases),np.array(vels)

def report(contacts,phases,vels,gait,dt,command):
    if len(contacts)<10:
        print("机器人在摔倒之前没有采集到足够多的数据")
        return None

    duty=duty_factor(contacts)
    freq=stride_frequencies(contacts,dt)
    offsets=phase_offsets(contacts,dt)
    matched,desired=schedule_match(contacts,phases,gait)
    ref_offsets,ref_duty=gait_params(gait)

    print("\ncommanded gait      : %s (reference offsets %s, duty %.2f)"
          % (gait, tuple(ref_offsets), ref_duty))
    print("commanded velocity  : vx=%.2f vy=%.2f yaw=%.2f"
          % (command[0], command[1], command[2]))
    print("achieved velocity   : vx=%.3f vy=%.3f"
          % (vels[:, 0].mean(), vels[:, 1].mean()))
    print("schedule match      : %.1f%%  (the gait_phase reward, as a percentage)"
          % (100 * matched))
    print()
    print("%-20s %10s %10s %12s %12s" % ("foot", "duty", "stride Hz",
                                         "phase (meas)", "phase (ref)"))
    print("-" * 68)
    for i, label in enumerate(FOOT_LABELS):
        print("%-20s %10.3f %10.2f %12s %12.2f"
              % (label, duty[i], freq[i],
                 "nan" if np.isnan(offsets[i]) else "%.2f" % offsets[i],
                 ref_offsets[i]))
    print("-" * 68)
    print("mean duty factor    : %.3f   (reference %.2f)" % (duty.mean(), ref_duty))
    print("mean stride freq    : %.2f Hz" % freq.mean())

    # The interpretation the numbers are actually for.
    notes = []
    if duty.mean() > ref_duty + 0.15:
        notes.append("Feet stay down longer than commanded: the policy is "
                     "shuffling rather than swinging. Raise feet_air_time.")
    if matched < 0.7:
        notes.append("Poor schedule match: either gait_phase's weight is too "
                     "low, or the commanded frequency is outside what the "
                     "policy was trained on.")
    if np.nanmax(np.abs(offsets - ref_offsets)) > 0.2:
        notes.append("Measured phase offsets do not match the reference: the "
                     "policy is running a different gait from the one asked for.")
    if notes:
        print("\nnotes:")
        for n in notes:
            print("  - " + n)

    return dict(duty=duty, freq=freq, offsets=offsets, match=matched,
                desired=desired)

def plot(contacts, desired, dt, gait, path):
    if not HAS_MPL:
        print("matplotlib is not installed; skipping the figure.")
        return
    n = len(contacts)
    t = np.arange(n) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ax, data, title in (
        (axes[0], desired, "commanded schedule"),
        (axes[1], contacts, "measured contacts"),
    ):
        for i in range(4):
            on = data[:, i] > 0.5
            ax.fill_between(t, i - 0.4, i + 0.4, where=on,
                            color=FOOT_COLORS[i], step="mid", linewidth=0)
        ax.set_yticks(range(4))
        ax.set_yticklabels(FOOT_ORDER)
        ax.set_ylim(-0.6, 3.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

    axes[1].set_xlabel("time (s)")
    fig.suptitle("Gait diagram - %s" % gait, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote " + path)

def main():
    p=argparse.ArgumentParser(description="绘制步态分析图")
    p.add_argument("--model",type=str,default=GO2_ROOT/"models"/"go2_ppo_final.zip")
    p.add_argument("--vec-normalize",type=str,default=GO2_ROOT/"models"/"go2_ppo_vecnormalize.pkl")
    p.add_argument("--xml",type=str,default=PROJ_ROOT/"mujoco_menagerie"/"unitree_go2"/"scene.xml")
    p.add_argument("--gait", default="trot", choices=list(GAITS_TYPES))
    p.add_argument("--all-gaits", action="store_true")
    p.add_argument("--cmd", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    p.add_argument("--frequency", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=GO2_ROOT/"analysis")
    args=p.parse_args()
    vecnorm = args.vec_normalize
    if vecnorm is None:
        for guess in ("models/go2_ppo_vecnormalize.pkl",
                      "models/vecnormalize_final.pkl"):
            if os.path.exists(guess):
                vecnorm = guess
                break
    predict = load_policy(args.model, vecnorm)
    env = Go2Env(xml_path=args.xml, randomize_dynamics=False, push_enable=False,
                 command_resample_interval_s=0.0,
                 max_episode_steps=args.steps + 200)
    os.makedirs(args.out_dir, exist_ok=True)

    gaits=[g for g in GAITS_TYPES if g!="stand"] if args.all_gaits else [args.gait]
    try:
        for gait in gaits:
            contacts, phases, vels = record(
                env, predict, args.cmd, gait, args.frequency, args.steps,
                seed=args.seed
            )
            result = report(contacts, phases, vels, gait, env.dt, args.cmd)
            if result is not None:
                plot(contacts, result["desired"], env.dt, gait,
                     os.path.join(args.out_dir, "gait_%s.png" % gait))
    finally:
        env.close()

if __name__=="__main__":
    main()



