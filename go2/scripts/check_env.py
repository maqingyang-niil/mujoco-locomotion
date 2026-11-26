'''
针对环境的快速检查,每次训练之前调用这个
'''

import argparse
from pathlib import Path
from random import sample
from go2.envs.gaits import gait_params
import yaml
from go2.envs.commands import CommandRanges, ranges_from_config
from go2.envs.go2_env import Go2Env
from go2.envs.gaits import desired_contact
import numpy as np
GO2_ROOT=Path(__file__).resolve().parents[1]

# 分区美观
def section(title:str):
    print("\n"+title)
    print("-"*len(title)*2)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default=GO2_ROOT/"configs"/"training_config.yaml")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--ranges",choices=("initial","final"),default="initial",
                   help="Which command ranges to check. INITIAL is where the " \
                   "curriculum starts from and decides whether or not the simulation " \
                   "is learnable.")
    args=p.parse_args()
    with open(args.config) as f:
        config=yaml.safe_load(f)

    env_cfg=dict(config["environment"])
    cmd_cfg=dict(config["commands"])
    env_cfg.update(
        reward_weights=(config.get("reward") or {}).get("weights"),
        command_ranges=ranges_from_config(cmd_cfg.get(args.ranges)),
        stand_probability=cmd_cfg.get("stand_probability",0.10),
        command_resample_interval_s=cmd_cfg.get("resample_interval_s",5.0),
        gaits=tuple(cmd_cfg.get("gaits",["trot"])),
        feasibility_margin=cmd_cfg.get("feasibility_margin",0.85),
        randomize_dynamics=False,# 不作为检测对象
        push_enable=False,
    )
    env=Go2Env(**env_cfg)

    section("模型")
    print("physics dt          : %.4f" % env.physics_dt)
    print("decimation          : %d" % env.decimation)
    print("control rate        : %.1f" % (1.0/env.dt))
    print("observation dim     : %d" % env.obs_dim)
    print("action dim          : %d" % env.n_jnts)
    print("foot geoms          : %s -> %s" % (list(env.FOOT_GEOM_NAMES),env.foot_geom_ids.tolist()))
    print("torque limits       : %s" % np.round(env.torque_limits,1).tolist())

    section("正常位姿")
    print("default joint pos   : %s" % np.round(env.default_jnt_pos,3).tolist())
    print("nominal height      : %.3f m" % env.nominal_base_height)

    expected=np.tile([0.0,0.9,-1.8],4)
    ok=np.allclose(env.default_jnt_pos,expected,atol=1e-6)
    print("matches Go2 home    : %s" % ("OK" if ok else "NO"))

    section("站立测试")
    env.reset(seed=0)
    env.data.qpos[7:]=env.default_jnt_pos
    env.data.qpos[2]=env.nominal_base_height
    env.data.qpos[3:7]=np.array([1.0,0.0,0.0,0.0])
    env.data.qvel[:]=0.0
    env.set_command(lin_vel_x=0.0,lin_vel_y=0.0,ang_vel_yaw=0.0,gait="stand")
    heights,contacts=[],[]
    fell=False
    for _ in range(150):
        _,_,terminated,_,info=env.step(np.zeros(12))
        heights.append(info["base_height"])
        contacts.append(info["contacts"].sum())
        if terminated:
            fell=True
            break
    if fell:
        print("机器人倒下,当命令要求静止站立")
    else:
        h=np.array(heights[50:])
        print("height mean/std     : %.3f / %.4f m" % (h.mean(),h.std()))
        print("mean foot down      : %.2f / 4" % np.mean(contacts[50:]))
        print("verdict             : %s" % ("OK" if h.std()<0.02 and h.mean()>0.2 else "NOT OK"))

    section("速度范围测试")
    stride=env.max_stride_length
    sampler=env.command_sampler
    command_ranges=sampler.ranges
    margin=sampler.feasibility_margin
    torque_ok=env.kp*env.action_scale<env.torque_limits[1]
    print("action scale        : %.2f rad" % env.action_scale)
    print("PD torque/max       : %.2f / %.1f N*m" % (env.kp*env.action_scale,env.torque_limits[1]))
    print("stance foot travel  : %.2f cm" % (stride*100))
    print("max speed per hz    : %s" % sampler.max_speed_per_hz)
    print("max_speed           : %s" % sampler.max_speed)
    print("margin              : %.2f" % margin)

    print("%-9s %12s %14s" % ("frequency", "static bound", "sampler limit"))
    f_min,f_max=command_ranges.gait_frequency

    def sampler_limit(f):
        v=float("inf")
        if sampler.max_speed_per_hz:
            v=sampler.max_speed_per_hz*f
        if sampler.max_speed:
            v=min(v,sampler.max_speed)
        return v*margin
    freqs=np.append(np.arange(f_min,f_max,0.5),f_max)
    for f in freqs:
        limit = sampler_limit(f)
        print(
            "%7.1f Hz %11.2f %14.2f"
            % (f, stride * f, limit)
        )
    asked_x=max(abs(x) for x in command_ranges.lin_vel_x)
    asked_y=max(abs(y) for y in command_ranges.lin_vel_y)
    reachable_x=sampler_limit(f_max)
    reachable_y=0.5*reachable_x
    speed_ok=asked_x<=reachable_x+1e-6 and asked_y<=reachable_y+1e-6
    print()
    print("vx requested/reachable    :%.2f / %.2f" % (asked_x,asked_y))
    print("vy requested/reachable    :%.2f / %.2f" % (asked_y,asked_y))
    print("torque verdicr            :%s" % ("OK" if torque_ok else "NOT OK"))
    print("speed verdicr             :%s" % ("OK" if speed_ok else "NOT OK"))

    section("静止策略基线")
    rng=np.random.default_rng(0)
    cmds=[env.command_sampler.sample(rng) for _ in range(20000)]
    vels=np.array([c.vel for c in cmds])
    lin_err=np.linalg.norm(vels[:,:2],axis=1)
    ang_err=np.abs(vels[:,2])
    score=0.5*np.exp(-lin_err**2/env.lin_vel_sigma)+0.5*np.exp(-ang_err**2/env.ang_vel_sigma)
    print("当机器人站立不动时,得到的奖励与curriculum的奖励阈值比较")
    print("如果站立不动的情况下获得的奖励抬高,机器人可能直接不会移动")
    print()
    print("do nothing tracking score : %.2f" % score.mean())
    print("curriculum threshold      : 0.85")
    print("verdict                   : %s" % ("OK" if score.mean()<0.70 else "TOO HIGH"))

    section("步态参数检测")
    # 潜在提升点，env.command_sampler.gait
    offsets,duty=gait_params("trot")
    rows=[]
    for phase in np.linspace(0,1,8,endpoint=False):
        d=desired_contact(phase,offsets,duty)
        rows.append("  phi=%.2f  " % phase + " ".join(
            ("[]" if x else "..") for x in d))
    print("  " + "   ".join(env.FOOT_GEOM_NAMES))
    print("\n".join(rows))

    section("随机动作测试 %d steps" % args.steps)
    obs,_=env.reset(seed=1)
    rng=np.random.default_rng(1)
    rewards = []
    term_sums = {}
    episodes = 0
    lengths = []
    length = 0
    bad = []
    for _ in range(args.steps):
        obs,reward,terminated,truncated,info=env.step(rng.uniform(-1,1,12))
        length+=1
        if not np.all(np.isfinite(obs)):
            bad.append("infinit observation")
        if not np.isfinite(reward):
            bad.append("infinit reward")
        rewards.append(reward)
        for k,v in info.items():
            if k.startswith("rew/"):
                term_sums[k]=term_sums.get(k,0.0)+v
                if not np.isfinite(v):
                    bad.append("infinit "+k)
        if terminated or truncated:
            episodes+=1
            lengths.append(length)
            length=0
            obs,_=env.reset()
    
    print("mean reward/step : %+.4f" % np.mean(rewards))
    print("episodes         : %d" % episodes)
    if lengths:
        print("mean length      : %.1f steps, %.2f s" % (np.mean(lengths),np.mean(lengths)*env.dt))
    print("obs range        : [%.2f, %.2f]" % (obs.min(), obs.max()))

    print("\nreward decomposition (mean per step):")
    # Share of the total *magnitude*, not of the net sum. Dividing by the net
    # sum is meaningless when carrots and sticks nearly cancel - you get shares
    # over 100%, which tells you nothing about which term is loud.
    magnitude = sum(abs(v) for v in term_sums.values()) or 1.0
    for name, value in sorted(term_sums.items(), key=lambda kv: -abs(kv[1])):
        print("  %-28s %+9.5f   (%5.1f%% of gross)" % (name[len("rew/"):],value/args.steps,abs(value) / magnitude * 100))
    print("  %-28s %+9.5f" % ("NET", sum(term_sums.values()) / args.steps))

    section("奖励项完整性检查")
    logged = {k[len("rew/"):] for k in term_sums}
    configured = {k for k, w in env.reward_weights.items() if w != 0.0}
    missing = configured - logged
    extra = logged - configured
    print("configured non-zero terms : %d" % len(configured))
    print("terms seen in info        : %d" % len(logged))
    if missing:
        print("MISSING FROM INFO         : %s" % sorted(missing))
    if extra:
        print("UNEXPECTED IN INFO        : %s" % sorted(extra))

    env.close()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
