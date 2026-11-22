from dataclasses import dataclass
import numpy as np
from .utils import _backend
import jax










@dataclass
class RewardState:

    # 基座状态
    lin_vel_base:np.ndarray|jax.Array          # (3,) 线性速度，机身坐标系
    ang_vel_base:np.ndarray|jax.Array          # (3,) 角速度，机身坐标系
    proj_gravity:np.ndarray                    # (3,) 重力方向，机身坐标系
    base_height:float                          # 机身高度

    # 关节
    joint_pos:np.ndarray                       # (12,) 当前的关节位置
    joint_vel:np.ndarray                       # (12,) 当前的关节速度
    joint_vel_prev:np.ndarray                  # (12,) 前一状态的关节速度，用来计算关节加速度
    default_joint_pos:np.ndarray               # (12,) 默认位置的关节位置
    soft_joint_limits:np.ndarray               # (12,) 关节软限位
    torques:np.ndarray                         # (12,) 关节力矩

    # 动作
    action_curr:np.ndarray                     # (12,) 当前动作
    action_prev:np.ndarray                     # (12,) 前一状态动作

    # 脚
    contact:np.ndarray                         # (4,) 当前脚的触地情况，触地为1.0,腾空为0.0
    desired_contact:np.ndarray                 # (4,) 期望脚的触地状态
    foot_air_time:np.ndarray                   # (4,) 脚的腾空时间
    foot_new_contact:np.ndarray                # (4,) 上一控制部未接触地面，当前控制部接触地面的情况为1.0,其余为0.0
    foot_vel_xy:np.ndarray                     # (4,2) 脚的xy方向的角速度
    foot_height:np.ndarray                     # (4,) 脚里地面的高度，世界坐标系
    

    # 命令
    cmd:np.ndarray                             # (4,) 命令：x方向线速度，y方向线速度，yaw方向角速度,机身高度

    # 终止
    undesired_contacts:float                   # 非足触地次数
    dt:float                                   # 控制步的时间间隔

    '''
    指数追踪的kernal width
    sigma越大,对误差越宽容
    sigma越小,对误差越敏感
    '''
    lin_vel_sigma:float=0.20                   
    ang_vel_sigma:float=0.25
    target_foot_height:float=0.08              # 期望足部的腾空高度
    target_swing_time:float=0.25               # 期望足部的腾空时间

# xy线速度
def track_lin_vel_xy(r:RewardState):
    xp=_backend(r.lin_vel_base)
    err=xp.sum(xp.square(r.cmd[:2]-r.lin_vel_base[:2]))
    return xp.exp(-err/r.lin_vel_sigma)

# yaw角速度
def track_ang_vel_yaw(r:RewardState):
    xp=_backend(r.ang_vel_base)
    err=xp.square(r.cmd[2]-r.ang_vel_base[2])
    return xp.exp(-err/r.ang_vel_sigma)

# 步态
def gait_phase(r:RewardState):
    xp=_backend(r.contact)
    matches=r.contact*r.desired_contact+(1.0-r.contact)*(1.0-r.desired_contact)
    return xp.mean(matches)

# 脚腾空时间
'''
计算速度命令足够大才计算腾空奖励
腾空时间最大为设定的target swing time
只用当前控制步刚刚着地的脚才参与计算奖励
'''
def foot_air_time(r:RewardState):
    xp=_backend(r.foot_air_time)
    should_award=xp.where(xp.linalg.norm(r.cmd[:3])>0.1,1.0,0.0)
    credited=xp.minimum(r.foot_air_time,r.target_swing_time)
    return xp.sum(credited*r.foot_new_contact)*should_award

# 脚腾空高度
'''
只用期望腾空的脚的腾空高度才参与计算奖励,以免鼓励不必要的脚去腾空
'''
def foot_height(r:RewardState):
    xp=_backend(r.foot_height)
    desired_swing=1.0-r.desired_contact
    return xp.sum(desired_swing*xp.square(r.foot_height-r.target_foot_height))

# z方向的速度
def lin_vel_z(r:RewardState):
    xp=_backend(r.lin_vel_base)
    return xp.square(r.lin_vel_base[2])

# 绕x,y轴的旋转角速度
def ang_vel_xy(r:RewardState):
    xp=_backend(r.ang_vel_base)
    return xp.sum(xp.square(r.ang_vel_base[:2]))

# 机身倾斜
def orientation(r:RewardState):
    xp=_backend(r.proj_gravity)
    return xp.sum(xp.square(r.proj_gravity[:2]))

# 机身高度
def base_height(r:RewardState):
    xp=_backend(r.base_height)
    return xp.square(r.base_height-r.cmd[3])

# 力矩大小
def torques(r:RewardState):
    xp=_backend(r.torques)
    return xp.sum(xp.square(r.torques))

# 关节加速度
def joint_acceleration(r:RewardState):
    xp=_backend(r.joint_vel)
    return xp.sum(xp.square((r.joint_vel-r.joint_vel_prev)/r.dt))

# 动作变化率
def action_rate(r:RewardState):
    xp=_backend(r.action_curr)
    return xp.sum(xp.square(r.action_curr-r.action_prev))

# 关节限位
def joint_limit(r:RewardState):
    xp=_backend(r.joint_pos)
    below=xp.clip(r.soft_joint_limits[:,0]-r.joint_pos,0.0,None)
    above=xp.clip(r.joint_pos-r.soft_joint_limits[:,1],0.0,None)
    return xp.sum(below+above)

# 碰撞
def collision(r:RewardState):
    return r.undesired_contacts

# 脚步滑动
def foot_slip(r:RewardState):
    xp=_backend(r.foot_vel_xy)
    return xp.sum(xp.sum(xp.square(r.foot_vel_xy),axis=-1)*r.contact)

# 站立
def stand_still(r:RewardState):
    xp=_backend(r.joint_pos)
    standing=xp.where(xp.linalg.norm(r.cmd[:3])<0.1,1.0,0.0)
    return xp.sum(xp.abs(r.joint_pos-r.default_joint_pos))*standing

# 存活
def alive(r:RewardState):
    xp=_backend(r.joint_pos)
    return xp.ones(())

REWARD_TERMS={
    "track_lin_vel_xy":   track_lin_vel_xy,
    "track_ang_vel_yaw":  track_ang_vel_yaw,
    "gait_phase":         gait_phase,
    "foot_air_time":      foot_air_time,
    "foot_height":        foot_height,
    "lin_vel_z":          lin_vel_z,
    "ang_vel_xy":         ang_vel_xy,
    "orientation":        orientation,
    "base_height":        base_height,
    "torques":            torques,
    "joint_acceleration": joint_acceleration,
    "action_rate":        action_rate,
    "joint_limit":        joint_limit,
    "collision":          collision,
    "foot_slip":          foot_slip,
    "stand_still":        stand_still,
    "alive":              alive,
}

REWARD_WEIGHTS_DEFAULT={
    "track_lin_vel_xy":   1.5,
    "track_ang_vel_yaw":  0.75,
    "gait_phase":         1.5,
    "foot_air_time":      1.0,
    "foot_height":        -30.0,
    "lin_vel_z":          -2.0,
    "ang_vel_xy":         -0.05,
    "orientation":        -1.0,
    "base_height":        -5.0,
    "torques":            -2.0e-4,
    "joint_acceleration": -2.5e-7,
    "action_rate":        -0.01,
    "joint_limit":        -10,
    "collision":          -1.0,
    "foot_slip":          -0.05,
    "stand_still":        -0.5,
    "alive":              0.25,
}

# 配置奖励函数权重
'''
先复制default的奖励函数权重,然后从yaml中覆盖对应项并检查不存在的奖励函数权重
'''
def resolve_weights(cfg:dict):
    weights=dict(REWARD_WEIGHTS_DEFAULT)
    if cfg:
        unknown=set(cfg)-set(REWARD_TERMS)
        if unknown:
            raise KeyError("unknown terms in config: %s"% sorted(unknown))
    weights.update({k: float(v) for k,v in cfg.items()})
    return weights

# 计算非零奖励函数权重
'''
没有经过dt放缩的奖励之和,dt需要由被调用者提供计算
'''
def compute(state,weights:dict):
    per_term={}
    total=0.0
    for name,weight in weights.items():
        if weight==0:
            continue
        value=REWARD_TERMS[name](state)*weight
        per_term[name]=value
        total+=value
    return total,per_term