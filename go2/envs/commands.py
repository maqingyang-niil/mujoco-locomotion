import numpy as np
from dataclasses import dataclass
from typing import Self
from .gaits import GAITS_TYPES





# 命令范围类
@dataclass
class CommandRanges:
    lin_vel_x:tuple=(-1.0,1.5)
    lin_vel_y:tuple=(-0.7,0.7)
    ang_vel_yaw:tuple=(-1.5,1.5)
    gait_frequency:tuple=(1.5,3.0)
    base_height:tuple=(0.28,0.34)

    # 两个命令范围线性插值
    def linear_interpolate(self,other:Self,t)->Self:
        t=float(np.clip(t,0.0,1.0))

        def _interpolate(a,b):
            return ((1.0-t)*a[0]+t*b[0],(1.0-t)*a[1]+t*b[1])
        return CommandRanges(
            lin_vel_x=_interpolate(self.lin_vel_x,other.lin_vel_x),
            lin_vel_y=_interpolate(self.lin_vel_y,other.lin_vel_y)
        )

    # 输出命令范围字典
    def as_dict(self)->dict:
        return {
            "lin_vel_x":tuple(self.lin_vel_x),
            "lin_vel_y":tuple(self.lin_vel_y),
            "ang_vel_yaw":tuple(self.ang_vel_yaw),
            "gait_frequency":tuple(self.gait_frequency),
            "base_height":tuple(self.base_height),
        }

# 命令类
@dataclass
class Command:
    lin_vel_x:float=0.0
    lin_vel_y:float=0.0
    ang_vel_yaw:float=0.0
    gait:str="trot"
    gait_frequency:float=2.0
    base_height:float=0.3

    # 命令速度
    '''
    x方向线速度
    y方向线速度
    yaw方向角速度
    '''
    @property
    def vel(self)->np.array:
        return np.array(
            [
                self.lin_vel_x,
                self.lin_vel_y,
                self.ang_vel_yaw,
            ],
            dtype=np.float64,
        )

    # 是否静止站立
    '''
    合速度模值小于0.1看作站立静止
    '''
    @property
    def is_standing(self)->bool:
        return float(np.linalg.norm(self.vel))<0.1

class CommandSampler:
    def __init__(
        self,
        ranges=None,
        stand_probability=0.10,
        resample_interval_s=5.0,
        gaits=("trot",),
        gait_probability=None,
        max_speed_per_hz=None,
        max_speed=None,
        feasibility_margin=0.85,
    ):
        self.ranges=ranges or CommandRanges()
        self.stand_probability=stand_probability
        self.resample_interval_s=resample_interval_s
        self.gait_probability=gait_probability
        self.max_speed_per_hz=max_speed_per_hz
        self.max_speed=max_speed
        self.feasibility_margin=feasibility_margin

        for g in gaits:
            if g not in GAITS_TYPES:
                raise RuntimeError("unknown gait %r"% g)
        self.gaits=tuple(gaits)

    # 从命令范围采样命令
    '''
    首先处理步态,步态频率以及基座高度,速度需要经过最大值clip后再放入命令中
    '''
    def sample(self,rng:np.random.Generator)->Command:
        ranges=self.ranges
        gait=str(rng.choice(self.gaits,p=self.gait_probability))
        # 速度需要之后才处理
        cmd=Command(
            gait=gait,
            gait_frequency=float(rng.uniform(*ranges.gait_frequency)),
            base_height=float(rng.uniform(*ranges.base_height))
        )

        if rng.random()<self.stand_probability:
            cmd.gait="stand"
            return cmd

        lo_x,hi_x=ranges.lin_vel_x
        lo_y,hi_y=ranges.lin_vel_y
        if self.max_speed_per_hz or self.max_speed:
            v_max=float("inf")
            if self.max_speed_per_hz:
                v_max=self.max_speed_per_hz*cmd.gait_frequency
            if self.max_speed:
                v_max=min(v_max,self.max_speed)
            v_max*=self.feasibility_margin
            lo_x=max(lo_x,-v_max)
            hi_x=min(hi_x,v_max)
            lo_y=max(lo_y,-0.5*v_max)
            hi_y=min(hi_y,0.5*v_max)

        cmd.lin_vel_x=float(rng.uniform(lo_x,hi_x))
        cmd.lin_vel_y=float(rng.uniform(lo_y,hi_y))
        cmd.ang_vel_yaw=float(rng.uniform(*ranges.ang_vel_yaw))

        if float(np.linalg.norm(cmd.vel))<0.15:
            cmd.lin_vel_x=0.0
            cmd.lin_vel_y=0.0
            cmd.ang_vel_yaw=0.0
            cmd.gait="stand"
        return cmd

    def should_resample(self,elapsed_s)->bool:
        if self.resample_interval_s<=0:
            return False
        return elapsed_s>=self.resample_interval_s

# 根据分值扩大命令的范围
class CommandCurriculum:
    def __init__(
        self,
        initial_comm:CommandRanges,
        final_comm:CommandRanges,
        threshold=0.85,
        step=0.05,
        decay_threshold=0.65,
    ):
        self.initial_comm=initial_comm
        self.final_comm=final_comm
        self.threshold=float(threshold)
        self.decay_threshold=float(decay_threshold)
        self.step=float(step)
        self.level=0.0

    def update(self,tracking_score)->float:
        if tracking_score>=self.threshold:
            self.level=min(1.0,self.level+self.step)
        elif tracking_score<self.decay_threshold:
            self.level=max(0.0,self.level-self.step)
        return self.level

    @property
    def ranges(self)->CommandRanges:
        return self.initial_comm.linear_interpolate(self.final_comm,self.level)

# 从config中加载命令范围
def ranges_from_config(cfg:dict)->CommandRanges:
    if not cfg:
        return CommandRanges()
    d=CommandRanges()
    return CommandRanges(
        lin_vel_x=tuple(cfg.get("lin_vel_x",d.lin_vel_x)),
        lin_vel_y=tuple(cfg.get("lin_vel_y",d.lin_vel_y)),
        ang_vel_yaw=tuple(cfg.get("ang_vel_yaw",d.ang_vel_yaw)),
        gait_frequency=tuple(cfg.get("gait_frequency",d.gait_frequency)),
        base_height=tuple(cfg.get("base_height",d.base_height)),
    )