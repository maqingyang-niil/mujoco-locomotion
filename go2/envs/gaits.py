import numpy as np
from .utils import _backend
# 四脚顺序
FOOT_ORDER=("FL","FR","RL","RR")

# 步态类型
GAITS_DICT:dict[str,dict]={
    "trot": {"offsets":(0.00,0.50,0.50,0.00),"duty":0.50},
    "pace": {"offsets":(0.00,0.50,0.00,0.50),"duty":0.50},
    "bound":{"offsets":(0.00,0.00,0.50,0.50),"duty":0.50},
    "walk": {"offsets":(0.00,0.50,0.25,0.75),"duty":0.75},
    "pronk":{"offsets":(0.00,0.00,0.00,0.00),"duty":0.50},
    "stand":{"offsets":(0.00,0.00,0.00,0.00),"duty":1.00},
}

GAITS_TYPES=tuple(GAITS_DICT.keys())

# 获取步态参数
def gait_params(type:str)->tuple[np.ndarray,float]:
    if type not in GAITS_TYPES:
        raise KeyError(f"unknown gait {type!r}")
    g=GAITS_DICT[type]
    return np.asarray(g["offsets"],dtype=np.float64),float(g["duty"])

# 更新相位
def step_phase(phase:float,frequency:float,dt:float)->float:
    return float((phase+frequency*dt)%1.0)

# 获取脚的期望姿态
'''
触地: 1.0
抬起: 0.0
'''
def desired_contact(phase:float,offsets,duty:float):
    xp=_backend(offsets)
    local_phase=xp.mod(phase+offsets,1.0)
    return xp.where(local_phase<duty,1.0,0.0)


# 四只脚各自的时钟周期信号
def clock_signal(phase:float,offsets):
    xp=_backend(offsets)
    local_phase=2.0*xp.pi*xp.mod(phase+offsets,1.0)
    return xp.concatenate([xp.sin(local_phase),xp.cos(local_phase)])

