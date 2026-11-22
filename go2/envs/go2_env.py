from pathlib import Path
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import os
from gymnasium import spaces
from .commands import CommandRanges,Command,CommandCurriculum,CommandSampler
from . import gaits as gait_module
from . import rewards as reward_module
class Go2Env(gym.Env):
    PROJECT_ROOT=Path(__file__).resolve().parents[2]
    FOOT_GEOM_NAMES=("FL","FR","RL","RR")
    WORLD_GRAVITY_DIR=np.array([0.0,0.0,-1.0])
    OBS_SCALE_ANG_VEL=0.25
    OBS_SCALE_LIN_VEL=2.0
    OBS_SCALE_JNT_VEL=0.05
    OBS_SCALE_CMD=np.array([2.0,2.0,0.25],dtype=np.float64)
    def __init__(
            self,
            xml_path=PROJECT_ROOT/"mujoco_menagerie"/"unitree_go2"/"scene.xml",
            #rand
            decimation=10,
            action_scale=0.4,
            kp=55.0,
            kd=1.4,
            max_episode_steps=1000,
            #termination
            base_height_min=0.12,
            pitch_roll_max=0.8,
            #randomisation
            randomize_dynamics=False,
            reset_noise_scale=1.0,
            obs_noise_scale=0.0,
            #commands
            command_ranges=None,
            stand_probability=0.1,
            command_resample_interval_s=5.0,
            max_speed_per_hz=0.40,
            max_speed=1.0,
            feasibility_margin=0.15,
            gaits=("trot",),
            #reward
            reward_weights=None,
            lin_vel_sigma=0.2,
            ang_vel_sigma=0.25,
            #robust
            push_enable=True,
            push_interval_s=(3.0,7.0),
            push_duration_s=0.15,
            push_force=(-40.0,40.0),
            #others
            render_mode=None,
            render_size=(640,480),
            ):
        self.model=mujoco.MjModel.from_xml_path(str(xml_path))
        self.data=mujoco.MjData(self.model)

        self.decimation=int(decimation)
        self.physics_dt=float(self.model.opt.timestep)
        self.dt=self.decimation*self.physics_dt
        self.metadata["render_fps"]=int(round(1.0/self.dt))

        self.action_scale=float(action_scale)
        self.kd=float(kd)
        self.kp=float(kp)
        self.max_episode_steps=int(max_episode_steps)
        self.base_height_min=float(base_height_min)
        self.pitch_roll_max=float(pitch_roll_max)
        self.reset_noise_scale=float(reset_noise_scale)
        self.randomize_dynamics=bool(randomize_dynamics)
        self.obs_noise_scale=float(obs_noise_scale)
        self.render_mode=render_mode
        self.render_size=tuple(render_size)
        self.push_enable=bool(push_enable)
        self.push_interval_s=tuple(push_interval_s)
        self.push_duration_s=float(push_duration_s)
        self.push_force=tuple(push_force)

        # keyframe是模型中预先保存的一组机器人状态
        self._home_key_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_KEY,"home")
        if self._home_key_id<0:
            raise RuntimeError("The MJCF has no keyframe")
        self.default_jnt_pos=self.model.key_qpos[self._home_key_id,7:].copy()
        self.nominal_base_height=float(self.model.key_qpos[self._home_key_id, 2])

        self.n_jnts=self.model.nu
        self.actuated_jnt_ids=self.model.actuator_trnid[:,0].copy()
        jnt_ranges=self.model.jnt_range[self.actuated_jnt_ids].copy()
        span=jnt_ranges[:,1]-jnt_ranges[:,0]
        self.soft_jnt_limits=np.stack([jnt_ranges[:,0]+0.05*span,jnt_ranges[:,1]-0.05*span],axis=1)
        self.torque_limits=self.model.actuator_ctrlrange[:,1].copy()

        self.floor_geom_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"floor")
        if self.floor_geom_id<0:
            raise RuntimeError("scene.xml is missing a geom named 'floor'")
        #足部id
        self.foot_geom_ids=[]
        for name in self.FOOT_GEOM_NAMES:
            geom_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name)
            if geom_id<0:
                raise RuntimeError("Foot geom %r not found"% name)
            self.foot_geom_ids.append(geom_id)
        self.foot_geom_ids=np.array(self.foot_geom_ids,dtype=np.int32)

        #足部球体半径
        self.foot_radius=float(self.model.geom_size[self.foot_geom_ids[0]][0])

        #身体id
        self.base_body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"base")
        if self.base_body_id<0:
            raise RuntimeError("scene.xml has no body named 'base'")
        foot_set=set(self.foot_geom_ids.tolist())

        #机器人身体上可以参与碰撞的id
        self.non_foot_geom_ids=np.array(
            [
                g 
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g]!=0
                and g not in foot_set
                and self.model.geom_contype[g]!=0
            ],
            dtype=np.int32,
        )

        #躯干碰撞id
        self.trunk_geom_ids=np.array(
            [g for g in self.non_foot_geom_ids if self.model.geom_bodyid[g]==self.base_body_id],
            dtype=np.int32,
        )

        self.max_stride_length=self.measure_stride_length()
        self.command_sampler=CommandSampler(
            ranges=command_ranges or CommandRanges(),
            stand_probability=stand_probability,
            resample_interval_s=command_resample_interval_s,
            gaits=tuple(gaits),
            max_speed_per_hz=max_speed_per_hz,
            max_speed=max_speed,
            feasibility_margin=feasibility_margin,
        )
        self.command=Command()
        self.gait_phase=0.0
        self._gait_offsets,self._gait_duty=gait_module.gait_params(self.command.gait)

        self.reward_weights=reward_module.resolve_weights(reward_weights)
        self.lin_vel_sigma=float(lin_vel_sigma)
        self.ang_vel_sigma=float(ang_vel_sigma)

        # Domain randomisation默认值
        self.default_geom_friction=self.model.geom_friction.copy()
        self.default_body_mass=self.model.body_mass.copy()
        self._kp_scale=1.0
        self._kd_scale=1.0

        # 动作状态空间
        self.action_space=spaces.Box(low=-1.0,high=1.0,shape=(self.n_jnts,),dtype=np.float32)
        self.obs_dim=14+self.n_jnts*3
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dim,),dtype=np.float32)

        # episode状态
        self.step_cnt=0
        self.prev_action=np.zeros(self.n_jnts)
        self.curr_action=np.zeros(self.n_jnts)
        self.prev_jnt_vel=np.zeros(self.n_jnts)
        self.last_troques=np.zeros(self.n_jnts)
        self.feet_air_time=np.zeros(4)
        self.last_contact=np.zeros(4)
        self.time_since_command=0.0
        self.push_countdown=0.0
        self.push_remaining=0.0
        self.reward_terms={}
        self.tracking_scores=[]

        self.lin_vel_base=np.zeros(3)
        self.ang_vel_base=np.zeros(3)
        self.proj_gravity=self.WORLD_GRAVITY_DIR.copy()

        self.viewer=None
        self.renderer=None
        self.camera=None

    def set_command_ranges(self,ranges):
        if isinstance(ranges,dict):
            ranges=CommandRanges(**ranges)
        self.command_sampler.ranges=ranges

    # 设置命令
    '''
    x方向速度
    y方向速度
    yaw方向角速度
    步态
    步频
    机身高度
    '''
    def set_command(
        self,
        lin_vel_x=None,
        lin_vel_y=None,
        ang_vel_yaw=None,
        gait=None,
        gait_frequency=None,
        base_height=None
    ):
        if lin_vel_x is not None:
            self.command.lin_vel_x=float(lin_vel_x)
        if lin_vel_y is not None:
            self.command.lin_vel_y=float(lin_vel_y)
        if ang_vel_yaw is not None:
            self.command.ang_vel_yaw=float(ang_vel_yaw)
        if gait is not None:
            self.command.gait=str(gait)
        if gait_frequency is not None:
            self.command.gait_frequency=float(gait_frequency)
        if base_height is not None:
            self.command.base_height=float(base_height)
        self._apply_gait(self.command.gait)
        self.time_since_command=0.0

    # 设置速度命令
    '''
    x方向速度
    y方向速度
    yaw方向角速度
    '''
    def set_cmd_vel(self,cmd_vel):
        self.set_command(lin_vel_x=cmd_vel[0],lin_vel_y=cmd_vel[1],ang_vel_yaw=cmd_vel[2])

    # 读取命令
    def get_command(self):
        return self.command

    # 上一rollout奖励平均
    def _last_tracking_score(self):
        if not self.tracking_scores:
            return None
        score=float(np.mean(self.tracking_scores))
        self.tracking_scores.clear()
        return score

    # 应用步态
    '''
    根据步态的名字,从GAIT_TERMS里面取出对应的offsets和duty应用
    '''
    def _apply_gait(self,type:str):
        self._gait_offsets,self._gait_duty=gait_module.gait_params(type)

    # 计算步长
    '''
    通过计算最大的步长,从而可以通过步长乘步频的方式计算出最大的速度,从而识别出
    超过最大速度的速度指令
    计算只用到左前腿的关节
    z_tolerance用来计算在脚可以支撑地面的情况下可以向前向后扩展多远,这才算是步幅
    '''
    def measure_stride_length(self,samples=41,z_tolerance=0.3):
        data=mujoco.MjData(self.model)
        geom_id=int(self.foot_geom_ids[0])
        thigh_jnt_id=int(self.actuated_jnt_ids[1])
        calf_jnt_id=int(self.actuated_jnt_ids[2])
        t_lo,t_hi=self.model.jnt_range[thigh_jnt_id]
        c_lo,c_hi=self.model.jnt_range[calf_jnt_id]

        thigh_addr=int(self.model.jnt_qposadr[thigh_jnt_id])
        calf_addr=int(self.model.jnt_qposadr[calf_jnt_id])

        def foot_xz(dth,dca):
            mujoco.mj_resetDataKeyframe(self.model,data,self._home_key_id)
            data.qpos[thigh_addr]+=dth
            data.qpos[calf_addr]+=dca
            mujoco.mj_kinematics(self.model,data)
            p=data.geom_xpos[geom_id]-data.xpos[self.base_body_id]
            return p[0],p[2]

        z0=foot_xz(0.0,0.0)[1]
        a=self.action_scale
        xs=[]
        for dth in np.linspace(-a,a,samples):
            if not (t_lo<=self.default_jnt_pos[1]+dth<=t_hi):
                continue
            for dca in np.linspace(-a,a,samples):
                if not (c_lo<=self.default_jnt_pos[2]+dca<=c_hi):
                    continue
                x,z=foot_xz(dth,dca)
                if abs(z-z0)<z_tolerance:
                    xs.append(x)
        return float(max(xs)-min(xs)) if xs else 0.0

    # 期望脚腾空时间
    def _target_swing_time(self):
        freq=max(float(self.command.gait_frequency),1e-3)
        return (1.0-self._gait_duty)/freq

    # 叉乘
    @staticmethod
    def _cross3(a0,a1,a2,b0,b1,b2):
        return (a1*b2-a2*b1,
                a2*b0-a0*b2,
                a0*b1-a1*b0)

    # 逆旋转
    '''
    quat:机器人当前的朝向
    vec:世界坐标系中的三维向量
    '''
    @staticmethod
    def _quat_rotate_inverse(quat,vec):
        w=float(quat[0])
        ux,uy,uz=float(quat[1]),float(quat[2]),float(quat[3])
        vx,vy,vz=float(vec[0]),float(vec[1]),float(vec[2])
        tx,ty,tz=Go2Env._cross3(ux,uy,uz,vx,vy,vz)
        tx,ty,tz=2.0*tx,2.0*ty,2.0*tz
        mx,my,mz=Go2Env._cross3(ux,uy,uz,tx,ty,tz)
        return np.array([vx-w*tx+mx,vy-w*ty+my,vz-w*tz+mz])

    # 更新机器人信息
    '''
    更新机器人坐标系下的速度、角速度、重力
    '''
    def _refresh_frame_cache(self):
        q=self.data.qpos[3:7]
        self.lin_vel_base=self._quat_rotate_inverse(q,self.data.qvel[0:3])
        self.ang_vel_base=self._quat_rotate_inverse(q,self.data.qvel[3:6])
        self.proj_gravity=self._quat_rotate_inverse(q,self.WORLD_GRAVITY_DIR)

    # 读取机器人坐标系下的线速度
    def _base_lin_vel(self):
        return self.lin_vel_base

    # 读取机器人坐标系下的角速度
    def _base_ang_vel(self):
        return self.ang_vel_base

    # 读取机器人坐标系下的重力方向
    def _base_proj_gravity(self):
        return self.proj_gravity

    # 返回碰撞情况
    '''
    contacts:足部与地面的碰撞情况
    undesired:非足部与地面碰撞
    trunk:躯干是否和地面碰撞
    '''
    def _contact_state(self,force_threshold=1.0):
        contacts=np.zeros(4)
        undesired=0.0
        trunk=False

        foot_ids=self.foot_geom_ids
        force=np.zeros(6)
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            g1,g2=c.geom1,c.geom2
            if g1==self.floor_geom_id:
                other=g2
            elif g2==self.floor_geom_id:
                other=g1
            else:
                continue
            mujoco.mj_contactForce(self.model,self.data,i,force)
            if abs(force[0])<force_threshold:
                continue
            hit=np.where(foot_ids==other)[0]
            if hit.size:
                contacts[hit[0]]=1.0
            else:
                undesired+=1.0
                if other in self.trunk_geom_ids:
                    trunk=True
        return contacts,undesired,trunk

    # 脚高度
    def _foot_height(self):
        return self.data.geom_xpos[self.foot_geom_ids][:,2]

    # 脚xy方向的速度
    def _foot_vel_xy(self):
        out=np.zeros((4,2))
        temp=np.zeros(6)
        for i,geom_id in enumerate(self.foot_geom_ids):
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_GEOM,int(geom_id),temp,0)
            out[i]=temp[3:5]
        return out

    # 获取观测
    '''
    3:投影重力方向
    3:基座角速度
    3:基座线速度
    12:当前关节位置
    12:当前关节速度
    12:上一步动作
    3:速度命令
    2:步态时钟
    '''
    def _get_obs(self):
        jnt_pos=self.data.qpos[7:]-self.default_jnt_pos
        jnt_vel=self.data.qvel[6:]
        phase=2.0*np.pi*self.gait_phase
        obs=np.concatenate(
            [
                self._base_proj_gravity(),
                self._base_ang_vel()*self.OBS_SCALE_ANG_VEL,
                self._base_lin_vel()*self.OBS_SCALE_LIN_VEL,
                jnt_pos,
                jnt_vel*self.OBS_SCALE_JNT_VEL,
                self.prev_action,
                self.command.vel*self.OBS_SCALE_CMD,
                [np.sin(phase),np.cos(phase)],
            ]
        )
        if self.obs_noise_scale>0.0:
            obs=obs+self.np_random.normal(0.0,self.obs_noise_scale,obs.shape)
        return obs.astype(np.float32)

    def _build_reward_state(self,contacts,undesired,new_contact,foot_air_time):
        desired=gait_module.desired_contact(self.gait_phase,self._gait_offsets,self._gait_duty)
        return reward_module.RewardState(
            lin_vel_base=self._base_lin_vel(),
            ang_vel_base=self._base_ang_vel(),
            proj_gravity=self._base_proj_gravity(),
            base_height=float(self.data.qpos[2]),
            joint_pos=self.data.qpos[7:].copy(),
            joint_vel=self.data.qvel[6:].copy(),
            joint_vel_prev=self.prev_jnt_vel.copy(),
            default_joint_pos=self.default_jnt_pos,
            soft_joint_limits=self.soft_jnt_limits,
            torques=self.last_troques.copy(),
            action_curr=self.curr_action.copy(),
            action_prev=self.prev_action.copy(),
            contact=contacts,
            desired_contact=desired,
            foot_air_time=foot_air_time,
            foot_new_contact=new_contact,
            foot_vel_xy=self._foot_vel_xy(),
            foot_height=self._foot_height(),
            target_swing_time=self._target_swing_time(),
            cmd=np.concatenate([self.command.vel,[self.command.base_height]]),
            undesired_contacts=undesired,
            dt=self.dt,
            lin_vel_sigma=self.lin_vel_sigma,
            ang_vel_sigma=self.ang_vel_sigma,
        )

    # 随机化
    '''
    摩擦系数
    质量
    PD控制参数
    '''
    def _randomize_domain(self):
        self._kd_scale=1.0
        self._kp_scale=1.0
        if not self.randomize_dynamics:
            return
        # 摩擦力
        friction_scale=self.np_random.uniform(0.4,1.4)
        for gid in list(self.foot_geom_ids)+[self.floor_geom_id]:
            self.model.geom_friction[gid]=self.default_geom_friction[gid]*friction_scale
        # 质量
        self.model.body_mass[:]=self.default_body_mass
        self.model.body_mass[self.base_body_id]=self.default_body_mass[self.base_body_id]+self.np_random.uniform(-1.0,2.0)
        # 控制参数
        self._kp_scale=float(self.np_random.uniform(0.8,1.2))
        self._kd_scale=float(self.np_random.uniform(0.8,1.2))

    # 规划推力
    '''
    从期望的推力时间中随机选取一个时间,作为当前时刻到推力到来的时间间隔
    然后将推力时间初始化为0
    '''
    def _schedule_push(self):
        lo,hi=self.push_interval_s
        self.push_countdown=float(self.np_random.uniform(lo,hi))
        self.push_remaining=0.0

    # 更新推力
    '''
    每个策略控制步都会调用一次

    '''
    def _update_push(self):
        if not self.push_enable:
            return
        if self.push_remaining>0.0:
            self.push_remaining-=self.dt
            if self.push_remaining<=0:
                self.data.xfrc_applied[self.base_body_id,:3]=0.0
                self._schedule_push()
            return
        self.push_countdown-=self.dt
        if self.push_countdown<=0:
            f=self.np_random.uniform(self.push_force[0],self.push_force[1],size=3)
            f[2]*=0.25
            self.data.xfrc_applied[self.base_body_id,:3]=f
            self.push_remaining=self.push_duration_s

    @staticmethod
    def _euler_to_quat(roll,pitch,yaw):
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    '''
    结束上一轮episode,并生成一个新的初始状态
    初始并随机化:关节角度,机身高度,roll,pitch,yaw,基座线速度,基座角速度,关节角速度
    清除推力
    随机化物理参数,包含:PD控制参,摩擦系数,质量
    防止脚生成在地面以下
    采样本局命令与步态
    返回初始obs
    '''
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model,self.data,self._home_key_id)
        n=self.reset_noise_scale
        self.data.qpos[7:]+=self.np_random.uniform(-n,n,size=self.n_jnts)
        self.data.qpos[2]+=self.np_random.uniform(-0.02,0.05)
        yaw=self.np_random.uniform(-np.pi,np.pi)
        roll_pitch=self.np_random.uniform(-0.05,0.05,size=2)
        self.data.qpos[3:7]=self._euler_to_quat(roll_pitch[0],roll_pitch[1],yaw)
        self.data.qvel[0:3]=self.np_random.uniform(-0.3,0.3,size=3)
        self.data.qvel[3:6]=self.np_random.uniform(-0.3,0.3,size=3)
        self.data.qvel[6:]=self.np_random.uniform(-0.5,0.5,size=self.n_jnts)
        self.data.xfrc_applied[:]=0.0

        self._randomize_domain()
        # 因为修改了模型参数，所以需要重新计算
        mujoco.mj_forward(self.model,self.data)

        # 防止脚生成在地面以下
        foot_z=self.data.geom_xpos[self.foot_geom_ids][:,2]-self.foot_radius
        lift=max(0.0,0.002-float(foot_z.min()))
        if lift>0.0:
            self.data.qpos[2]+=lift
            mujoco.mj_forward(self.model,self.data)

        self._refresh_frame_cache()
        self.command=self.command_sampler.sample(self.np_random)
        self._apply_gait(self.command.gait)

        self.gait_phase=float(self.np_random.uniform(0.0,1.0))

        self.step_cnt=0
        self.prev_action[:]=0.0
        self.curr_action[:]=0.0
        self.prev_jnt_vel[:]=self.data.qvel[6:]
        self.last_troques[:]=0.0
        self.feet_air_time[:]=0.0
        self.last_contact[:]=0.0
        self.time_since_command=0.0
        self.reward_terms={}
        self.tracking_scores.clear()
        self._schedule_push()
        return self._get_obs(),{}

    # 检查episode是否终止
    '''
    躯干是否触地
    基座过低
    躯干倾斜过大
    '''
    def _check_termination(self,trunk:bool):
        if trunk:
            return True
        if self.data.qpos[2]<self.base_height_min:
            return True
        gz=self._base_proj_gravity()[2]
        return bool(gz>-np.cos(self.pitch_roll_max))

    '''
    执行一次策略控制步并收集obs

    '''
    def step(self,action):
        action=np.clip(np.asarray(action,dtype=np.float64),-1.0,1.0)
        self.curr_action=action.copy()
        target_pos=self.default_jnt_pos+action*self.action_scale
        kp=self.kp*self._kp_scale
        kd=self.kd*self._kd_scale
        self._update_push()
        for _ in range(self.decimation):
            torque=kp*(target_pos-self.data.qpos[7:])-kd*self.data.qvel[6:]
            torque=np.clip(torque,-self.torque_limits,self.torque_limits)
            self.data.ctrl[:]=torque
            mujoco.mj_step(self.model,self.data)
        self.last_troques=torque
        self._refresh_frame_cache()

        self.step_cnt+=1
        self.time_since_command+=self.dt
        self.gait_phase=gait_module.step_phase(self.gait_phase,self.command.gait_frequency,self.dt)

        contacts,undesired,trunk=self._contact_state()
        new_contact=((contacts>0.5)&(self.last_contact<0.5)).astype(float)
        air_time_at_landing=self.feet_air_time*new_contact
        self.feet_air_time=np.where(contacts>0.5,0.0,self.feet_air_time+self.dt)
        self.last_contact=contacts

        state=self._build_reward_state(contacts,undesired,new_contact,air_time_at_landing)
        raw_reward,terms=reward_module.compute(state,self.reward_weights)
        reward=float(raw_reward)*self.dt
        self.reward_terms={k: float(v)*self.dt for k,v in terms.items()}
        terminated=self._check_termination(trunk)
        truncated=self.step_cnt>=self.max_episode_steps

        info=self._build_info(state,contacts)

        if self.command_sampler.should_resample(self.time_since_command):
            self.command=self.command_sampler.sample(self.np_random)
            self._apply_gait(self.command.gait)
            self.time_since_command=0.0

        self.prev_jnt_vel[:]=self.data.qvel[6:]
        self.prev_action=action.copy()
        obs=self._get_obs()

        if self.render_mode=="human":
            self.render()
        return obs,reward,terminated,truncated,info

    # 创建诊断信息
    def _build_info(self,state,contacts):
        lin_err=float(np.linalg.norm(state.cmd[:2] - state.lin_vel_base[:2]))
        ang_err = float(abs(state.cmd[2] - state.ang_vel_base[2]))
        info = {
            "x_position": float(self.data.qpos[0]),
            "base_height": float(self.data.qpos[2]),
            "lin_vel_b": state.lin_vel_base.copy(),
            "ang_vel_b": state.ang_vel_base.copy(),
            "cmd": state.cmd.copy(),
            "gait": self.command.gait,
            "gait_phase": self.gait_phase,
            "contacts": contacts.copy(),
            "tracking_lin_err": lin_err,
            "tracking_ang_err": ang_err,
            "tracking_score": float(
                np.exp(-lin_err ** 2 / self.lin_vel_sigma) * 0.5
                + np.exp(-ang_err ** 2 / self.ang_vel_sigma) * 0.5
            ),
        }
        self.tracking_scores.append(info["tracking_score"])
        for k,v in self.reward_terms.items():
            info["rew/"+k]=v
        return info

    # 可视化
    def render(self):
        if self.render_mode=="human":
            if self.viewer is None:
                w,h=self.render_size
                self.renderer=mujoco.Renderer(self.model,height=h,width=w)
                self.camera=mujoco.MjvCamera()
                self.camera.type=mujoco.mjtCamera.mjCAMERA_TRACKING
                self._camera.trackbodyid = self.base_body_id
                self._camera.distance = 2.0
                self._camera.azimuth = 135
                self._camera.elevation = -20
            self.renderer.update_scene(self.data, camera=self.camera)
            return self.renderer.render()
        return None

    # 释放渲染资源
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer=None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer=None
            