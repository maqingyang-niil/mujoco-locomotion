"""Flat-ground velocity tracking for the Menagerie ANYmal-C."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class AnymalFlatEnv(gym.Env):
    """Train on flat ground and replay specific conditions via reset options.

    Friction and mass scale are sampled once per episode. With push_force > 0,
    one horizontal push is sampled per episode. All pushes use world axes.
    """

    def __init__(
        self,
        *,
        friction_range=None,
        mass_scale_range=None,
        push_force=0.0,
        target_speed=0.5,
        target_lateral_speed=0.0,
        lateral_speed_range=None,
        target_yaw_rate=0.0,
        yaw_rate_range=None,
    ):
        super().__init__()
        models = Path(__file__).resolve().parents[1] / "models" / "anybotics_anymal_c"
        self.model = mujoco.MjModel.from_xml_path(str(models / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base").id

        joint_ids = self.model.actuator_trnid[:, 0]
        self.joint_qpos = self.model.jnt_qposadr[joint_ids]
        self.joint_qvel = self.model.jnt_dofadr[joint_ids]
        self.standing_qpos = self.model.qpos0.copy()
        self.standing_qpos[2] = 0.56
        # Joint targets follow the Menagerie standing pose in anymal_c_mjx.xml.
        self.standing_targets = np.array(
            [0, np.pi / 6, -np.pi / 4] * 2 + [0, -np.pi / 6, np.pi / 4] * 2
        )
        self.standing_qpos[self.joint_qpos] = self.standing_targets

        self.foot_ids = np.flatnonzero(
            (self.model.geom_priority == 1)
            & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            & (self.model.geom_contype != 0)
        )
        if len(self.foot_ids) != 4:
            raise ValueError("Expected four Menagerie foot collision geoms")
        self.floor_id = self.model.geom("floor").id
        self.nominal_friction = self.model.geom_friction[self.foot_ids, 0].copy()
        self.nominal_mass = self.model.body_mass[self.base_id]
        self.nominal_inertia = self.model.body_inertia[self.base_id].copy()
        self.friction_range = friction_range
        self.mass_scale_range = mass_scale_range
        self.push_force = push_force
        self.target_speed = target_speed
        self.target_lateral_speed = target_lateral_speed
        self.lateral_speed_range = lateral_speed_range
        self.target_yaw_rate = target_yaw_rate
        self.yaw_rate_range = yaw_rate_range

        self.frame_skip = 10
        self.episode_seconds = 10.0
        self.action_space = spaces.Box(-1, 1, shape=(self.model.nu,), dtype=np.float32)
        # Base height, local gravity, local velocities, joint state, velocity commands.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(37,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        friction = options.get("friction")
        if friction is None and self.friction_range is not None:
            friction = self.np_random.uniform(*self.friction_range)
        mass_scale = options.get("mass_scale")
        if mass_scale is None:
            mass_scale = (
                1.0 if self.mass_scale_range is None
                else self.np_random.uniform(*self.mass_scale_range)
            )
        if friction is not None and (not np.isfinite(friction) or friction < 0):
            raise ValueError("friction must be finite and nonnegative")
        if not np.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError("mass_scale must be finite and positive")

        self.model.geom_friction[self.foot_ids, 0] = (
            self.nominal_friction if friction is None else friction
        )
        self.model.body_mass[self.base_id] = self.nominal_mass * mass_scale
        self.model.body_inertia[self.base_id] = self.nominal_inertia * mass_scale
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.standing_qpos
        self.data.ctrl[:] = self.standing_targets
        mujoco.mj_forward(self.model, self.data)
        self.foot_contacts = self._foot_ground_contacts()
        self.feet_air_time = np.zeros(len(self.foot_ids))
        self.feet_landed = np.zeros(len(self.foot_ids), dtype=bool)
        self.feet_landing_air_time = np.zeros(len(self.foot_ids))
        self.previous_action = np.zeros(self.action_space.shape)

        self.command = float(options.get("target_speed", self.target_speed))
        if "target_lateral_speed" in options:
            self.lateral_command = float(options["target_lateral_speed"])
        elif self.lateral_speed_range is not None:
            self.lateral_command = float(self.np_random.uniform(*self.lateral_speed_range))
        else:
            self.lateral_command = float(self.target_lateral_speed)
        if not np.isfinite(self.lateral_command):
            raise ValueError("target_lateral_speed must be finite")
        if "target_yaw_rate" in options:
            self.yaw_rate_command = float(options["target_yaw_rate"])
        elif self.yaw_rate_range is not None:
            self.yaw_rate_command = float(self.np_random.uniform(*self.yaw_rate_range))
        else:
            self.yaw_rate_command = float(self.target_yaw_rate)
        if not np.isfinite(self.yaw_rate_command):
            raise ValueError("target_yaw_rate must be finite")
        self.push = options.get("push")
        if "push" not in options and self.push_force > 0:
            angle = self.np_random.uniform(0, 2 * np.pi)
            self.push = (
                self.np_random.uniform(2, 8),
                0.15,
                self.push_force * np.cos(angle),
                self.push_force * np.sin(angle),
            )
        self.conditions = {
            "foot_friction": tuple(self.model.geom_friction[self.foot_ids, 0]),
            "mass_scale": float(mass_scale),
            "push": self.push,
        }
        return self._observation(), self._info(False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action with shape {self.action_space.shape}")
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = self.standing_targets + 0.35 * action
        push_applied = False
        for _ in range(self.frame_skip):
            if self.push is not None:
                start, duration, fx, fy = self.push
                if start <= self.data.time < start + duration:
                    self.data.xfrc_applied[self.base_id, :2] = (fx, fy)
                    push_applied = True
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[self.base_id, :2] = 0
        mujoco.mj_forward(self.model, self.data)

        self._update_feet_air_time()
        info = self._info(push_applied)
        info["action_delta"] = action - self.previous_action
        upright = self.data.xmat[self.base_id, 8]
        terminated = bool(info["base_height"] < 0.27 or upright < 0.5)
        truncated = bool(self.data.time >= self.episode_seconds - 1e-9)

        # ==================== 奖励计算开始 ====================
        dt=self.model.opt.timestep*self.frame_skip
        #奖励前进、侧移速度接近指令
        forward_error=info["speed_error"]
        lateral_error=info["lateral_speed_error"]
        track_lin_vel_xy_exp_reward=np.exp(-4.0*(forward_error**2+lateral_error**2))
        #奖励转向角速度接近指令
        yaw_rate_error=info["yaw_rate_error"]
        track_ang_vel_z_exp_reward=0.5*np.exp(-4.0*(yaw_rate_error**2))
        #脚腾空时间奖励
        linear_command_speed=np.hypot(info["target_speed"],info["target_lateral_speed"])
        feet_air_time_reward=0.0
        if linear_command_speed>0.1:
            landing_air_time=np.asarray(info["feet_landing_air_time"])
            feet_air_time_reward=0.5*(np.maximum(landing_air_time-0.2,0.0)).sum()
        #惩罚机身倾斜
        gravity_local=-self.data.xmat[self.base_id,6:9]
        gx,gy=gravity_local[:2]
        flat_orientation_l2_penalty=-0.5*(gx**2+gy**2)
        #惩罚机身的上下速度
        vertical_speed=self.data.qvel[2]
        lin_vel_z_l2_penalty=-1.0*(vertical_speed**2)
        #惩罚机身滚转俯仰角速度
        angular_vel=self._base_velocity()[:3]
        ang_vel_x,ang_vel_y=angular_vel[:2]
        ang_vel_xy_l2_penalty=-0.05*(ang_vel_x**2+ang_vel_y**2)
        #惩罚各关节力矩
        joint_torques=self.data.qfrc_actuator[self.joint_qvel]
        dof_torques_l2_penalty=-0.000025*np.sum(joint_torques**2)
        #惩罚各关节加速度
        joint_acc=self.data.qacc[self.joint_qvel]
        dof_acc_l2_penalty=-0.00000025*np.sum(joint_acc**2)
        #惩罚相邻两步的动作差
        action_delta=info["action_delta"]
        action_rate_l2_penalty=-0.05*np.sum(action_delta**2)
        #非期望接触惩罚
        undesired_contact_penalty=-0.2*info["undesired_contact_count"]

        reward = (
            track_lin_vel_xy_exp_reward
            + track_ang_vel_z_exp_reward
            + feet_air_time_reward
            + flat_orientation_l2_penalty
            + lin_vel_z_l2_penalty
            + ang_vel_xy_l2_penalty
            + dof_torques_l2_penalty
            + dof_acc_l2_penalty
            + action_rate_l2_penalty
            + undesired_contact_penalty
        )
        # ==================== 奖励计算结束 ====================

        self.previous_action = action.copy()
        return self._observation(), float(reward), terminated, truncated, info

    def _ground_contact_geoms(self):
        ground_contacts = set()
        for contact in self.data.contact:
            geom_a, geom_b = contact.geom
            if geom_a == self.floor_id:
                ground_contacts.add(geom_b)
            elif geom_b == self.floor_id:
                ground_contacts.add(geom_a)
        return ground_contacts

    def _foot_ground_contacts(self):
        ground_contacts = self._ground_contact_geoms()
        return np.array([foot in ground_contacts for foot in self.foot_ids])

    def _update_feet_air_time(self):
        contacts = self._foot_ground_contacts()
        dt = self.model.opt.timestep * self.frame_skip
        self.feet_landed = contacts & (self.feet_air_time > 0.0)
        self.feet_landing_air_time = np.where(
            self.feet_landed, self.feet_air_time, 0.0
        )
        self.feet_air_time = np.where(contacts, 0.0, self.feet_air_time + dt)
        self.foot_contacts = contacts

    def _base_velocity(self):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1
        )
        return velocity

    def _observation(self):
        gravity_local = -self.data.xmat[self.base_id, 6:9]
        return np.concatenate(
            (
                self.data.qpos[2:3],
                gravity_local,
                self._base_velocity(),
                self.data.qpos[self.joint_qpos] - self.standing_targets,
                self.data.qvel[self.joint_qvel],
                [self.command, self.lateral_command, self.yaw_rate_command],
            ),
            dtype=np.float32,
        )

    def _info(self, push_applied):
        velocity = self._base_velocity()
        forward_speed = float(velocity[3])
        yaw_rate = float(velocity[2])
        undesired_body_ids = {
            int(self.model.geom_bodyid[geom_id])
            for geom_id in self._ground_contact_geoms().difference(self.foot_ids)
        }
        undesired_body_ids.discard(0)
        return {
            **self.conditions,
            "time": float(self.data.time),
            "base_height": float(self.data.qpos[2]),
            "target_speed": self.command,
            "target_lateral_speed": self.lateral_command,
            "target_yaw_rate": self.yaw_rate_command,
            "yaw_rate": yaw_rate,
            "yaw_rate_error": yaw_rate - self.yaw_rate_command,
            "forward_speed": forward_speed,
            "lateral_speed": float(velocity[4]),
            "speed_error": forward_speed - self.command,
            "lateral_speed_error": float(velocity[4]) - self.lateral_command,
            "push_applied": push_applied,
            "undesired_contact_bodies": tuple(
                self.model.body(body_id).name for body_id in sorted(undesired_body_ids)
            ),
            "undesired_contact_count": len(undesired_body_ids),
            "foot_contacts": tuple(self.foot_contacts),
            "feet_air_time": tuple(self.feet_air_time),
            "feet_landed": tuple(self.feet_landed),
            "feet_landing_air_time": tuple(self.feet_landing_air_time),
        }
