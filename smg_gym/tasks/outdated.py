
useRelativeControl: True
dofSpeedScale: 10.0
actionsMovingAverage: 1.0


def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

     action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

      # apply actions
      self.pre_physics_step(action_tensor)

       # step physics and render each frame
       for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                       torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


def apply_actions(self):

    if self.use_relative_control:
        targets = self.prev_targets[:, self.control_joint_dof_indices] + self.dof_speed_scale * self.dt * self.action_buf

        self.cur_targets[:, self.control_joint_dof_indices] = tensor_clamp(
            targets,
            self.hand_dof_lower_limits[self.control_joint_dof_indices],
            self.hand_dof_upper_limits[self.control_joint_dof_indices],
        )
    else:

        self.cur_targets[:, self.control_joint_dof_indices] = scale(
            self.action_buf,
            self.hand_dof_lower_limits[self.control_joint_dof_indices],
            self.hand_dof_upper_limits[self.control_joint_dof_indices],
        )

        self.cur_targets[:, self.control_joint_dof_indices] = (
            self.act_moving_average * self.cur_targets[:, self.control_joint_dof_indices]
            + (1.0 - self.act_moving_average) * self.prev_targets[:, self.control_joint_dof_indices]
        )

        self.cur_targets[:, self.control_joint_dof_indices] = tensor_clamp(
            self.cur_targets[:, self.control_joint_dof_indices],
            self.hand_dof_lower_limits[self.control_joint_dof_indices],
            self.hand_dof_upper_limits[self.control_joint_dof_indices],
        )

    self.prev_targets[:, self.control_joint_dof_indices] = self.cur_targets[:, self.control_joint_dof_indices]
    self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


# dimensions of the system
_dims = SMGObjectTaskDimensions

# Constants for limits
# Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/trifinger_platform.py#L68
# maximum joint torque (in N-m) applicable on each actuator
_max_torque_Nm = 0.36
# maximum joint velocity (in rad/s) on each actuator
_max_velocity_radps = 10

# limits of the robot (mapped later: str -> torch.tensor)
_robot_limits: dict = {
    "joint_position": SimpleNamespace(
        # matches those on the real robot
        low=np.array([-0.785, -1.396, -1.047] * _dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.785, 1.047, 1.57] * _dims.NumFingers.value, dtype=np.float32),
        default=np.array([0.0, 0.9, -1.7] * _dims.NumFingers.value, dtype=np.float32),
    ),
    "joint_velocity": SimpleNamespace(
        low=np.full(_dims.JointVelocityDim.value, -_max_velocity_radps, dtype=np.float32),
        high=np.full(_dims.JointVelocityDim.value, _max_velocity_radps, dtype=np.float32),
        default=np.zeros(_dims.JointVelocityDim.value, dtype=np.float32),
    ),
    "joint_torque": SimpleNamespace(
        low=np.full(_dims.JointTorqueDim.value, -_max_torque_Nm, dtype=np.float32),
        high=np.full(_dims.JointTorqueDim.value, _max_torque_Nm, dtype=np.float32),
        default=np.zeros(_dims.JointTorqueDim.value, dtype=np.float32),
    ),
    "fingertip_position": SimpleNamespace(
        low=np.array([-0.4, -0.4, 0] * _dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.4, 0.4, 0.5] * _dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_orientation": SimpleNamespace(
        low=-np.ones(4 * _dims.NumFingers.value, dtype=np.float32),
        high=np.ones(4 * _dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_velocity": SimpleNamespace(
        low=np.full(_dims.VelocityDim.value, -0.2, dtype=np.float32),
        high=np.full(_dims.VelocityDim.value, 0.2, dtype=np.float32),
    ),
    "fingertip_wrench": SimpleNamespace(
        low=np.full(_dims.WrenchDim.value, -1.0, dtype=np.float32),
        high=np.full(_dims.WrenchDim.value, 1.0, dtype=np.float32),
    ),
    # used if we want to have joint stiffness/damping as parameters`
    "joint_stiffness": SimpleNamespace(
        low=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
        high=np.array([50.0, 50.0, 50.0] * _dims.NumFingers.value, dtype=np.float32),
    ),
    "joint_damping": SimpleNamespace(
        low=np.array([0.01, 0.03, 0.0001] * _dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 3.0, 0.01] * _dims.NumFingers.value, dtype=np.float32),
    ),
    "bool_tip_contacts": SimpleNamespace(
        low=np.zeros(_dims.NumFingers.value, dtype=np.float32),
        high=np.ones(_dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * _dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
    ),
}

# limits of the object (mapped later: str -> torch.tensor)
_object_limits: dict = {
    "position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0], dtype=np.float32),
        high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
    ),
    "orientation": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "keypoint_position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0] * _dims.NumKeypoints.value, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.3] * _dims.NumKeypoints.value, dtype=np.float32),
    ),
    "linear_velocity": SimpleNamespace(
        low=np.full(_dims.LinearVelocityDim.value, -0.5, dtype=np.float32),
        high=np.full(_dims.LinearVelocityDim.value, 0.5, dtype=np.float32),
    ),
    "angular_velocity": SimpleNamespace(
        low=np.full(_dims.AngularVelocityDim.value, -0.5, dtype=np.float32),
        high=np.full(_dims.AngularVelocityDim.value, 0.5, dtype=np.float32),
    ),
}

# limits of the object (mapped later: str -> torch.tensor)
_target_limits: dict = {
    "active_quat": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "pivot_axel_vector": SimpleNamespace(
        low=np.full(_dims.VecDim.value, -1.0, dtype=np.float32),
        high=np.full(_dims.VecDim.value, 1.0, dtype=np.float32),
    ),
    "pivot_axel_position": SimpleNamespace(
        low=np.full(_dims.PosDim.value, -0.025, dtype=np.float32),
        high=np.full(_dims.PosDim.value, 0.025, dtype=np.float32),
    ),
}

# PD gains for the robot (mapped later: str -> torch.tensor)
# Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L49-L65
_robot_dof_gains = {
    # The kp and kd gains of the PD control of the fingers.
    # Note: This depends on simulation step size and is set for a rate of 250 Hz.
    "stiffness": [10.0, 10.0, 10.0] * _dims.NumFingers.value,
    "damping": [0.1, 0.3, 0.001] * _dims.NumFingers.value,
    # The kd gains used for damping the joint motor velocities during the
    # safety torque check on the joint motors.
    "safety_damping": [0.08, 0.08, 0.04] * _dims.NumFingers.value
}
