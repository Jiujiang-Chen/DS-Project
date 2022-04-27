
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


# used to randomise the initial pose of the hand
if self.randomize and self.rand_hand_joints:
    self.init_joint_mins = to_torch(np.array([
        -20.0*(np.pi/180),
        7.5*(np.pi/180),
        -10.0*(np.pi/180),
    ] * 3), device=self.device)

    self.init_joint_maxs = to_torch(np.array([
        20.0*(np.pi/180),
        7.5*(np.pi/180),
        -10.0*(np.pi/180),
    ] * 3), device=self.device)

else:
    self.init_joint_mins = to_torch(np.array([
        0.0*(np.pi/180),
        7.5*(np.pi/180),
        -10.0*(np.pi/180),
    ] * 3), device=self.device)

    self.init_joint_maxs = to_torch(np.array([
        0.0*(np.pi/180),
        7.5*(np.pi/180),
        -10.0*(np.pi/180),
    ] * 3), device=self.device)


# contacts
tip_1_contact_pos, tip_2_contact_pos, tip_3_contact_pos = [], [], []
tip_1_contact_force, tip_2_contact_force, tip_3_contact_force = [], [], []
for contact in contacts:
    body0 = contact['body0']
    body1 = contact['body1']
    tip_contact_pos = contact['localPos1']
    force_mag = contact['lambda']

    if body0 == self.obj_body_idx and body1 == self.tip_body_idxs[0]:
        tip_1_contact_pos.append([tip_contact_pos['x'], tip_contact_pos['y'], tip_contact_pos['z']])
        tip_1_contact_force.append(force_mag)
    if body0 == self.obj_body_idx and body1 == self.tip_body_idxs[1]:
        tip_2_contact_pos.append([tip_contact_pos['x'], tip_contact_pos['y'], tip_contact_pos['z']])
        tip_2_contact_force.append(force_mag)
    if body0 == self.obj_body_idx and body1 == self.tip_body_idxs[2]:
        tip_3_contact_pos.append([tip_contact_pos['x'], tip_contact_pos['y'], tip_contact_pos['z']])
        tip_3_contact_force.append(force_mag)

# average contact positions and forces
self.contact_positions[i, 0, :] = to_torch(np.array(tip_1_contact_pos).mean(axis=0))
self.contact_positions[i, 1, :] = to_torch(np.array(tip_2_contact_pos).mean(axis=0))
self.contact_positions[i, 2, :] = to_torch(np.array(tip_3_contact_pos).mean(axis=0))
self.contact_force_mags[i, 0, :] = to_torch(np.array(tip_1_contact_force).mean(axis=0))
self.contact_force_mags[i, 1, :] = to_torch(np.array(tip_2_contact_force).mean(axis=0))
self.contact_force_mags[i, 2, :] = to_torch(np.array(tip_3_contact_force).mean(axis=0))





















## ========================== Norm observations ============================

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        self.progress_buf += 1

        self.refresh_tensors()
        self.compute_observations()
        self.fill_observation_buffer()
        self.fill_states_buffer()
        self.norm_obs_state_buffer()
        self.compute_reward_and_termination()

        if self.viewer and self.debug_viz:
            self.visualise_features()


    def norm_obs_state_buffer(self):
        if self.cfg["normalize_obs"]:
            # for normal obs
            self._obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )
            # for asymmetric obs
            if self.cfg["asymmetric_obs"]:
                self._states_buf = scale_transform(
                    self.states_buf,
                    lower=self._observations_scale.low,
                    upper=self._observations_scale.high
                )

    def __setup_mdp_spaces(self):
        """
        Configures the observations, state and action spaces.
        """

        # Action scale for the MDP
        # Note: This is order sensitive.
        if self.cfg["env"]["command_mode"] == "position":
            # action space is joint positions
            self._action_scale.low = self._robot_limits["joint_position"].low
            self._action_scale.high = self._robot_limits["joint_position"].high

        elif self.cfg["env"]["command_mode"] == "torque":
            # action space is joint torques
            self._action_scale.low = self._robot_limits["joint_torque"].low
            self._action_scale.high = self._robot_limits["joint_torque"].high

        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)

        # Observations scale for the MDP
        # check if policy outputs normalized action [-1, 1] or not.
        if self.cfg["env"]["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full((self.num_actions,), -1, dtype=torch.float, device=self.device),
                high=torch.full((self.num_actions,), 1, dtype=torch.float, device=self.device)
            )
        else:
            obs_action_scale = self._action_scale

        # Note: This is order sensitive.
        self._observations_scale.low = torch.cat([
            # robot
            self._robot_limits["joint_position"].low,
            self._robot_limits["joint_velocity"].low,
            self._robot_limits["fingertip_position"].low,
            self._robot_limits["fingertip_orientation"].low,

            # action
            obs_action_scale.low,

            # tactile
            self._robot_limits["bool_tip_contacts"].low,
            self._robot_limits["net_tip_contact_forces"].low,
            self._robot_limits["tip_contact_positions"].low,
            self._robot_limits["tip_contact_normals"].low,
            self._robot_limits["tip_contact_force_mags"].low,

            # object
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            self._object_limits["keypoint_position"].low,
            self._object_limits["linear_velocity"].low,
            self._object_limits["angular_velocity"].low,

            # target
            self._object_limits["position"].low,
            self._object_limits["orientation"].low,
            self._object_limits["keypoint_position"].low,
            self._target_limits["active_quat"].low,
            self._target_limits["pivot_axel_vector"].low,
            self._target_limits["pivot_axel_position"].low,
        ])

        self._observations_scale.high = torch.cat([
            # robot
            self._robot_limits["joint_position"].high,
            self._robot_limits["joint_velocity"].high,
            self._robot_limits["fingertip_position"].high,
            self._robot_limits["fingertip_orientation"].high,

            # action
            obs_action_scale.high,

            # tactile
            self._robot_limits["bool_tip_contacts"].high,
            self._robot_limits["net_tip_contact_forces"].high,
            self._robot_limits["tip_contact_positions"].low,
            self._robot_limits["tip_contact_normals"].low,
            self._robot_limits["tip_contact_force_mags"].low,

            # object
            self._object_limits["position"].high,
            self._object_limits["orientation"].high,
            self._object_limits["keypoint_position"].high,
            self._object_limits["linear_velocity"].high,
            self._object_limits["angular_velocity"].high,

            # target
            self._object_limits["position"].high,
            self._object_limits["orientation"].high,
            self._object_limits["keypoint_position"].high,
            self._target_limits["active_quat"].high,
            self._target_limits["pivot_axel_vector"].high,
            self._target_limits["pivot_axel_position"].high,
        ])

        # check that dimensions match
        # observations
        if self._observations_scale.low.shape[0] != self.num_obs or self._observations_scale.high.shape[0] != self.num_obs:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {self.num_obs}."
            raise AssertionError(msg)

        # actions
        if self._action_scale.low.shape[0] != self.num_actions or self._action_scale.high.shape[0] != self.num_actions:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {self.num_actions}."

            raise AssertionError(msg)
