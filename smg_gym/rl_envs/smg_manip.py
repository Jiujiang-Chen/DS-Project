import numpy as np
import os
import torch
from torch import Tensor
from rlgpu.utils.torch_jit_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from smg_gym.rl_games_helpers.utils.torch_jit_utils import randomize_rotation, quat_axis
from smg_gym.rl_envs.base_hand_env import BaseShadowModularGrasper
from smg_gym.assets import get_assets_path, add_assets_path
from pybullet_object_models import primitive_objects as object_set

class SMGManip(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless
    ):
        """
        Obs =
        joint_pos (9)
        joint_vel (9)
        obj_pose (7)
        obj_vel (6)
        prev_actions (9)
        tip_contacts (3)
        obj_keypoint_pos (9)
        tcp_pos (9)
        goal_pose (7)
        goal_keypoint_pos (9)
        rel_goal_orn (4)

        total = 81
        """
        cfg["env"]["numObservations"] = 81
        cfg["env"]["numActions"] = 9

        self.dist_reward_scale = cfg["env"]["distRewardScale"]
        self.rot_reward_scale = cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = cfg["env"]["successTolerance"]
        self.reach_goal_bonus = cfg["env"]["reachGoalBonus"]
        self.fall_penalty = cfg["env"]["fallPenalty"]
        self.rot_eps = cfg["env"]["rotEps"]
        self.max_consecutive_successes = cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = cfg["env"]["avFactor"]

        super(SMGManip, self).__init__(
            cfg,
            sim_params,
            physics_engine,
            device_type,
            device_id,
            headless
        )

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

    def _setup_goal(self):

        model_list = object_set.getModelList()
        asset_root = object_set.getDataPath()
        asset_file = os.path.join(self.obj_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set initial state of goal
        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x,
             self.goal_displacement.y,
             self.goal_displacement.z],
            device=self.device
        )

        self.init_goal_pose = gymapi.Transform()
        self.init_goal_pose.p = self.init_obj_pose.p + self.goal_displacement
        self.init_goal_pose.r = self.init_obj_pose.r

        self.init_goal_vel = gymapi.Velocity()
        self.init_goal_vel.linear = gymapi.Vec3(0.0, 0.0, 0.0)
        self.init_goal_vel.angular = gymapi.Vec3(0.0, 0.0, 0.0)

        return goal_asset

    def _create_goal_actor(self, env, idx):

        handle = self.gym.create_actor(
            env,
            self.goal_asset,
            self.init_goal_pose,
            "goal_actor_{}".format(idx),
            0,
            0
        )

        return handle

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create assets and variables
        self.hand_asset = self._setup_hand()
        self.obj_asset = self._setup_obj()
        self.obj_kp_geoms, self.obj_kp_positions = self._setup_keypoints()
        self.goal_kp_geoms, self.goal_kp_positions = self._setup_keypoints()
        self.goal_asset = self._setup_goal()

        # get indices useful for contacts
        self.n_tips = 3
        self.obj_body_idx, self.tip_body_idxs = self._get_contact_idxs(self.obj_asset, self.hand_asset)
        self.tcp_body_idxs = self._get_sensor_tcp_idxs(self.hand_asset)

        # collect useful indeces and handles
        self.envs = []
        self.hand_actor_handles = []
        self.hand_indices = []
        self.obj_actor_handles = []
        self.obj_indices = []
        self.init_obj_states = []
        self.goal_actor_handles = []
        self.goal_indices = []
        self.init_goal_states = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # setup hand
            hand_actor_handle = self._create_hand_actor(env_ptr, i)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor_handle, gymapi.DOMAIN_SIM)

            # setup obj
            obj_actor_handle = self._create_obj_actor(env_ptr, i)
            obj_idx = self.gym.get_actor_index(env_ptr, obj_actor_handle, gymapi.DOMAIN_SIM)

            # setup goal
            goal_actor_handle = self._create_goal_actor(env_ptr, i)
            goal_idx = self.gym.get_actor_index(env_ptr, goal_actor_handle, gymapi.DOMAIN_SIM)

            # append handles and indeces
            self.envs.append(env_ptr)
            self.hand_actor_handles.append(hand_actor_handle)
            self.hand_indices.append(hand_idx)
            self.obj_actor_handles.append(obj_actor_handle)
            self.obj_indices.append(obj_idx)
            self.goal_actor_handles.append(goal_actor_handle)
            self.goal_indices.append(goal_idx)

            # append states
            self.init_obj_states.append([
                self.init_obj_pose.p.x, self.init_obj_pose.p.y, self.init_obj_pose.p.z,
                self.init_obj_pose.r.x, self.init_obj_pose.r.y, self.init_obj_pose.r.z, self.init_obj_pose.r.w,
                self.init_obj_vel.linear.x, self.init_obj_vel.linear.y, self.init_obj_vel.linear.z,
                self.init_obj_vel.angular.x, self.init_obj_vel.angular.y, self.init_obj_vel.angular.z,
            ])

            self.init_goal_states.append([
                self.init_goal_pose.p.x, self.init_goal_pose.p.y, self.init_goal_pose.p.z,
                self.init_goal_pose.r.x, self.init_goal_pose.r.y, self.init_goal_pose.r.z, self.init_goal_pose.r.w,
                self.init_goal_vel.linear.x, self.init_goal_vel.linear.y, self.init_goal_vel.linear.z,
                self.init_goal_vel.angular.x, self.init_goal_vel.angular.y, self.init_goal_vel.angular.z,
            ])

        # convert indices to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(self.goal_indices, dtype=torch.long, device=self.device)

        # convert states to tensors (TODO: make this more intuitive shape from start)
        self.init_obj_states = to_torch(self.init_obj_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.init_goal_states = to_torch(self.init_goal_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)


    def reset(self, env_ids, goal_env_ids):

        # reset hand
        hand_positions = torch.zeros((len(env_ids), self.n_hand_dofs), device=self.device)
        hand_velocities = torch.zeros((len(env_ids), self.n_hand_dofs), device=self.device)

        # add randomisation to the joint poses
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.n_hand_dofs), device=self.device)
        rand_init_pos = self.init_joint_mins + (self.init_joint_maxs - self.init_joint_mins) * rand_floats[:, :self.n_hand_dofs]

        self.dof_pos[env_ids, :] = rand_init_pos[:]
        self.dof_vel[env_ids, :] = hand_velocities[:]

        self.prev_targets[env_ids, :self.n_hand_dofs] = rand_init_pos
        self.cur_targets[env_ids, :self.n_hand_dofs] = rand_init_pos

        hand_ids_int32 = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_ids_int32),
            len(env_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_ids_int32),
            len(env_ids)
        )

        # reset object
        self.root_state_tensor[self.obj_indices[env_ids]] = self.init_obj_states[env_ids].clone()

        # randomise rotation
        if self.randomize and self.rand_init_orn:
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            new_object_rot = randomize_rotation(
                rand_floats[:, 0],
                rand_floats[:, 1],
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids]
            )
            self.root_state_tensor[self.obj_indices[env_ids], 3:7] = new_object_rot

        # update goal_pos
        self.reset_target_pose(env_ids, apply_reset=False)

        # set the root state tensor to reset object and goal pose
        # has to be done together for some reason...
        reset_indices = torch.unique(torch.cat([self.obj_indices[env_ids],
                                                self.goal_indices[env_ids],
                                                self.goal_indices[goal_env_ids]]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(reset_indices),
            len(reset_indices)
        )

        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_goal_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids]
        )

        self.root_state_tensor[self.goal_indices[env_ids]] = self.init_goal_states[env_ids].clone()
        self.root_state_tensor[self.goal_indices[env_ids], 3:7] = new_goal_rot

        if apply_reset:
            goal_object_indices = self.goal_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(goal_object_indices),
                len(env_ids)
            )

        self.reset_goal_buf[env_ids] = 0

    def pre_physics_step(self, actions):

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then apply reset in targ pose
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, apply reset in reset() func
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids, apply_reset=False)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.use_relative_control:
            targets = self.prev_targets[:, self.control_joint_dof_indices] + self.dof_speed_scale * self.dt * self.actions

            self.cur_targets[:, self.control_joint_dof_indices] = tensor_clamp(
                targets,
                self.hand_dof_lower_limits[self.control_joint_dof_indices],
                self.hand_dof_upper_limits[self.control_joint_dof_indices]
            )
        else:

            self.cur_targets[:, self.control_joint_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[self.control_joint_dof_indices],
                self.hand_dof_upper_limits[self.control_joint_dof_indices]
            )

            self.cur_targets[:, self.control_joint_dof_indices] = \
                self.act_moving_average * self.cur_targets[:, self.control_joint_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets[:, self.control_joint_dof_indices]

            self.cur_targets[:, self.control_joint_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.control_joint_dof_indices],
                self.hand_dof_lower_limits[self.control_joint_dof_indices],
                self.hand_dof_upper_limits[self.control_joint_dof_indices]
            )

        self.prev_targets[:, self.control_joint_dof_indices] = self.cur_targets[:, self.control_joint_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


    def update_goal_keypoints(self):

        # update the current keypoint positions
        for i in range(self.n_keypoints):
            self.goal_kp_positions[i] = self.goal_base_pos + quat_axis(self.goal_base_orn, axis=i) * self.kp_dist

        # visualise keypoints
        if self.viewer and self.debug_viz:

            for i in range(self.num_envs):
                for j in range(self.n_keypoints):
                    pose = gymapi.Transform()

                    pose.p = gymapi.Vec3(
                        self.goal_kp_positions[j][i,0],
                        self.goal_kp_positions[j][i,1],
                        self.goal_kp_positions[j][i,2]
                    )

                    pose.r = gymapi.Quat(0, 0, 0, 1)

                    gymutil.draw_lines(
                        self.goal_kp_geoms[j],
                        self.gym,
                        self.viewer,
                        self.envs[i],
                        pose
                    )

    def compute_observations(self):

        # get which tips are in contact
        self.tip_contacts, self.n_tip_contacts = self.get_tip_contacts()

        # get tcp positions
        tcp_states = self.rigid_body_tensor[:, self.tcp_body_idxs, :]
        self.tcp_pos = tcp_states[..., 0:3].reshape(self.num_envs, 9)
        # self.tcp_orn = tcp_states[..., 3:7]
        # self.tcp_linvel = tcp_states[..., 7:10]
        # self.tcp_angvel = tcp_states[..., 10:13]

        # get hand joint pos and vel
        self.hand_joint_pos = self.dof_pos[:, :].squeeze()
        self.hand_joint_vel = self.dof_vel[:, :].squeeze()

        # get object pose / vel
        self.obj_base_pos = self.root_state_tensor[self.obj_indices, 0:3]
        self.obj_base_orn = self.root_state_tensor[self.obj_indices, 3:7]
        self.obj_base_linvel = self.root_state_tensor[self.obj_indices, 7:10]
        self.obj_base_angvel = self.root_state_tensor[self.obj_indices, 10:13]

        # get goal pose
        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.root_state_tensor[self.goal_indices, 3:7]

        # get keypoint positions
        self.update_obj_keypoints()
        self.update_goal_keypoints()

        # obs_buf shape=(num_envs, num_obs)
        self.obs_buf[:, :9] = unscale(
            self.hand_joint_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits
        )
        self.obs_buf[:, 9:18] = self.hand_joint_vel
        self.obs_buf[:, 18:21] = self.obj_base_pos
        self.obs_buf[:, 21:25] = self.obj_base_orn
        self.obs_buf[:, 25:28] = self.obj_base_linvel
        self.obs_buf[:, 28:31] = self.obj_base_angvel
        self.obs_buf[:, 31:40] = self.actions
        self.obs_buf[:, 40:43] = self.tip_contacts
        self.obs_buf[:, 43:46] = self.obj_kp_positions[0]
        self.obs_buf[:, 46:49] = self.obj_kp_positions[1]
        self.obs_buf[:, 49:52] = self.obj_kp_positions[2]
        self.obs_buf[:, 52:61] = self.tcp_pos
        self.obs_buf[:, 61:64] = self.goal_base_pos
        self.obs_buf[:, 64:68] = self.goal_base_orn
        self.obs_buf[:, 68:71] = self.goal_kp_positions[0] - self.goal_displacement_tensor
        self.obs_buf[:, 71:74] = self.goal_kp_positions[1] - self.goal_displacement_tensor
        self.obs_buf[:, 74:77] = self.goal_kp_positions[2] - self.goal_displacement_tensor
        self.obs_buf[:, 77:81] = quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn))

        return self.obs_buf

    def compute_reward(self):
        """
        Reward computed after observation so vars set in compute_obs can
        be used here
        """

        # retrieve environment observations from buffer
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:]
        ) = compute_manip_reward(
            self.obj_base_pos,
            self.obj_base_orn,
            self.goal_base_pos - self.goal_displacement_tensor,
            self.goal_base_orn,
            self.actions,
            self.max_episode_length,
            self.fall_reset_dist,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_manip_reward(
        obj_base_pos: Tensor,
        obj_base_orn: Tensor,
        targ_base_pos: Tensor,
        targ_base_orn: Tensor,
        actions: Tensor,
        max_episode_length: float,
        fall_reset_dist: float,
        dist_reward_scale: float,
        rot_reward_scale: float,
        rot_eps: float,
        action_penalty_scale: float,
        success_tolerance: float,
        reach_goal_bonus: float,
        fall_penalty: float,
        max_consecutive_successes: float,
        av_factor: float,
        rew_buf: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        reset_goal_buf: Tensor,
        successes: Tensor,
        consecutive_successes: Tensor
    ): # -> Tuple[Tensor, Tensor]

    # Distance from the hand to the object
    goal_dist = torch.norm(obj_base_pos - targ_base_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(targ_base_orn))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # calc dist and orn rew
    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(torch.abs(rot_dist) <= success_tolerance, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(goal_dist >= fall_reset_dist, reward + fall_penalty, reward)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length, reward + 0.5 * fall_penalty, reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(goal_dist >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)

        # resets when max consecutive successes reached
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    # find average consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
