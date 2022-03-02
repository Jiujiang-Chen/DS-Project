import numpy as np
import os
import torch
from torch import Tensor

from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate
from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import unscale
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import tensor_clamp
from isaacgym.torch_utils import scale
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from pybullet_object_models import primitive_objects as object_set

from smg_gym.rl_games_helpers.utils.torch_jit_utils import randomize_rotation
from smg_gym.rl_games_helpers.utils.torch_jit_utils import quat_axis
from smg_gym.assets import add_assets_path


class BaseShadowModularGrasper(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        # setup params
        # self.num_envs = self.cfg["env"]["numEnvs"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # action params
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # reward/termination params
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.dist_reward_scale = cfg["env"]["distRewardScale"]
        self.rot_reward_scale = cfg["env"]["rotRewardScale"]
        self.require_contact = cfg["env"]["requireContact"]
        self.contact_reward_scale = cfg["env"]["contactRewardScale"]
        self.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = cfg["env"]["successTolerance"]
        self.reach_goal_bonus = cfg["env"]["reachGoalBonus"]
        self.fall_reset_dist = self.cfg["env"]["fallResetDist"]
        self.fall_penalty = cfg["env"]["fallPenalty"]
        self.max_consecutive_successes = cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = cfg["env"]["avFactor"]
        self.rot_eps = cfg["env"]["rotEps"]

        # randomisation params
        self.randomize = self.cfg["task"]["randomize"]
        self.rand_hand_joints = self.cfg["task"]["randHandJoints"]
        self.rand_init_orn = self.cfg["task"]["randInitOrn"]

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # change viewer camera
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(2, 2, 2)
            cam_target = gymapi.Vec3(0, 0, 1)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # get useful numbers
        self.n_sim_bodies = self.gym.get_sim_rigid_body_count(self.sim)
        self.n_env_bodies = self.gym.get_sim_rigid_body_count(self.sim) // self.num_envs

        # create views of actor_root tensor
        # shape = (num_environments, num_actors * 13)
        # 13 -> position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # create views of dof tensor
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 1]

        # create views of rigid body states
        # shape = (num_environments, num_bodies * 13)
        self.rigid_body_tensor = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, self.n_env_bodies, 13)

        # create views of contact_force tensor
        # default shape = (n_envs, n_bodies * 3)
        self.contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.n_env_bodies, 3)

        # setup useful incices
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # setup goal / successes buffers
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        # refresh all tensors
        self.refresh_tensors()

    def create_sim(self):

        self.dt = self.sim_params.dt

        # set the up axis to be z-up given that assets are y-up by default
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0

        self.gym.add_ground(self.sim, plane_params)

    def _setup_hand(self):

        asset_root = add_assets_path("robot_assets/smg")
        asset_file = "smg_tactip.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.collapse_fixed_joints = False
        asset_options.armature = 0.00001
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.convex_decomposition_from_submeshes = False
        asset_options.vhacd_enabled = False
        asset_options.flip_visual_attachments = False

        hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.control_joint_names = [
            "SMG_F1J1",
            "SMG_F1J2",
            "SMG_F1J3",
            "SMG_F2J1",
            "SMG_F2J2",
            "SMG_F2J3",
            "SMG_F3J1",
            "SMG_F3J2",
            "SMG_F3J3",
        ]
        self.control_joint_dof_indices = [self.gym.find_asset_dof_index(hand_asset, name) for name in self.control_joint_names]
        self.control_joint_dof_indices = to_torch(self.control_joint_dof_indices, dtype=torch.long, device=self.device)

        # get counts from hand asset
        self.n_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.n_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.n_hand_dofs = self.gym.get_asset_dof_count(hand_asset)

        # set initial joint state tensor (get updated on reset and step)
        self.prev_targets = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)

        # used to randomise the initial pose of the hand
        if self.randomize and self.rand_hand_joints:
            self.init_joint_mins = to_torch(
                np.array(
                    [
                        -20.0 * (np.pi / 180),
                        7.5 * (np.pi / 180),
                        -10.0 * (np.pi / 180),
                    ]
                    * 3
                )
            )

            self.init_joint_maxs = to_torch(
                np.array(
                    [
                        20.0 * (np.pi / 180),
                        7.5 * (np.pi / 180),
                        -10.0 * (np.pi / 180),
                    ]
                    * 3
                )
            )

        else:
            self.init_joint_mins = to_torch(
                np.array(
                    [
                        0.0 * (np.pi / 180),
                        7.5 * (np.pi / 180),
                        -10.0 * (np.pi / 180),
                    ]
                    * 3
                )
            )

            self.init_joint_maxs = to_torch(
                np.array(
                    [
                        0.0 * (np.pi / 180),
                        7.5 * (np.pi / 180),
                        -10.0 * (np.pi / 180),
                    ]
                    * 3
                )
            )

        # get hand limits
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.n_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(hand_dof_props["upper"][i])

        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)

        return hand_asset

    def _setup_obj(self):

        asset_root = object_set.getDataPath()
        asset_file = os.path.join(self.obj_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        obj_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set initial state for the object
        self.init_obj_pose = gymapi.Transform()
        self.init_obj_pose.p = gymapi.Vec3(0.0, 0.0, 0.245)
        self.init_obj_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.init_obj_vel = gymapi.Velocity()
        self.init_obj_vel.linear = gymapi.Vec3(0.0, 0.0, 0.0)
        self.init_obj_vel.angular = gymapi.Vec3(0.0, 0.0, 0.0)

        return obj_asset

    def _setup_goal(self):

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
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device
        )

        self.init_goal_pose = gymapi.Transform()
        self.init_goal_pose.p = self.init_obj_pose.p + self.goal_displacement
        self.init_goal_pose.r = self.init_obj_pose.r

        self.init_goal_vel = gymapi.Velocity()
        self.init_goal_vel.linear = gymapi.Vec3(0.0, 0.0, 0.0)
        self.init_goal_vel.angular = gymapi.Vec3(0.0, 0.0, 0.0)

        return goal_asset

    def _setup_keypoints(self, color=(1, 0, 0)):

        self.kp_dist = 0.05
        self.n_keypoints = 3

        kp_positions = [
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
        ]

        kp_geoms = []
        kp_geoms.append(self._get_kp_geom(color=(1, 0, 0)))
        kp_geoms.append(self._get_kp_geom(color=(0, 1, 0)))
        kp_geoms.append(self._get_kp_geom(color=(0, 0, 1)))

        return kp_geoms, kp_positions

    def _get_kp_geom(self, color=(1, 0, 0)):

        sphere_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.0), r=gymapi.Quat(0, 0, 0, 1))

        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 12, 12, sphere_pose, color=color)  # rad  # n_lat  # n_lon
        return sphere_geom

    def _get_contact_idxs(self, env, obj_actor_handle, hand_actor_handle):

        obj_body_name = self.gym.get_actor_rigid_body_names(env, obj_actor_handle)
        obj_body_idx = self.gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_name[0], gymapi.DOMAIN_ENV)

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tip_body_names = [name for name in hand_body_names if "tactip_tip" in name]
        tip_body_idxs = [
            self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names
        ]

        return obj_body_idx, tip_body_idxs

    def _get_sensor_tcp_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tcp_body_names = [name for name in hand_body_names if "tcp" in name]
        tcp_body_idxs = [
            self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tcp_body_names
        ]

        return tcp_body_idxs

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create assets and variables
        self.hand_asset = self._setup_hand()
        self.obj_asset = self._setup_obj()
        self.obj_kp_geoms, self.obj_kp_positions = self._setup_keypoints()
        self.goal_kp_geoms, self.goal_kp_positions = self._setup_keypoints()
        self.goal_asset = self._setup_goal()

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
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

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
            self.init_obj_states.append(
                [
                    self.init_obj_pose.p.x,
                    self.init_obj_pose.p.y,
                    self.init_obj_pose.p.z,
                    self.init_obj_pose.r.x,
                    self.init_obj_pose.r.y,
                    self.init_obj_pose.r.z,
                    self.init_obj_pose.r.w,
                    self.init_obj_vel.linear.x,
                    self.init_obj_vel.linear.y,
                    self.init_obj_vel.linear.z,
                    self.init_obj_vel.angular.x,
                    self.init_obj_vel.angular.y,
                    self.init_obj_vel.angular.z,
                ]
            )

            self.init_goal_states.append(
                [
                    self.init_goal_pose.p.x,
                    self.init_goal_pose.p.y,
                    self.init_goal_pose.p.z,
                    self.init_goal_pose.r.x,
                    self.init_goal_pose.r.y,
                    self.init_goal_pose.r.z,
                    self.init_goal_pose.r.w,
                    self.init_goal_vel.linear.x,
                    self.init_goal_vel.linear.y,
                    self.init_goal_vel.linear.z,
                    self.init_goal_vel.angular.x,
                    self.init_goal_vel.angular.y,
                    self.init_goal_vel.angular.z,
                ]
            )

        # convert indices to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(self.goal_indices, dtype=torch.long, device=self.device)

        # get indices useful for contacts
        self.n_tips = 3
        self.obj_body_idx, self.tip_body_idxs = self._get_contact_idxs(env_ptr, obj_actor_handle, hand_actor_handle)
        self.tcp_body_idxs = self._get_sensor_tcp_idxs(env_ptr, hand_actor_handle)

        # convert states to tensors (TODO: make this more intuitive shape from start)
        self.init_obj_states = to_torch(self.init_obj_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.init_goal_states = to_torch(self.init_goal_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

    def _create_hand_actor(self, env_ptr, idx):

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.025)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        self.gym.begin_aggregate(env_ptr, self.n_hand_bodies, self.n_hand_shapes, False)

        handle = self.gym.create_actor(env_ptr, self.hand_asset, pose, "hand_actor_{}".format(idx), -1, -1)

        # Configure DOF properties
        props = self.gym.get_actor_dof_properties(env_ptr, handle)
        props["driveMode"] = [gymapi.DOF_MODE_POS] * self.n_hand_dofs
        props["stiffness"] = [5000.0] * self.n_hand_dofs
        props["damping"] = [100.0] * self.n_hand_dofs
        self.gym.set_actor_dof_properties(env_ptr, handle, props)

        self.gym.end_aggregate(env_ptr)

        return handle

    def _create_obj_actor(self, env_ptr, idx):

        handle = self.gym.create_actor(env_ptr, self.obj_asset, self.init_obj_pose, "obj_actor_{}".format(idx), -1, -1)

        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        obj_props[0].mass = 0.25
        self.gym.set_actor_rigid_body_properties(env_ptr, handle, obj_props)

        return handle

    def _create_goal_actor(self, env, idx):

        handle = self.gym.create_actor(env, self.goal_asset, self.init_goal_pose, "goal_actor_{}".format(idx), 0, 0)

        return handle

    def get_tip_contacts(self):

        # get envs where obj is contacted
        obj_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.obj_body_idx, :], dim=1) > 0,
            torch.ones(size=(self.num_envs,), device=self.device),
            torch.zeros(size=(self.num_envs,), device=self.device),
        )

        # reshape to (n_envs, n_tips)
        obj_contacts = obj_contacts.repeat(self.n_tips, 1).T

        # get envs where tips are contacted
        tip_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.tip_body_idxs, :], dim=2) > 0,
            torch.ones(size=(self.num_envs, self.n_tips), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device),
        )

        # get envs where object and tips are contated
        tip_contacts = torch.where(
            obj_contacts > 0, tip_contacts, torch.zeros(size=(self.num_envs, self.n_tips), device=self.device)
        )
        n_tip_contacts = torch.sum(tip_contacts, dim=1)

        return tip_contacts, n_tip_contacts

    def update_obj_keypoints(self):

        # update the current keypoint positions
        for i in range(self.n_keypoints):
            self.obj_kp_positions[i] = self.obj_base_pos + quat_axis(self.obj_base_orn, axis=i) * self.kp_dist

        # visualise keypoints
        if self.viewer and self.debug_viz:

            self.gym.clear_lines(self.viewer)

            for i in range(self.num_envs):
                for j in range(self.n_keypoints):
                    pose = gymapi.Transform()

                    pose.p = gymapi.Vec3(
                        self.obj_kp_positions[j][i, 0], self.obj_kp_positions[j][i, 1], self.obj_kp_positions[j][i, 2]
                    )

                    pose.r = gymapi.Quat(0, 0, 0, 1)

                    gymutil.draw_lines(self.obj_kp_geoms[j], self.gym, self.viewer, self.envs[i], pose)

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
                        self.goal_kp_positions[j][i, 0], self.goal_kp_positions[j][i, 1], self.goal_kp_positions[j][i, 2]
                    )

                    pose.r = gymapi.Quat(0, 0, 0, 1)

                    gymutil.draw_lines(self.goal_kp_geoms[j], self.gym, self.viewer, self.envs[i], pose)

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
        self.obs_buf[:, :9] = unscale(self.hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
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
            self.consecutive_successes[:],
        ) = compute_manip_reward(
            self.obj_base_pos,
            self.obj_base_orn,
            self.goal_base_pos - self.goal_displacement_tensor,
            self.goal_base_orn,
            self.actions,
            self.n_tip_contacts,
            self.max_episode_length,
            self.fall_reset_dist,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.require_contact,
            self.contact_reward_scale,
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
            self.consecutive_successes,
        )

        # self.extras['successes'] = self.successes
        self.extras["consecutive_successes"] = self.consecutive_successes.mean()

    def reset_idx(self, env_ids, goal_env_ids):

        # reset hand
        hand_velocities = torch.zeros((len(env_ids), self.n_hand_dofs), device=self.device)

        # add randomisation to the joint poses
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.n_hand_dofs), device=self.device)
        rand_init_pos = (
            self.init_joint_mins + (self.init_joint_maxs - self.init_joint_mins) * rand_floats[:, : self.n_hand_dofs]
        )

        self.dof_pos[env_ids, :] = rand_init_pos[:]
        self.dof_vel[env_ids, :] = hand_velocities[:]

        self.prev_targets[env_ids, : self.n_hand_dofs] = rand_init_pos
        self.cur_targets[env_ids, : self.n_hand_dofs] = rand_init_pos

        hand_ids_int32 = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_ids_int32), len(env_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_ids_int32), len(env_ids)
        )

        # reset object
        self.root_state_tensor[self.obj_indices[env_ids]] = self.init_obj_states[env_ids].clone()

        # randomise rotation
        if self.randomize and self.rand_init_orn:
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
            new_object_rot = randomize_rotation(
                rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            )
            self.root_state_tensor[self.obj_indices[env_ids], 3:7] = new_object_rot

        # update goal_pos
        self.reset_target_pose(env_ids, apply_reset=False)

        # set the root state tensor to reset object and goal pose
        # has to be done together for some reason...
        reset_indices = torch.unique(
            torch.cat([self.obj_indices[env_ids], self.goal_indices[env_ids], self.goal_indices[goal_env_ids]]).to(torch.int32)
        )

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Reset goal with rand orientation
        """
        pass

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
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.use_relative_control:
            targets = self.prev_targets[:, self.control_joint_dof_indices] + self.dof_speed_scale * self.dt * self.actions

            self.cur_targets[:, self.control_joint_dof_indices] = tensor_clamp(
                targets,
                self.hand_dof_lower_limits[self.control_joint_dof_indices],
                self.hand_dof_upper_limits[self.control_joint_dof_indices],
            )
        else:

            self.cur_targets[:, self.control_joint_dof_indices] = scale(
                self.actions,
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

    def post_physics_step(self):
        self.progress_buf += 1

        self.refresh_tensors()
        self.compute_observations()
        self.compute_reward()

    def refresh_tensors(self):
        # refresh all state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


@torch.jit.script
def compute_manip_reward(
    obj_base_pos: Tensor,
    obj_base_orn: Tensor,
    targ_base_pos: Tensor,
    targ_base_orn: Tensor,
    actions: Tensor,
    n_tip_contacts: Tensor,
    max_episode_length: float,
    fall_reset_dist: float,
    dist_reward_scale: float,
    rot_reward_scale: float,
    require_contact: bool,
    contact_reward_scale: float,
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
    consecutive_successes: Tensor,
):  # -> Tuple[Tensor, Tensor]

    # Distance from the hand to the object
    goal_dist = torch.norm(obj_base_pos - targ_base_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(targ_base_orn))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # calc dist and orn rew
    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # add penalty for large actions
    action_penalty = torch.sum(actions**2, dim=-1) * action_penalty_scale

    # add reward for maintaining tips in contact
    contact_rew = n_tip_contacts * contact_reward_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty + contact_rew

    # zero reward when less than 2 tips in contact
    if require_contact:
        reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), reward)

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
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
