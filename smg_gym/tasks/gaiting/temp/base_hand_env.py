from typing import Deque, Dict, Tuple
from collections import deque
import numpy as np
import os
import torch

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import unscale
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import tensor_clamp
from isaacgym.torch_utils import scale
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import quat_conjugate
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_from_angle_axis
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask

from pybullet_object_models import primitive_objects as object_set

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.utils.torch_jit_utils import lgsk_kernel
from smg_gym.utils.draw_utils import get_sphere_geom
from smg_gym.assets import add_assets_path


class BaseShadowModularGrasper(VecTask):

    _state_history_len = 2
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):

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
        self.fall_reset_dist = self.cfg["env"]["fallResetDist"]
        self.require_contact = cfg["env"]["requireContact"]
        self.pos_reward_scale = cfg["env"]["posRewardScale"]
        self.orn_reward_scale = cfg["env"]["ornRewardScale"]
        self.vel_reward_scale = cfg["env"]["velRewardScale"]
        self.contact_reward_scale = cfg["env"]["contactRewardScale"]
        self.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
        self.fall_penalty_scale = cfg["env"]["fallPenaltyScale"]

        # randomisation params
        self.randomize = self.cfg["task"]["randomize"]
        self.rand_hand_joints = self.cfg["task"]["randHandJoints"]
        self.rand_obj_init_orn = self.cfg["task"]["randObjInitOrn"]
        self.rand_pivot_pos = self.cfg["task"]["randPivotPos"]
        self.rand_pivot_axel = self.cfg["task"]["randPivotAxel"]

        super().__init__(
            config=self.cfg,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless
        )

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

        self.angvel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            self._object_state_history.append(self.root_state_tensor[self.obj_indices])
            curr_history_length += 1

        # refresh all tensors
        self.refresh_tensors()

        # init goal pose
        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.root_state_tensor[self.goal_indices, 3:7]

    def create_sim(self):

        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0

        self.gym.add_ground(self.sim, plane_params)

    def _setup_hand(self):

        asset_root = add_assets_path('robot_assets/smg_minitip')
        asset_file = "smg_tactip.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.collapse_fixed_joints = False
        asset_options.armature = 0.00001
        asset_options.thickness = 0.0
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        # asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_VEL)
        # asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        asset_options.convex_decomposition_from_submeshes = False
        asset_options.flip_visual_attachments = False

        asset_options.vhacd_enabled = False
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 100000
        # asset_options.vhacd_params.concavity = 0.0025
        # asset_options.vhacd_params.alpha = 0.04
        # asset_options.vhacd_params.beta = 1.0
        # asset_options.vhacd_params.convex_hull_downsampling = 4
        # asset_options.vhacd_params.max_num_vertices_per_ch = 256

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        self.hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.control_joint_names = [
            "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
            "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
            "SMG_F3J1", "SMG_F3J2", "SMG_F3J3"
        ]
        self.control_joint_dof_indices = [self.gym.find_asset_dof_index(
            self.hand_asset, name) for name in self.control_joint_names]
        self.control_joint_dof_indices = to_torch(self.control_joint_dof_indices, dtype=torch.long, device=self.device)

        # get counts from hand asset
        self.n_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.n_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        self.n_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)

        # set initial joint state tensor (get updated on reset and step)
        self.prev_targets = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.n_hand_dofs), dtype=torch.float, device=self.device)

        # used to randomise the initial pose of the hand
        if self.randomize and self.rand_hand_joints:
            self.init_joint_mins = to_torch(np.array([
                -20.0*(np.pi/180),
                10*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

            self.init_joint_maxs = to_torch(np.array([
                20.0*(np.pi/180),
                10*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

        else:
            self.init_joint_mins = to_torch(np.array([
                0.0*(np.pi/180),
                10*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

            self.init_joint_maxs = to_torch(np.array([
                0.0*(np.pi/180),
                10*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

        # get hand limits
        hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.n_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])

        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)

    def _setup_obj(self):

        asset_root = object_set.getDataPath()
        asset_file = os.path.join(self.obj_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.override_com = False
        asset_options.override_inertia = False
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_linear_velocity = 10.0
        asset_options.max_angular_velocity = 5.0
        asset_options.thickness = 0.0
        asset_options.convex_decomposition_from_submeshes = True
        asset_options.flip_visual_attachments = False
        asset_options.vhacd_enabled = False
        self.obj_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set object properties
        # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/collision_objects.py#L96
        obj_props = self.gym.get_asset_rigid_shape_properties(self.obj_asset)
        for p in obj_props:
            p.friction = 2.0
            p.torsion_friction = 0.001
            p.rolling_friction = 0.0
            p.restitution = 0.0
            p.thickness = 0.0
        self.gym.set_asset_rigid_shape_properties(self.obj_asset, obj_props)

        # set initial state for the object
        self.default_obj_pos = (0.0, 0.0, 0.26)
        self.default_obj_orn = (0.0, 0.0, 0.0, 1.0)
        self.default_obj_linvel = (0.0, 0.0, 0.0)
        self.default_obj_angvel = (0.0, 0.0, 0.1)
        self.obj_displacement_tensor = to_torch(self.default_obj_pos, dtype=torch.float, device=self.device)

    def _setup_goal(self):
        asset_root = object_set.getDataPath()
        asset_file = os.path.join(self.obj_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        self.goal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set initial state of goal
        self.default_goal_pos = (-0.2, -0.06, 0.4)
        self.default_goal_orn = (0.0, 0.0, 0.0, 1.0)
        self.goal_displacement_tensor = to_torch(self.default_goal_pos, dtype=torch.float, device=self.device)

    def _setup_pivot_point(self):

        # fixed vars
        self.pivot_line_scale = 0.2
        self.pivot_axel_p1 = to_torch(self.default_goal_pos, dtype=torch.float,
                                      device=self.device).repeat((self.num_envs, 1))

        # vars that change on reset (with randomisation)
        self.pivot_point_pos_offset = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_point_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        self.pivot_axel_worldframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_workframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_objframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.current_obj_pivot_axel_worldframe = torch.zeros(size=(self.num_envs, 3), device=self.device)

    def _setup_keypoints(self):

        self.kp_dist = 0.05
        self.n_keypoints = 6

        self.obj_kp_positions = torch.zeros(size=(self.num_envs, self.n_keypoints, 3), device=self.device)
        self.goal_kp_positions = torch.zeros(size=(self.num_envs, self.n_keypoints, 3), device=self.device)

        self.kp_basis_vecs = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], device=self.device)

        self.kp_geoms = []
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 0, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 1, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 0, 1)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 1, 0)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(0, 1, 1)))
        self.kp_geoms.append(get_sphere_geom(rad=0.005, color=(1, 0, 1)))

    def _get_contact_idxs(self, env, obj_actor_handle, hand_actor_handle):

        obj_body_name = self.gym.get_actor_rigid_body_names(env, obj_actor_handle)
        obj_body_idx = self.gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_name[0], gymapi.DOMAIN_ENV)

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tip_body_names = [name for name in hand_body_names if 'tactip_tip' in name]
        tip_body_idxs = [self.gym.find_actor_rigid_body_index(
            env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names]

        return obj_body_idx, tip_body_idxs

    def _get_sensor_fingertip_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tcp_body_names = [name for name in hand_body_names if 'tcp' in name]
        tcp_body_idxs = [self.gym.find_actor_rigid_body_index(
            env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tcp_body_names]

        return tcp_body_idxs

    def _get_pivot_point_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        body_names = [name for name in hand_body_names if 'pivot_point' in name]
        body_idxs = [self.gym.find_actor_rigid_body_index(
            env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in body_names]

        return body_idxs

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create assets and variables
        self._setup_hand()
        self._setup_obj()
        self._setup_keypoints()
        self._setup_goal()
        self._setup_pivot_point()

        # collect useful indeces and handles
        self.envs = []
        self.hand_actor_handles = []
        self.hand_indices = []
        self.obj_actor_handles = []
        self.obj_indices = []
        self.goal_actor_handles = []
        self.goal_indices = []

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

            # TODO: Create a goal actor due to a bug that causes random behaviour on reset of multiple envs.
            # where last object in buffer appear to initialise with random velocities,
            # Having a fixed goal object absorbs this.
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

        # convert indices to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(self.goal_indices, dtype=torch.long, device=self.device)

        # get indices useful for contacts
        self.n_tips = 3
        self.obj_body_idx, self.tip_body_idxs = self._get_contact_idxs(env_ptr, obj_actor_handle, hand_actor_handle)
        self.fingertip_body_idxs = self._get_sensor_fingertip_idxs(env_ptr, hand_actor_handle)

        # get indices of pivot points
        self.pivot_point_body_idxs = self._get_pivot_point_idxs(env_ptr, hand_actor_handle)

    def _create_hand_actor(self, env_ptr, idx):

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.025)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        self.gym.begin_aggregate(env_ptr, self.n_hand_bodies, self.n_hand_shapes, False)

        handle = self.gym.create_actor(env_ptr, self.hand_asset, pose, "hand_actor_{}".format(idx), -1, -1)

        # Configure DOF properties
        props = self.gym.get_actor_dof_properties(env_ptr, handle)
        props["driveMode"] = [gymapi.DOF_MODE_POS]*self.n_hand_dofs
        props["stiffness"] = [5000.0]*self.n_hand_dofs
        props["damping"] = [100.0]*self.n_hand_dofs
        self.gym.set_actor_dof_properties(env_ptr, handle, props)

        self.gym.end_aggregate(env_ptr)

        return handle

    def _create_obj_actor(self, env_ptr, idx):

        init_obj_pose = gymapi.Transform()
        init_obj_pose.p = gymapi.Vec3(*self.default_obj_pos)
        init_obj_pose.r = gymapi.Quat(*self.default_obj_orn)

        handle = self.gym.create_actor(
            env_ptr,
            self.obj_asset,
            init_obj_pose,
            "obj_actor_{}".format(idx),
            -1,
            -1
        )

        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        for p in obj_props:
            p.mass = 0.25
            p.inertia.x = gymapi.Vec3(0.001, 0.0, 0.0)
            p.inertia.y = gymapi.Vec3(0.0, 0.001, 0.0)
            p.inertia.z = gymapi.Vec3(0.0, 0.0, 0.001)
        self.gym.set_actor_rigid_body_properties(env_ptr, handle, obj_props)

        return handle

    def _create_goal_actor(self, env, idx):
        init_goal_pose = gymapi.Transform()
        init_goal_pose.p = gymapi.Vec3(*self.default_goal_pos)
        init_goal_pose.r = gymapi.Quat(*self.default_goal_orn)
        handle = self.gym.create_actor(env, self.goal_asset, init_goal_pose, "goal_actor_{}".format(idx), 0, 0)
        return handle

    def get_tip_contacts(self):

        # get envs where obj is contacted
        obj_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.obj_body_idx, :], dim=1) > 0,
            torch.ones(size=(self.num_envs, ), device=self.device),
            torch.zeros(size=(self.num_envs,), device=self.device)
        )

        # reshape to (n_envs, n_tips)
        obj_contacts = obj_contacts.repeat(self.n_tips, 1).T

        # get envs where tips are contacted
        tip_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.tip_body_idxs, :], dim=2) > 0,
            torch.ones(size=(self.num_envs,  self.n_tips), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device)
        )

        # get envs where object and tips are contated
        tip_contacts = torch.where(
            obj_contacts > 0,
            tip_contacts,
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device)
        )
        n_tip_contacts = torch.sum(tip_contacts, dim=1)

        return tip_contacts, n_tip_contacts

    def update_keypoints(self):

        # update the current keypoint positions
        for i in range(self.n_keypoints):
            self.obj_kp_positions[:, i, :] = self.obj_base_pos + \
                quat_rotate(self.obj_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)
            self.goal_kp_positions[:, i, :] = self.goal_base_pos + \
                quat_rotate(self.goal_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)

    def compute_observations(self):

        # get which tips are in contact
        self.tip_contacts, self.n_tip_contacts = self.get_tip_contacts()

        # get fingertip positions
        fingertip_states = self.rigid_body_tensor[:, self.fingertip_body_idxs, :]
        self.fingertip_pos = fingertip_states[..., 0:3].reshape(self.num_envs, 9)
        self.fingertip_orn = fingertip_states[..., 3:7]
        self.fingertip_linvel = fingertip_states[..., 7:10]
        self.fingertip_angvel = fingertip_states[..., 10:13]

        # get hand joint pos and vel
        self.hand_joint_pos = self.dof_pos[:, :].squeeze()
        self.hand_joint_vel = self.dof_vel[:, :].squeeze()

        # get object pose / vel
        self.obj_base_pos = self.root_state_tensor[self.obj_indices, 0:3]
        self.obj_base_orn = self.root_state_tensor[self.obj_indices, 3:7]
        self.obj_base_linvel = self.root_state_tensor[self.obj_indices, 7:10]
        self.obj_base_angvel = self.root_state_tensor[self.obj_indices, 10:13]

        # append histories
        self._object_state_history.appendleft(self.root_state_tensor[self.obj_indices])

        # get keypoint positions
        self.update_keypoints()

        # compute pivot axel base on current object orn
        self.current_obj_pivot_axel_worldframe = quat_rotate(self.obj_base_orn, self.pivot_axel_objframe)

        # obs_buf shape=(num_envs, num_obs)
        self.obs_buf[:, :9] = unscale(self.hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        self.obs_buf[:, 9:18] = self.hand_joint_vel
        self.obs_buf[:, 18:21] = self.obj_base_pos
        self.obs_buf[:, 21:25] = self.obj_base_orn
        self.obs_buf[:, 25:28] = self.obj_base_linvel
        self.obs_buf[:, 28:31] = self.obj_base_angvel
        self.obs_buf[:, 31:40] = self.actions
        self.obs_buf[:, 40:43] = self.tip_contacts
        self.obs_buf[:, 43:52] = self.fingertip_pos
        self.obs_buf[:, 52:55] = self.goal_base_pos
        self.obs_buf[:, 55:59] = self.goal_base_orn
        self.obs_buf[:, 59:63] = quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn))
        self.obs_buf[:, 63:66] = self.pivot_axel_workframe
        self.obs_buf[:, 66:69] = self.pivot_point_pos_offset
        self.obs_buf[:, 69:87] = (self.obj_kp_positions
                                  - self.obj_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)
        self.obs_buf[:, 87:105] = (self.goal_kp_positions
                                   - self.goal_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)
        return self.obs_buf

    def reset_hand(self, env_ids_for_reset):

        num_envs_to_reset = len(env_ids_for_reset)

        # reset hand
        hand_velocities = torch.zeros((num_envs_to_reset, self.n_hand_dofs), device=self.device)

        # add randomisation to the joint poses
        rand_floats = torch_rand_float(-1.0, 1.0, (num_envs_to_reset, self.n_hand_dofs), device=self.device)
        rand_init_pos = self.init_joint_mins + \
            (self.init_joint_maxs - self.init_joint_mins) * rand_floats[:, :self.n_hand_dofs]

        self.dof_pos[env_ids_for_reset, :] = rand_init_pos[:]
        self.dof_vel[env_ids_for_reset, :] = hand_velocities[:]

        self.prev_targets[env_ids_for_reset, :self.n_hand_dofs] = rand_init_pos
        self.cur_targets[env_ids_for_reset, :self.n_hand_dofs] = rand_init_pos

        hand_ids_int32 = self.hand_indices[env_ids_for_reset].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_ids_int32),
            num_envs_to_reset
        )

    def reset_object(self, env_ids_for_reset):

        num_envs_to_reset = len(env_ids_for_reset)

        # set obj pos and vel to default
        object_pos = to_torch(self.default_obj_pos, dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))
        object_linvel = to_torch(self.default_obj_linvel, dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))
        object_angvel = to_torch(self.default_obj_angvel, dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))

        # randomise object rotation
        if self.randomize and self.rand_obj_init_orn:
            rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids_for_reset), 2), device=self.device)
            object_orn = randomize_rotation(
                rand_floats[:, 0],
                rand_floats[:, 1],
                self.x_unit_tensor[env_ids_for_reset],
                self.y_unit_tensor[env_ids_for_reset]
            )
        else:
            object_orn = to_torch(self.default_obj_orn, dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))

        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 0:3] = object_pos
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 3:7] = object_orn
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 7:10] = object_linvel
        self.root_state_tensor[self.obj_indices[env_ids_for_reset], 10:13] = object_angvel

        reset_obj_indices = self.obj_indices[env_ids_for_reset].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(reset_obj_indices),
            len(reset_obj_indices)
        )

    def reset_target_axis(self, env_ids_for_reset):
        """
        Reset target axis of rotation
        """
        pass

    def reset_target_pose(self, env_ids_for_reset):

        # rotate goal pose
        rotate_interval = torch.ones(size=(self.num_envs, ), device=self.device) * 45 * np.pi / 180
        rotate_quat = quat_from_angle_axis(rotate_interval, self.pivot_axel_worldframe)

        # get goal pose / vel
        self.goal_base_orn = quat_mul(self.goal_base_orn, rotate_quat)

    def reset_idx(self, env_ids_for_reset):

        self.reset_hand(env_ids_for_reset)
        self.reset_object(env_ids_for_reset)
        self.reset_target_axis(env_ids_for_reset)
        self.reset_target_pose(env_ids_for_reset)

        # reset buffers
        self.progress_buf[env_ids_for_reset] = 0
        self.reset_buf[env_ids_for_reset] = 0
        self.angvel_buf[env_ids_for_reset] = 0

    def pre_physics_step(self, actions):

        env_ids_for_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids_for_reset) > 0:
            self.reset_idx(env_ids_for_reset)

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

        if self.viewer and self.debug_viz:
            self.visualise_features()

    def refresh_tensors(self):
        # refresh all state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def visualise_features(self):

        self.gym.clear_lines(self.viewer)

        for i in range(self.num_envs):

            for j in range(self.n_keypoints):
                pose = gymapi.Transform()

                # visualise object keypoints
                pose.p = gymapi.Vec3(
                    self.obj_kp_positions[i, j, 0],
                    self.obj_kp_positions[i, j, 1],
                    self.obj_kp_positions[i, j, 2]
                )

                pose.r = gymapi.Quat(0, 0, 0, 1)

                gymutil.draw_lines(
                    self.kp_geoms[j],
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    pose
                )

                # visualise goal keypoints
                pose.p = gymapi.Vec3(
                    self.goal_kp_positions[i, j, 0],
                    self.goal_kp_positions[i, j, 1],
                    self.goal_kp_positions[i, j, 2]
                )

                gymutil.draw_lines(
                    self.kp_geoms[j],
                    self.gym,
                    self.viewer,
                    self.envs[i],
                    pose
                )

            # visualise pivot axel
            pivot_p1 = gymapi.Vec3(
                self.pivot_axel_p1[i, 0],
                self.pivot_axel_p1[i, 1],
                self.pivot_axel_p1[i, 2]
            )

            pivot_axel_p2_worldframe = self.pivot_axel_p1 + self.pivot_axel_worldframe * self.pivot_line_scale
            pivot_p2 = gymapi.Vec3(
                pivot_axel_p2_worldframe[i, 0],
                pivot_axel_p2_worldframe[i, 1],
                pivot_axel_p2_worldframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2,
                gymapi.Vec3(1.0, 1.0, 0.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )

            # visualise object frame pivot_axel
            pivot_axel_p2_objframe = self.pivot_axel_p1 + self.current_obj_pivot_axel_worldframe * self.pivot_line_scale

            pivot_p2_objframe = gymapi.Vec3(
                pivot_axel_p2_objframe[i, 0],
                pivot_axel_p2_objframe[i, 1],
                pivot_axel_p2_objframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2_objframe,
                gymapi.Vec3(0.0, 1.0, 1.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )

    def compute_reward(self):
        """
        Reward computed after observation so vars set in compute_obs can
        be used here
        """
        self.rew_buf[:] = 0.
        self.reset_buf[:] = 0.

        # retrieve environment observations from buffer
        self.rew_buf[:], self.reset_buf[:], log_dict = compute_manip_reward_keypoints(
            self.obj_kp_positions - self.obj_displacement_tensor,
            self.goal_kp_positions - self.goal_displacement_tensor,
            self.actions,
            self.max_episode_length,
            self.fall_reset_dist,
            self.n_tip_contacts,
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
        )

        self.extras.update({"metrics/"+k: v.mean() for k, v in log_dict.items()})


@torch.jit.script
def compute_manip_reward_keypoints(
                obj_kps: torch.Tensor,
                goal_kps: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance from the pivot point to the object base
    kp_deltas = torch.norm(obj_kps - goal_kps, p=2, dim=-1)
    min_kp_dist, _ = kp_deltas.min(dim=-1)

    # bound and scale rewards such that they are in similar ranges
    kp_dist_rew = lgsk_kernel(kp_deltas, scale=50., eps=2.).mean(dim=-1) * 40.0

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = kp_dist_rew

    # zero reward when less than 2 tips in contact
    # total_reward = torch.where(n_tip_contacts < n_contact_lim, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    # total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(min_kp_dist >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'num_tip_contacts': n_tip_contacts,
        'total_reward': total_reward,
    }

    return total_reward, resets, info
