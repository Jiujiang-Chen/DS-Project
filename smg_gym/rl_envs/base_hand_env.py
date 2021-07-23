import numpy as np
import os
import torch
from torch import Tensor

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from smg_gym.rl_games_helpers.utils.torch_jit_utils import randomize_rotation, quat_axis
from smg_gym.assets import get_assets_path, add_assets_path
from pybullet_object_models import primitive_objects as object_set

class BaseShadowModularGrasper(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.num_envs = self.cfg["env"]["numEnvs"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.fall_reset_dist = self.cfg["env"]["fallResetDist"]

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.randomize = self.cfg["task"]["randomize"]
        self.rand_hand_joints = self.cfg["task"]["randHandJoints"]
        self.rand_init_orn = self.cfg["task"]["randInitOrn"]

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # change viewer camera
        if self.viewer != None:
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

        # refresh all tensors
        self.refresh_tensors()


    def create_sim(self):

        self.dt = self.sim_params.dt

        # set the up axis to be z-up given that assets are y-up by default
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

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

        asset_root = add_assets_path('robot_assets/smg')
        asset_file =  "smg_tactip.urdf"

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
            "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
            "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
            "SMG_F3J1", "SMG_F3J2", "SMG_F3J3"
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
            self.init_joint_mins = to_torch(np.array([
                -20.0*(np.pi/180),
                7.5*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

            self.init_joint_maxs = to_torch(np.array([
                20.0*(np.pi/180),
                7.5*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

        else:
            self.init_joint_mins = to_torch(np.array([
                0.0*(np.pi/180),
                7.5*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

            self.init_joint_maxs = to_torch(np.array([
                0.0*(np.pi/180),
                7.5*(np.pi/180),
                -10.0*(np.pi/180),
            ] * 3))

        # get hand limits
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.n_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])

        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)

        return hand_asset

    def _setup_obj(self):
        self.obj_name = 'sphere'

        model_list = object_set.getModelList()
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


    def _setup_keypoints(self, color=(1,0,0)):

        self.kp_dist = 0.05
        self.n_keypoints = 3

        kp_positions = [
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_keypoints), device=self.device),
        ]

        kp_geoms = []
        kp_geoms.append(self._get_kp_geom(color=(1,0,0)))
        kp_geoms.append(self._get_kp_geom(color=(0,1,0)))
        kp_geoms.append(self._get_kp_geom(color=(0,0,1)))

        return kp_geoms, kp_positions

    def _get_kp_geom(self, color=(1,0,0)):

        sphere_pose = gymapi.Transform(
            p=gymapi.Vec3(0.0, 0.0, 0.0),
            r=gymapi.Quat(0, 0, 0, 1)
        )

        sphere_geom = gymutil.WireframeSphereGeometry(
            0.01, # rad
            12, # n_lat
            12, # n_lon
            sphere_pose,
            color=color
        )
        return sphere_geom

    def _get_contact_idxs(self, env, obj_actor_handle, hand_actor_handle):

        obj_body_name = self.gym.get_actor_rigid_body_names(env, obj_actor_handle)
        obj_body_idx = self.gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_name[0], gymapi.DOMAIN_ENV)

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tip_body_names = [name for name in hand_body_names if 'tactip_tip' in name]
        tip_body_idxs = [self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names]

        return obj_body_idx, tip_body_idxs

    def _get_sensor_tcp_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tcp_body_names = [name for name in hand_body_names if 'tcp' in name]
        tcp_body_idxs = [self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tcp_body_names]

        return tcp_body_idxs

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.hand_asset = self._setup_hand()
        self.obj_asset = self._setup_obj()
        self.obj_kp_geoms, self.obj_kp_positions = self._setup_keypoints()

        # collect useful indeces and handles
        self.envs = []
        self.hand_actor_handles = []
        self.hand_indices = []
        self.obj_actor_handles = []
        self.obj_indices = []
        self.init_obj_states = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            hand_actor_handle = self._create_hand_actor(env_ptr, i)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor_handle, gymapi.DOMAIN_SIM)

            obj_actor_handle = self._create_obj_actor(env_ptr, i)
            obj_idx = self.gym.get_actor_index(env_ptr, obj_actor_handle, gymapi.DOMAIN_SIM)

            # append handles and indeces
            self.envs.append(env_ptr)
            self.hand_actor_handles.append(hand_actor_handle)
            self.hand_indices.append(hand_idx)
            self.obj_actor_handles.append(obj_actor_handle)
            self.obj_indices.append(obj_idx)

            # append states
            self.init_obj_states.append([
                self.init_obj_pose.p.x, self.init_obj_pose.p.y, self.init_obj_pose.p.z,
                self.init_obj_pose.r.x, self.init_obj_pose.r.y, self.init_obj_pose.r.z, self.init_obj_pose.r.w,
                self.init_obj_vel.linear.x, self.init_obj_vel.linear.y, self.init_obj_vel.linear.z,
                self.init_obj_vel.angular.x, self.init_obj_vel.angular.y, self.init_obj_vel.angular.z,
            ])

        # convert handles and indeces to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.init_obj_states = to_torch(self.init_obj_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        # get indices useful for contacts
        self.n_tips = 3
        self.obj_body_idx, self.tip_body_idxs = self._get_contact_idxs(env_ptr, obj_actor_handle, hand_actor_handle)
        self.tcp_body_idxs = self._get_sensor_tcp_idxs(env_ptr, hand_actor_handle)

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

        handle = self.gym.create_actor(
            env_ptr,
            self.obj_asset,
            self.init_obj_pose,
            "obj_actor_{}".format(idx),
            -1,
            -1
        )

        obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        obj_props[0].mass = 0.25
        self.gym.set_actor_rigid_body_properties(env_ptr, handle, obj_props)

        return handle

    def get_tip_contacts(self):

        # get envs where obj is contacted
        obj_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.obj_body_idx, :], dim=1) > 0,
            torch.ones(size=(self.num_envs, ), device=self.device),
            torch.zeros(size=(self.num_envs,), device=self.device)
        )

        # reshape to (n_envs, n_tips)
        obj_contacts = obj_contacts.repeat(self.n_tips,1).T

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
                        self.obj_kp_positions[j][i,0],
                        self.obj_kp_positions[j][i,1],
                        self.obj_kp_positions[j][i,2]
                    )

                    pose.r = gymapi.Quat(0, 0, 0, 1)

                    gymutil.draw_lines(
                        self.obj_kp_geoms[j],
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

        # get keypoint positions
        self.update_obj_keypoints()

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

        return self.obs_buf

    def compute_reward(self):
        """
        Reward computed after observation so vars set in compute_obs can
        be used here
        """
        pass


    def reset(self, env_ids):

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

        obj_ids_int32 = self.obj_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_ids_int32),
            len(env_ids)
        )

        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

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
