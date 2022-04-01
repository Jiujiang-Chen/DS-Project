import numpy as np
import os
import torch

from isaacgym.torch_utils import unscale
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import tensor_clamp
from isaacgym.torch_utils import scale
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from pybullet_object_models import primitive_objects as object_set

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.utils.draw_utils import get_sphere_geom
from smg_gym.assets import add_assets_path


class BaseShadowModularGrasper(VecTask):
    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):

        # setup params
        self.cfg = cfg
        self.debug_viz = cfg["env"]["enableDebugVis"]
        self.env_spacing = cfg["env"]["envSpacing"]
        self.obj_name = cfg["env"]["objName"]

        # action params
        self.use_relative_control = cfg["env"]["useRelativeControl"]
        self.dof_speed_scale = cfg["env"]["dofSpeedScale"]
        self.act_moving_average = cfg["env"]["actionsMovingAverage"]

        super().__init__(config=cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

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

        # get indices useful for contacts
        self._setup_fingertip_tracking()
        self._setup_contact_tracking()

        # refresh all tensors
        self.refresh_tensors()

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.action_buf = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        plane_params.static_friction = 0.0
        plane_params.dynamic_friction = 0.0
        plane_params.restitution = 0

        self.gym.add_ground(self.sim, plane_params)

    def _setup_hand(self):

        asset_root = add_assets_path("robot_assets/smg_minitip")
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

        self.hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.control_joint_names = [
            "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
            "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
            "SMG_F3J1", "SMG_F3J2", "SMG_F3J3",
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

        # get hand limits
        hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []

        for i in range(self.n_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(hand_dof_props["upper"][i])

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
        self.default_obj_pos = (0.0, 0.0, 0.265)
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

    def _setup_keypoints(self):

        self.kp_dist = 0.06
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

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # create assets and variables
        self._setup_hand()
        self._setup_obj()
        self._setup_keypoints()
        self._setup_goal()

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

        # convert indices to tensors
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.obj_indices = to_torch(self.obj_indices, dtype=torch.long, device=self.device)
        self.goal_indices = to_torch(self.goal_indices, dtype=torch.long, device=self.device)

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

    def _setup_contact_tracking(self):
        self.n_tips = 3
        self.obj_body_idx, self.tip_body_idxs = self._get_contact_idxs(
            self.envs[0], self.obj_actor_handles[0], self.hand_actor_handles[0]
        )

    def _get_contact_idxs(self, env, obj_actor_handle, hand_actor_handle):

        obj_body_name = self.gym.get_actor_rigid_body_names(env, obj_actor_handle)
        obj_body_idx = self.gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_name[0], gymapi.DOMAIN_ENV)

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tip_body_names = [name for name in hand_body_names if "tactip_tip" in name]
        tip_body_idxs = [
            self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names
        ]

        return obj_body_idx, tip_body_idxs

    def _setup_fingertip_tracking(self):
        self.fingertip_tcp_body_idxs = self._get_fingertip_tcp_idxs(self.envs[0], self.hand_actor_handles[0])

    def _get_fingertip_tcp_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        tcp_body_names = [name for name in hand_body_names if "tcp" in name]
        fingertip_tcp_body_idxs = [
            self.gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tcp_body_names
        ]

        return fingertip_tcp_body_idxs

    def get_rich_fingertip_contacts(self):
        for i in range(self.num_envs):
            contacts = self.gym.get_env_rigid_contacts(self.envs[i])
            # print(contacts)

    def get_fingertip_contacts(self):

        # get envs where obj is contacted
        bool_obj_contacts = torch.where(
            torch.count_nonzero(self.contact_force_tensor[:, self.obj_body_idx, :], dim=1) > 0,
            torch.ones(size=(self.num_envs,), device=self.device),
            torch.zeros(size=(self.num_envs,), device=self.device),
        )

        # get envs where tips are contacted
        net_tip_contact_forces = self.contact_force_tensor[:, self.tip_body_idxs, :]
        bool_tip_contacts = torch.where(
            torch.count_nonzero(net_tip_contact_forces, dim=2) > 0,
            torch.ones(size=(self.num_envs, self.n_tips), device=self.device),
            torch.zeros(size=(self.num_envs, self.n_tips), device=self.device),
        )

        # repeat for n_tips shape=(n_envs, n_tips)
        onehot_obj_contacts = bool_obj_contacts.unsqueeze(1).repeat(1, self.n_tips)

        # get envs where object and tips are contated
        tip_object_contacts = torch.where(
            onehot_obj_contacts > 0, bool_tip_contacts, torch.zeros(size=(self.num_envs, self.n_tips), device=self.device)
        )
        n_tip_contacts = torch.sum(bool_tip_contacts, dim=1)

        return net_tip_contact_forces, tip_object_contacts, n_tip_contacts

    def pre_physics_step(self, actions):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        self.apply_resets()
        self.apply_actions(actions)

    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        self.progress_buf += 1

        self.refresh_tensors()
        self.compute_observations()
        self.fill_observation_buffer()
        self.fill_states_buffer()
        self.compute_reward_and_termination()

        if self.viewer and self.debug_viz:
            self.visualise_features()

    def refresh_tensors(self):
        """
        Refresh all state tensors.
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def standard_fill_buffer(self, buf, buf_cfg):
        """
        Fill observation buffer with observations shared across different tasks.

        Change offset outside of conditional to keep the same indices in the buffer, allows for disabling
        observations after learning.
        """

        # joint position
        start_offset = 0
        end_offset = start_offset + 9
        if buf_cfg["jointPos"]:
            buf[:, start_offset:end_offset] = unscale(
                self.hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

        # joint velocity
        start_offset = end_offset
        end_offset = start_offset + 9
        if buf_cfg["jointVel"]:
            buf[:, start_offset:end_offset] = self.hand_joint_vel

        # fingertip positions
        start_offset = end_offset
        end_offset = start_offset + 9
        if buf_cfg["fingertipPos"]:
            buf[:, start_offset:end_offset] = self.fingertip_pos

        # fingertip orn
        start_offset = end_offset
        end_offset = start_offset + 12
        if buf_cfg["fingertipOrn"]:
            buf[:, start_offset:end_offset] = self.fingertip_orn.reshape(self.num_envs, self.n_tips*4)

        # latest actions
        start_offset = end_offset
        end_offset = start_offset + 9
        if buf_cfg["lastAction"]:
            buf[:, start_offset:end_offset] = self.actions

        # boolean tips in contacts
        start_offset = end_offset
        end_offset = start_offset + 3
        if buf_cfg["boolTipContacts"]:
            buf[:, start_offset:end_offset] = self.tip_object_contacts

        # tip contact forces
        start_offset = end_offset
        end_offset = start_offset + 9
        if buf_cfg["tipContactForces"]:
            buf[:, start_offset:end_offset] = self.net_tip_contact_forces.reshape(self.num_envs, self.n_tips*3)

        # object position
        start_offset = end_offset
        end_offset = start_offset + 3
        if buf_cfg["objectPos"]:
            buf[:, start_offset:end_offset] = self.obj_base_pos

        # object orientation
        start_offset = end_offset
        end_offset = start_offset + 4
        if buf_cfg["objectOrn"]:
            buf[:, start_offset:end_offset] = self.obj_base_orn

        # object keypoints
        start_offset = end_offset
        end_offset = start_offset + self.n_keypoints*3
        if buf_cfg["objectKPs"]:
            buf[:, start_offset:end_offset] = (self.obj_kp_positions
                                               - self.obj_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)

        # object linear velocity
        start_offset = end_offset
        end_offset = start_offset + 3
        if buf_cfg["objectLinVel"]:
            buf[:, start_offset:end_offset] = self.obj_base_linvel

        # object angular velocity
        start_offset = end_offset
        end_offset = start_offset + 3
        if buf_cfg["objectAngVel"]:
            buf[:, start_offset:end_offset] = self.obj_base_angvel

        # goal position
        start_offset = end_offset
        end_offset = start_offset + 3
        if buf_cfg["GoalPos"]:
            buf[:, start_offset:end_offset] = self.goal_base_pos

        # goal orientation
        start_offset = end_offset
        end_offset = start_offset + 4
        if buf_cfg["GoalOrn"]:
            buf[:, start_offset:end_offset] = self.goal_base_orn

        # goal keypoints
        start_offset = end_offset
        end_offset = start_offset + self.n_keypoints*3
        if buf_cfg["GoalKPs"]:
            buf[:, start_offset:end_offset] = (self.goal_kp_positions
                                               - self.goal_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)

        # active quat between goal and object
        start_offset = end_offset
        end_offset = start_offset + 4
        if buf_cfg["activeQuat"]:
            self.obs_buf[:, start_offset:end_offset] = quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn))

        return start_offset, end_offset

    def compute_reward_and_termination(self):
        """
        Calculate the reward.
        """
        pass

    def reset_hand(self, env_ids_for_reset):
        """
        Reset joint positions on hand, randomisation limits handled in init_joint_mins/maxs.
        """
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
        """
        Reset the pose of the object.
        """

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

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose of the object.
        """
        pass

    def apply_resets(self):
        """
        Logic for applying resets
        """

        env_ids_for_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids_for_reset = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        actor_root_state_reset_indices = []

        # nothing to reset
        if len(env_ids_for_reset) == 0 and len(goal_env_ids_for_reset) == 0:
            return None

        # reset goals
        if len(goal_env_ids_for_reset) > 0:
            self.reset_target_pose(goal_env_ids_for_reset)
            actor_root_state_reset_indices.append(self.goal_indices[goal_env_ids_for_reset])

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_hand(env_ids_for_reset)
            self.reset_target_pose(env_ids_for_reset)
            self.reset_object(env_ids_for_reset)
            actor_root_state_reset_indices.append(self.obj_indices[env_ids_for_reset])
            actor_root_state_reset_indices.append(self.goal_indices[env_ids_for_reset])

        # set the root state tensor to reset object and goal pose
        # has to be done together for some reason...
        reset_indices = torch.unique(
            torch.cat(actor_root_state_reset_indices)
        ).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(reset_indices),
            len(reset_indices)
        )

        # reset buffers
        self.progress_buf[env_ids_for_reset] = 0
        self.reset_buf[env_ids_for_reset] = 0
        self.successes[env_ids_for_reset] = 0

    def apply_actions(self, actions):
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

    def compute_observations(self):

        # get which tips are in contact
        self.net_tip_contact_forces, self.tip_object_contacts, self.n_tip_contacts = self.get_fingertip_contacts()
        # self.get_rich_fingertip_contacts()

        # get tcp positions
        fingertip_states = self.rigid_body_tensor[:, self.fingertip_tcp_body_idxs, :]
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

        # get goal pose
        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.root_state_tensor[self.goal_indices, 3:7]

        # update the current keypoint positions
        for i in range(self.n_keypoints):
            self.obj_kp_positions[:, i, :] = self.obj_base_pos + \
                quat_rotate(self.obj_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)
            self.goal_kp_positions[:, i, :] = self.goal_base_pos + \
                quat_rotate(self.goal_base_orn, self.kp_basis_vecs[i].repeat(self.num_envs, 1) * self.kp_dist)

    def visualise_features(self):

        self.gym.clear_lines(self.viewer)

        for i in range(self.num_envs):

            # draw contacts
            if self.device == 'cpu':
                contact_color = gymapi.Vec3(1, 0, 0)
                self.gym.draw_env_rigid_contacts(self.viewer, self.envs[i], contact_color, 0.2, True)

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

    """
    Properties
    """

    def get_gravity(self) -> np.ndarray:
        """Returns the gravity set in the simulator.
        """
        gravity = self.sim_params.gravity
        return np.asarray([gravity.x, gravity.y, gravity.z])

    def get_sim_params(self) -> gymapi.SimParams:
        """Returns the simulator physics parameters.
        """
        return self.sim_params

    def get_state_shape(self) -> torch.Size:
        """Returns the size of the state buffer: [num. instances, num. of state]
        """
        return self.states_buf.size()

    def get_obs_shape(self) -> torch.Size:
        """Returns the size of the observation buffer: [num. instances, num. of obs]
        """
        return self.obs_buf.size()

    def get_action_shape(self) -> torch.Size:
        """Returns the size of the action buffer: [num. instances, num. of action]
        """
        return self.action_buf.size()

    def get_num_envs(self) -> int:
        """Returns number of environment instances
        """
        return self.num_envs

    def get_state_dim(self) -> int:
        """Returns imensions of state specfication.
        """
        return self.get_state_shape()[1]

    def get_obs_dim(self) -> int:
        """Returns imensions of obs specfication.
        """
        return self.get_obs_shape()[1]

    def get_action_dim(self) -> int:
        """Returns imensions of action specfication.
        """
        return self.get_action_shape()[1]
