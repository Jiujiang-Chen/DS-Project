import numpy as np
import os
import torch
from torch import Tensor

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

from smg_gym.assets import get_assets_path, add_assets_path
from pybullet_object_models import primitive_objects as object_set

class ShadowModularGrasper(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.num_envs = self.cfg["env"]["numEnvs"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]

        self.cfg["env"]["numObservations"] = 4
        self.cfg["env"]["numActions"] = 9

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

        # create views of actor_root tensor
        # shape = (num_environments, num_actors * 13)
        # 13 -> position([0:3]), rotation([3:7]), linear velocity([7:10]), angular velocity([10:13])
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # create views of dof tensor
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.n_hand_dofs, 2)[..., 1]

        # refresh all tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


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

    def _get_hand_asset(self):

        asset_root = add_assets_path('robot_assets/smg')
        asset_file =  "smg_tactip.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.collapse_fixed_joints = True
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
        self.num_control_dofs = len(self.control_joint_names)

        self.init_joint_mins = {
            "J1": -20.0*(np.pi/180),
            "J2": -25.0*(np.pi/180),
            "J3": -20.0*(np.pi/180),
        }
        self.init_joint_maxs = {
            "J1": 20.0*(np.pi/180),
            "J2": -15.0*(np.pi/180),
            "J3": 20.0*(np.pi/180),
        }

        self.control_joint_dof_indices = [self.gym.find_asset_dof_index(hand_asset, name) for name in self.control_joint_names]
        self.control_joint_dof_indices = to_torch(self.control_joint_dof_indices, dtype=torch.long, device=self.device)

        return hand_asset

    def _get_obj_asset(self):

        model_list = object_set.getModelList()
        asset_root = object_set.getDataPath()
        asset_file = os.path.join('sphere', "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        obj_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # set initial state for the object
        self.init_obj_pose = gymapi.Transform()
        self.init_obj_pose.p = gymapi.Vec3(0.0, 0.0, 0.445)
        self.init_obj_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.init_obj_vel = gymapi.Velocity()
        self.init_obj_vel.linear = gymapi.Vec3(0.0, 0.0, 0.0)
        self.init_obj_vel.angular = gymapi.Vec3(0.0, 0.0, 0.0)

        return obj_asset

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.hand_asset = self._get_hand_asset()
        self.obj_asset = self._get_obj_asset()

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

        # conver states to tensor
        self.init_obj_states = to_torch(self.init_obj_states, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        # set initial joint state tensor
        self.current_joint_state = torch.zeros((self.num_envs, self.num_control_dofs), dtype=torch.float, device=self.device)

    def _create_hand_actor(self, env_ptr, idx):

        self.n_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.n_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        self.n_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)

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
        obj_props[0].mass = 1.0
        self.gym.set_actor_rigid_body_properties(env_ptr, handle, obj_props)

        return handle

    def compute_reward(self):
        # retrieve environment observations from buffer
        self.rew_buf[:], self.reset_buf[:] = compute_smg_reward(
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)

        # self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        # self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        # self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        # self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()

        return self.obs_buf

    def reset(self, env_ids):

        # reset hand
        hand_positions = torch.zeros((len(env_ids), self.n_hand_dofs), device=self.device)
        hand_velocities = torch.zeros((len(env_ids), self.n_hand_dofs), device=self.device)

        self.dof_pos[env_ids, :] = hand_positions[:]
        self.dof_vel[env_ids, :] = hand_velocities[:]

        hand_ids_int32 = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_ids_int32),
            len(env_ids)
        )

        # reset object
        self.root_state_tensor[self.obj_indices[env_ids]] = self.init_obj_states[env_ids].clone()
        obj_ids_int32 = self.obj_indices[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_ids_int32),
            len(env_ids)
        )

        # reset buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # set initial joint state tensor
        self.current_joint_state = torch.zeros((self.num_envs, self.num_control_dofs), dtype=torch.float, device=self.device)

    def pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.device)
        self.cur_targets = self.current_joint_state[:, self.control_joint_dof_indices] + self.dof_speed_scale * self.dt * self.actions

        self.current_joint_state[:, self.control_joint_dof_indices] = self.cur_targets[:, self.control_joint_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))


    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_smg_reward(
        rew_buf: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        max_episode_length: float
    ): # -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = torch.ones_like(rew_buf)

    # set envs to terminate when reach criteria
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset
