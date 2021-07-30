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

        # what object to use
        self.obj_name = 'sphere'
        # self.obj_name = 'cube'
        # self.obj_name = 'icosahedron'

        super(SMGManip, self).__init__(
            cfg,
            sim_params,
            physics_engine,
            device_type,
            device_id,
            headless
        )


    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)

        # full rand
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
