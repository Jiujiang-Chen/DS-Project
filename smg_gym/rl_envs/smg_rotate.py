import numpy as np
import os
import torch
from torch import Tensor
from rlgpu.utils.torch_jit_utils import *

from smg_gym.rl_envs.base_hand_env import BaseShadowModularGrasper

class SMGRotate(BaseShadowModularGrasper):

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
        keypoint_pos (9)
        tcp_pos (9)

        total = 61
        """
        cfg["env"]["numObservations"] = 61
        cfg["env"]["numActions"] = 9

        super(SMGRotate, self).__init__(
            cfg,
            sim_params,
            physics_engine,
            device_type,
            device_id,
            headless
        )

    def compute_reward(self):
        """
        Reward computed after observation so vars set in compute_obs can
        be used here
        """

        init_obj_pos = self.init_obj_states[..., :3]

        # retrieve environment observations from buffer
        self.rew_buf[:], self.reset_buf[:] = compute_rotate_reward(
            self.obj_base_pos,
            self.obj_base_orn,
            self.obj_base_angvel,
            init_obj_pos,
            self.n_tip_contacts ,
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.fall_reset_dist
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_rotate_reward(
        obj_base_pos: Tensor,
        obj_base_orn: Tensor,
        obj_base_angvel: Tensor,
        init_obj_pos: Tensor,
        n_tip_contacts : Tensor,
        rew_buf: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        max_episode_length: float,
        fall_reset_dist: float
    ): # -> Tuple[Tensor, Tensor]

    # calc distance from starting position
    dist_from_start = torch.norm(obj_base_pos - init_obj_pos, p=2, dim=-1)

    # convert obj angvel into object baseframe
    # obj_angvel_objframe = quat_rotate(obj_base_orn, obj_base_angvel)

    # angular velocity around z axis
    reward = torch.ones_like(rew_buf) * obj_base_angvel[..., 2] * 1.0

    # add smaller penalty for velocity in other directions
    # reward = reward - (obj_base_angvel[..., 0] + obj_base_angvel[..., 1]) * 0.1

    # add small penalty for moving object too far
    # reward = reward - (dist_from_start * 0.1)

    # zero reward when less than 2 tips in contact
    reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), reward)

    # set envs to terminate when reach criteria

    # end of episode
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # obj droped too far
    reset = torch.where(dist_from_start >= fall_reset_dist, torch.ones_like(reset_buf), reset)

    return reward, reset
