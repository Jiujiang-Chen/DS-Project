"""
Train:
python train.py task=smg_pivot headless=True

Test:
python train.py task=smg_pivot task.env.numEnvs=8 test=True headless=False checkpoint=runs/smg_pivot/nn/smg_pivot.pth
"""

import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.reorient.base_hand_env import BaseShadowModularGrasper


class SMGPivot(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
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

        super(SMGPivot, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_pose(self, env_ids, apply_reset=False):

        # rand floats shape (n_envs, 3)
        rand_floats = torch_rand_float(
            -1.0, 1.0,
            (len(env_ids), 3),
            device=self.device
        )

        # based on 3rd float zero half for pivoting around either x or y axis
        rand_floats[:, 0] = torch.where(
            rand_floats[:, 2] > 0,
            torch.zeros(size=(len(env_ids),), device=self.device),
            rand_floats[:, 0],
        )
        rand_floats[:, 1] = torch.where(
            rand_floats[:, 2] <= 0,
            torch.zeros(size=(len(env_ids),), device=self.device),
            rand_floats[:, 1],
        )

        # pivot 1
        new_goal_rot = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids]
        )

        self.root_state_tensor[
            self.goal_indices[env_ids]
        ] = self.init_goal_states[env_ids].clone()
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
