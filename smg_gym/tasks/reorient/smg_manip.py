"""
Train:
python train.py task=smg_manip headless=True

Test:
python train.py task=smg_manip task.env.numEnvs=8 test=True headless=False checkpoint=runs/smg_manip/nn/smg_manip.pth
"""

import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.reorient.base_hand_env import BaseShadowModularGrasper


class SMGManip(BaseShadowModularGrasper):

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
        cfg["env"]["numObservations"] = 99
        cfg["env"]["numActions"] = 9

        # what object to use
        self.obj_name = 'sphere'
        # self.obj_name = 'cube'
        # self.obj_name = 'icosahedron'

        super(SMGManip, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(
            -1.0, 1.0,
            (len(env_ids), 2),
            device=self.device
        )

        # full rand
        new_goal_quat = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids]
        )

        self.root_state_tensor[self.goal_indices[env_ids], 3:7] = new_goal_quat

        if apply_reset:
            reset_goal_indices = self.goal_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(reset_goal_indices),
                len(env_ids)
            )

        self.reset_goal_buf[env_ids] = 0
