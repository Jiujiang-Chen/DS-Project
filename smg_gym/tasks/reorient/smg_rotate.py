"""
Train:
python train.py task=smg_rotate headless=True

Test:
python train.py task=smg_rotate test=True task.env.num_envs=8 headless=False checkpoint=runs/smg_rotate/nn/smg_rotate.pth
"""
import torch

from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.reorient.base_reorient import BaseReorient


class SMGRotate(BaseReorient):

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
        obj_keypoint_pos (18)
        tcp_pos (9)
        goal_pose (7)
        goal_keypoint_pos (18)
        rel_goal_orn (4)

        total = 99
        """
        cfg["env"]["numObservations"] = self.calculate_buffer_size(cfg["enabled_obs"])

        if cfg["asymmetric_obs"]:
            cfg["env"]["numStates"] = self.calculate_buffer_size(cfg["enabled_states"])

        cfg["env"]["numActions"] = 9

        super(SMGRotate, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_pose(self, goal_env_ids_for_reset):
        rand_floats = torch_rand_float(
            -1.0, 1.0,
            (len(goal_env_ids_for_reset), 1),
            device=self.device
        )

        # rotate
        new_goal_rot = randomize_rotation(
            torch.zeros(size=(len(goal_env_ids_for_reset),), device=self.device),
            rand_floats[:, 0],
            self.x_unit_tensor[goal_env_ids_for_reset],
            self.z_unit_tensor[goal_env_ids_for_reset]
        )

        self.root_state_tensor[self.goal_indices[goal_env_ids_for_reset], 3:7] = new_goal_rot
        self.reset_goal_buf[goal_env_ids_for_reset] = 0
