"""
Train:
python train.py task=smg_reorient headless=True

Test:
python train.py task=smg_reorient task.env.num_envs=8 test=True headless=False checkpoint=runs/smg_reorient/nn/smg_reorient.pth
"""
from isaacgym.torch_utils import torch_rand_float

from smg_gym.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.reorient.base_reorient import BaseReorient


class SMGReorient(BaseReorient):

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

        super(SMGReorient, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose of the object.
        """
        rand_floats = torch_rand_float(
            -1.0, 1.0,
            (len(goal_env_ids_for_reset), 2),
            device=self.device
        )

        # full rand
        new_goal_quat = randomize_rotation(
            rand_floats[:, 0],
            rand_floats[:, 1],
            self.x_unit_tensor[goal_env_ids_for_reset],
            self.y_unit_tensor[goal_env_ids_for_reset]
        )

        self.root_state_tensor[self.goal_indices[goal_env_ids_for_reset], 3:7] = new_goal_quat
        self.reset_goal_buf[goal_env_ids_for_reset] = 0
