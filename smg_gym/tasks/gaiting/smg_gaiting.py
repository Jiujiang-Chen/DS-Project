"""
Train:
python train.py task=smg_gaiting headless=True

Test:
python train.py task=smg_gaiting task.env.numEnvs=8 test=True headless=False checkpoint=runs/smg_gaiting/nn/smg_gaiting.pth
"""
import torch

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import torch_rand_float

from smg_gym.rl_games_helpers.utils.torch_jit_utils import randomize_rotation
from smg_gym.tasks.gaiting.base_hand_env import BaseShadowModularGrasper


class SMGGaiting(BaseShadowModularGrasper):

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
        cfg["env"]["numObservations"] = 62
        cfg["env"]["numActions"] = 9

        # what object to use
        self.obj_name = 'sphere'
        # self.obj_name = 'cube'
        # self.obj_name = 'icosahedron'

        super(SMGGaiting, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_axis(self, env_ids_for_reset):
        """Set target axis to rotate the object about.
        TODO: reset only the neccessary envs"""

        # get base pose of pivot point.
        pivot_states = self.rigid_body_tensor[:, self.pivot_point_body_idxs, :]

        # randomise position of pivot point
        if self.randomize and self.rand_pivot_pos:
            pos_offset = torch_rand_float(
                -0.025, 0.025,
                (self.num_envs, 3),
                device=self.device
            )
        else:
            pos_offset = to_torch([0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.pivot_point_pos = pivot_states[..., 0:3].reshape(self.num_envs, 3) + pos_offset

        # randomise orientation of pivot point
        if self.randomize and self.rand_pivot_orn:
            rand_floats = torch_rand_float(
                -1.0, 1.0,
                (self.num_envs, 2),
                device=self.device
            )

            # full rand
            self.pivot_point_orn = randomize_rotation(
                rand_floats[:, 0],
                rand_floats[:, 1],
                self.x_unit_tensor,
                self.y_unit_tensor
            )

        else:
            self.pivot_point_orn = pivot_states[..., 3:7].reshape(self.num_envs, 4)
