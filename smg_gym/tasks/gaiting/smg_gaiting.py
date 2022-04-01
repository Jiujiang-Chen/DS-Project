"""
Train:
python train.py task=smg_gaiting headless=True

# train low env num
python train.py task=smg_gaiting task.env.numEnvs=8 headless=False

Test:
python train.py task=smg_gaiting task.env.numEnvs=8 test=True headless=False checkpoint=runs/smg_gaiting/nn/smg_gaiting.pth
"""
import torch

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import torch_rand_float
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import quat_rotate_inverse

from smg_gym.tasks.gaiting.base_gaiting import BaseGaiting
from smg_gym.utils.torch_jit_utils import torch_random_dir
from smg_gym.utils.torch_jit_utils import torch_random_cardinal_dir


class SMGGaiting(BaseGaiting):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):
        """
        Obs:
            jointPos: 9
            jointVel: 9
            fingertipPos: 9
            fingertipOrn: 12
            lastAction: 9
            boolTipContacts: 3
            tipContactForces: 9
            objectPos: 3
            objectOrn: 4
            objectKPs: 18
            objectLinVel: 3
            objectAngVel: 3
            GoalPos: 3
            GoalOrn: 4
            GoalKPs: 18
            activeQuat: 4
            pivotAxelVec: 3
            pivotAxelPos: 3
        max_total = 126
        """
        cfg["env"]["numObservations"] = 126

        if cfg["asymmetricObs"]:
            cfg["env"]["numStates"] = 126

        cfg["env"]["numActions"] = 9

        super(SMGGaiting, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

    def reset_target_axis(self, env_ids_for_reset):
        """Set target axis to rotate the object about."""

        num_envs_to_reset = len(env_ids_for_reset)

        # get base pose of pivot point.
        pivot_states = self.rigid_body_tensor[env_ids_for_reset, self.pivot_point_body_idxs, :]
        pivot_point_pos = pivot_states[..., 0:3]
        pivot_point_orn = pivot_states[..., 3:7]

        # randomise position of pivot point
        if self.randomize and self.rand_pivot_pos:
            self.pivot_point_pos_offset[env_ids_for_reset, :] = torch_rand_float(
                -0.025, 0.025,
                (num_envs_to_reset, 3),
                device=self.device
            )
        else:
            self.pivot_point_pos_offset[env_ids_for_reset, :] = to_torch(
                [0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat((num_envs_to_reset, 1))

        self.pivot_point_pos[env_ids_for_reset, :] = pivot_point_pos + self.pivot_point_pos_offset[env_ids_for_reset, :]

        # randomise direction of pivot axel
        if self.randomize and self.rand_pivot_axel:
            # self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_dir(
            #     num_envs_to_reset,
            #     device=self.device
            # )
            self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_cardinal_dir(
                num_envs_to_reset,
                device=self.device
            )
        else:
            self.pivot_axel_workframe[env_ids_for_reset, :] = to_torch(
                [0.0, 0.0, 1.0],
                dtype=torch.float,
                device=self.device
            ).repeat((num_envs_to_reset, 1))

        self.pivot_axel_worldframe[env_ids_for_reset, :] = quat_rotate(
            pivot_point_orn, self.pivot_axel_workframe[env_ids_for_reset, :])

        # find the same pivot axel in the object frame
        obj_base_orn = self.root_state_tensor[self.obj_indices, 3:7]
        self.pivot_axel_objframe[env_ids_for_reset] = quat_rotate_inverse(
            obj_base_orn[env_ids_for_reset], self.pivot_axel_worldframe[env_ids_for_reset, :])
