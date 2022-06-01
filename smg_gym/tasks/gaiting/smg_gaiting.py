"""
Train:
python train.py task=smg_gaiting headless=True

# train low env num
python train.py task=smg_gaiting task.env.num_envs=8 headless=False train.params.config.horizon_length=1024  train.params.config.minibatch_size=32

Test:
python train.py task=smg_gaiting task.env.num_envs=8 test=True headless=False checkpoint=runs/smg_gaiting/nn/smg_gaiting.pth
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
            joint_pos: 9
            joint_vel: 9
            joint_eff: 9
            fingertip_pos: 9
            fingertip_orn: 12
            last_action: 9
            bool_tip_contacts: 3
            tip_contact_forces: 9
            ft_sensor_contact_forces: 9
            ft_sensor_contact_torques: 9
            tip_contact_positions: 9
            tip_contact_normals: 9
            tip_contact_force_magnitudes: 3
            object_pos: 3
            object_orn: 4
            object_kps: 18
            object_linvel: 3
            object_angvel: 3
            goal_pos: 3
            goal_orn: 4
            goal_kps: 18
            active_quat: 4
            pivot_axel_vec: 3
            pivot_axel_pos: 3
        max_total = 147
        """

        cfg["env"]["numObservations"] = 174

        if cfg["asymmetric_obs"]:
            cfg["env"]["numStates"] = 174

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

        # set the default axel direction
        self.pivot_axel_workframe[env_ids_for_reset, :] = to_torch(
            self.default_pivot_axel,
            dtype=torch.float,
            device=self.device
        ).repeat((num_envs_to_reset, 1))

        # randomise direction of pivot axel
        if self.randomize:
            if self.rand_pivot_axel == 'full_rand':
                self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_dir(
                    num_envs_to_reset,
                    device=self.device
                )
            elif self.rand_pivot_axel == 'cardinal':
                self.pivot_axel_workframe[env_ids_for_reset, :] = torch_random_cardinal_dir(
                    num_envs_to_reset,
                    device=self.device
                )

        self.pivot_axel_worldframe[env_ids_for_reset, :] = quat_rotate(
            pivot_point_orn, self.pivot_axel_workframe[env_ids_for_reset, :])

        # find the same pivot axel in the object frame
        obj_base_orn = self.root_state_tensor[self.obj_indices, 3:7]
        self.pivot_axel_objframe[env_ids_for_reset] = quat_rotate_inverse(
            obj_base_orn[env_ids_for_reset], self.pivot_axel_worldframe[env_ids_for_reset, :])
