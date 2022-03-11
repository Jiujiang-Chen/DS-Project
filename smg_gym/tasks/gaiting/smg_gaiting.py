"""
Train:
python train.py task=smg_gaiting headless=True

Test:
python train.py task=smg_gaiting task.env.numEnvs=8 test=True headless=False checkpoint=runs/smg_gaiting/nn/smg_gaiting.pth
"""

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

    def reset_target_axis(self, env_ids, apply_reset=False):
        pass
