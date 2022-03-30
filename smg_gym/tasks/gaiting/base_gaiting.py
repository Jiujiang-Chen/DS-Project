import torch
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import unscale
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import quat_from_angle_axis

from smg_gym.tasks.base_hand import BaseShadowModularGrasper
from smg_gym.tasks.reorient.rewards import compute_keypoint_reorient_reward


class BaseGaiting(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):
        # target vars
        self.rotate_increment_degrees = cfg["env"]["rotateIncrementDegrees"]

        # reward / termination vars
        self.reward_type = cfg["env"]["rewardType"]
        self.max_episode_length = cfg["env"]["episodeLength"]

        if self.reward_type not in ["keypoint"]:
            raise ValueError('Incorrect reward mode specified.')

        # randomisation params
        self.randomize = cfg["task"]["randomize"]
        self.rand_hand_joints = cfg["task"]["randHandJoints"]
        self.rand_obj_init_orn = cfg["task"]["randObjInitOrn"]
        self.rand_pivot_pos = cfg["task"]["randPivotPos"]
        self.rand_pivot_axel = cfg["task"]["randPivotAxel"]

        super(BaseGaiting, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

        self._setup_pivot_point()

    def _setup_pivot_point(self):

        # fixed vars
        self.pivot_line_scale = 0.2
        self.pivot_axel_p1 = to_torch(self.default_goal_pos, dtype=torch.float,
                                      device=self.device).repeat((self.num_envs, 1))

        # vars that change on reset (with randomisation)
        self.pivot_point_pos_offset = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_point_pos = torch.zeros(size=(self.num_envs, 3), device=self.device)

        self.pivot_axel_worldframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_workframe = torch.zeros(size=(self.num_envs, 3), device=self.device)
        self.pivot_axel_objframe = torch.zeros(size=(self.num_envs, 3), device=self.device)

        # get indices of pivot points
        self.pivot_point_body_idxs = self._get_pivot_point_idxs(self.envs[0], self.hand_actor_handles[0])

        # amount by which to rotate
        self.rotate_increment = torch.ones(size=(self.num_envs, ), device=self.device) * \
            self.rotate_increment_degrees * np.pi / 180

    def _get_pivot_point_idxs(self, env, hand_actor_handle):

        hand_body_names = self.gym.get_actor_rigid_body_names(env, hand_actor_handle)
        body_names = [name for name in hand_body_names if 'pivot_point' in name]
        body_idxs = [self.gym.find_actor_rigid_body_index(
            env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in body_names]

        return body_idxs

    def reset_target_pose(self, goal_env_ids_for_reset):
        """
        Reset target pose to initial pose of the object.
        """

        self.root_state_tensor[
            self.goal_indices[goal_env_ids_for_reset], 3:7
        ] = self.root_state_tensor[self.obj_indices[goal_env_ids_for_reset], 3:7]

        self.goal_base_pos = self.root_state_tensor[self.goal_indices, 0:3]
        self.goal_base_orn = self.root_state_tensor[self.goal_indices, 3:7]

    def rotate_target_pose(self, goal_env_ids_for_reset):
        """
        Rotate the target pose around the pivot axel.
        """
        # rotate goal pose
        rotate_quat = quat_from_angle_axis(self.rotate_increment, self.pivot_axel_objframe)

        self.root_state_tensor[
            self.goal_indices[goal_env_ids_for_reset], 3:7
        ] = quat_mul(self.goal_base_orn, rotate_quat)[goal_env_ids_for_reset, :]
        self.reset_goal_buf[goal_env_ids_for_reset] = 0

    def apply_resets(self):
        """
        Logic for applying resets
        """

        env_ids_for_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids_for_reset = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        actor_root_state_reset_indices = []

        # nothing to reset
        if len(env_ids_for_reset) == 0 and len(goal_env_ids_for_reset) == 0:
            return None

        # reset envs
        if len(env_ids_for_reset) > 0:
            self.reset_hand(env_ids_for_reset)
            self.reset_object(env_ids_for_reset)
            self.reset_target_pose(env_ids_for_reset)
            self.reset_target_axis(env_ids_for_reset)
            self.rotate_target_pose(env_ids_for_reset)
            actor_root_state_reset_indices.append(self.obj_indices[env_ids_for_reset])
            actor_root_state_reset_indices.append(self.goal_indices[env_ids_for_reset])

        # reset goals
        if len(goal_env_ids_for_reset) > 0:
            self.rotate_target_pose(goal_env_ids_for_reset)
            actor_root_state_reset_indices.append(self.goal_indices[goal_env_ids_for_reset])

        # set the root state tensor to reset object and goal pose
        # has to be done together for some reason...
        reset_indices = torch.unique(
            torch.cat(actor_root_state_reset_indices)
        ).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(reset_indices),
            len(reset_indices)
        )

        # reset buffers
        self.progress_buf[env_ids_for_reset] = 0
        self.reset_buf[env_ids_for_reset] = 0
        self.successes[env_ids_for_reset] = 0

    def fill_observations(self):
        """
        Fill observation buffer.
        """

        # obs_buf shape=(num_envs, num_obs)
        self.obs_buf[:, :9] = unscale(self.hand_joint_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        self.obs_buf[:, 9:18] = self.hand_joint_vel
        self.obs_buf[:, 18:21] = self.obj_base_pos
        self.obs_buf[:, 21:25] = self.obj_base_orn
        self.obs_buf[:, 25:28] = self.obj_base_linvel
        self.obs_buf[:, 28:31] = self.obj_base_angvel
        self.obs_buf[:, 31:40] = self.actions
        self.obs_buf[:, 40:43] = self.tip_contacts
        self.obs_buf[:, 43:52] = self.fingertip_pos
        self.obs_buf[:, 52:55] = self.goal_base_pos
        self.obs_buf[:, 55:59] = self.goal_base_orn
        self.obs_buf[:, 59:63] = quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn))
        self.obs_buf[:, 63:66] = self.pivot_axel_workframe
        self.obs_buf[:, 66:69] = self.pivot_point_pos_offset
        self.obs_buf[:, 69:87] = (self.obj_kp_positions
                                  - self.obj_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)
        self.obs_buf[:, 87:105] = (self.goal_kp_positions
                                   - self.goal_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)
        return self.obs_buf

    def compute_reward_and_termination(self):
        """
        Calculate the reward and termination (including goal successes) per env.
        """
        # trifinger - keypoint distance reward
        if self.reward_type == 'keypoint':
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.reset_goal_buf[:],
                self.successes[:],
                self.consecutive_successes[:],
                log_dict
            ) = compute_keypoint_reorient_reward(
                rew_buf=self.rew_buf,
                reset_buf=self.reset_buf,
                progress_buf=self.progress_buf,
                reset_goal_buf=self.reset_goal_buf,
                successes=self.successes,
                consecutive_successes=self.consecutive_successes,
                obj_kps=self.obj_kp_positions - self.obj_displacement_tensor,
                goal_kps=self.goal_kp_positions - self.goal_displacement_tensor,
                actions=self.actions,
                n_tip_contacts=self.n_tip_contacts,
                lgsk_scale=self.cfg["env"]["lgskScale"],
                lgsk_eps=self.cfg["env"]["lgskEps"],
                kp_dist_scale=self.cfg["env"]["kpDistScale"],
                success_tolerance=self.cfg["env"]["kpSuccessTolerance"],
                max_episode_length=self.cfg["env"]["episodeLength"],
                fall_reset_dist=self.cfg["env"]["fallResetDist"],
                require_contact=self.cfg["env"]["requireContact"],
                contact_reward_scale=self.cfg["env"]["contactRewardScale"],
                action_penalty_scale=self.cfg["env"]["actionPenaltyScale"],
                reach_goal_bonus=self.cfg["env"]["reachGoalBonus"],
                fall_penalty=self.cfg["env"]["fallPenalty"],
                av_factor=self.cfg["env"]["avFactor"],
            )

        self.extras.update({"metrics/"+k: v.mean() for k, v in log_dict.items()})

    def visualise_features(self):

        super().visualise_features()

        for i in range(self.num_envs):

            # visualise pivot axel
            pivot_p1 = gymapi.Vec3(
                self.pivot_axel_p1[i, 0],
                self.pivot_axel_p1[i, 1],
                self.pivot_axel_p1[i, 2]
            )

            pivot_axel_p2_worldframe = self.pivot_axel_p1 + self.pivot_axel_worldframe * self.pivot_line_scale
            pivot_p2 = gymapi.Vec3(
                pivot_axel_p2_worldframe[i, 0],
                pivot_axel_p2_worldframe[i, 1],
                pivot_axel_p2_worldframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2,
                gymapi.Vec3(1.0, 1.0, 0.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )

            # visualise object frame pivot_axel
            current_obj_pivot_axel_worldframe = quat_rotate(self.obj_base_orn, self.pivot_axel_objframe)
            pivot_axel_p2_objframe = self.pivot_axel_p1 + current_obj_pivot_axel_worldframe * self.pivot_line_scale

            pivot_p2_objframe = gymapi.Vec3(
                pivot_axel_p2_objframe[i, 0],
                pivot_axel_p2_objframe[i, 1],
                pivot_axel_p2_objframe[i, 2]
            )

            gymutil.draw_line(
                pivot_p1,
                pivot_p2_objframe,
                gymapi.Vec3(0.0, 1.0, 1.0),
                self.gym,
                self.viewer,
                self.envs[i],
            )
