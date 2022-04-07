import torch
import numpy as np

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgym.torch_utils import to_torch
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_rotate
from isaacgym.torch_utils import quat_from_angle_axis

from smg_gym.tasks.base_hand import BaseShadowModularGrasper
from smg_gym.tasks.reorient.rewards import compute_keypoint_reorient_reward
from smg_gym.tasks.reorient.rewards import compute_hybrid_reorient_reward


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
        default_pivot_axel = np.array(cfg["env"]["default_pivot_axel"])
        self.default_pivot_axel = default_pivot_axel/np.linalg.norm(default_pivot_axel)

        # reward / termination vars
        self.reward_type = cfg["env"]["reward_type"]
        self.max_episode_length = cfg["env"]["episode_length"]

        if self.reward_type not in ["hybrid", "keypoint"]:
            raise ValueError('Incorrect reward mode specified.')

        # randomisation params
        self.randomize = cfg["rand_params"]["randomize"]
        self.rand_hand_joints = cfg["rand_params"]["rand_hand_joints"]
        self.rand_obj_init_orn = cfg["rand_params"]["rand_obj_init_orn"]
        self.rand_pivot_pos = cfg["rand_params"]["rand_pivot_pos"]
        self.rand_pivot_axel = cfg["rand_params"]["rand_pivot_axel"]

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

    def fill_observation_buffer(self):
        """
        Fill observation buffer.
        shape = (num_envs, num_obs)
        """

        # fill obs buffer with observations shared across tasks.
        obs_cfg = self.cfg["enabled_obs"]
        start_offset, end_offset = self.standard_fill_buffer(self.obs_buf, obs_cfg)

        # target pivot axel vec
        start_offset = end_offset
        end_offset = start_offset + 3
        if obs_cfg["pivot_axel_vec"]:
            self.obs_buf[:, start_offset:end_offset] = self.pivot_axel_workframe

        # target pivot axel pos
        start_offset = end_offset
        end_offset = start_offset + 3
        if obs_cfg["pivot_axel_pos"]:
            self.obs_buf[:, start_offset:end_offset] = self.pivot_point_pos_offset

        return self.obs_buf

    def fill_states_buffer(self):
        """
        Fill states buffer.
        shape = (num_envs, num_obs)
        """
        # if states spec is empty then return
        if not self.cfg["asymmetric_obs"]:
            return

        # fill obs buffer with observations shared across tasks.
        states_cfg = self.cfg["enabled_states"]
        start_offset, end_offset = self.standard_fill_buffer(self.states_buf, states_cfg)

        # target pivot axel vec
        start_offset = end_offset
        end_offset = start_offset + 3
        if states_cfg["pivot_axel_vec"]:
            self.states_buf[:, start_offset:end_offset] = self.pivot_axel_workframe

        # target pivot axel pos
        start_offset = end_offset
        end_offset = start_offset + 3
        if states_cfg["pivot_axel_pos"]:
            self.states_buf[:, start_offset:end_offset] = self.pivot_point_pos_offset

        return self.states_buf

    def compute_reward_and_termination(self):
        """
        Calculate the reward and termination (including goal successes) per env.
        """

        centered_obj_kp_pos = self.obj_kp_positions - \
            self.obj_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1)
        centered_goal_kp_pos = self.goal_kp_positions - \
            self.goal_displacement_tensor.unsqueeze(0).unsqueeze(1).repeat(self.num_envs, self.n_keypoints, 1) - \
            self.pivot_point_pos_offset.unsqueeze(1).repeat(1, self.n_keypoints, 1)

        # shadow hand pos + orn distance rew
        if self.reward_type == 'hybrid':
            (
                self.rew_buf[:],
                self.reset_buf[:],
                self.reset_goal_buf[:],
                self.successes[:],
                self.consecutive_successes[:],
                log_dict
            ) = compute_hybrid_reorient_reward(
                rew_buf=self.rew_buf,
                reset_buf=self.reset_buf,
                progress_buf=self.progress_buf,
                reset_goal_buf=self.reset_goal_buf,
                successes=self.successes,
                consecutive_successes=self.consecutive_successes,
                obj_base_pos=self.obj_base_pos - self.obj_displacement_tensor,
                obj_base_orn=self.obj_base_orn,
                targ_base_pos=self.goal_base_pos - self.goal_displacement_tensor,
                targ_base_orn=self.goal_base_orn,
                actions=self.action_buf,
                n_tip_contacts=self.n_tip_contacts,
                dist_reward_scale=self.cfg["env"]["dist_reward_scale"],
                rot_reward_scale=self.cfg["env"]["rot_reward_scale"],
                rot_eps=self.cfg["env"]["rot_eps"],
                success_tolerance=self.cfg["env"]["rot_success_tolerance"],
                max_episode_length=self.cfg["env"]["episode_length"],
                fall_reset_dist=self.cfg["env"]["fall_reset_dist"],
                require_contact=self.cfg["env"]["require_contact"],
                contact_reward_scale=self.cfg["env"]["contact_reward_scale"],
                action_penalty_scale=self.cfg["env"]["action_penalty_scale"],
                reach_goal_bonus=self.cfg["env"]["reach_goal_bonus"],
                fall_penalty=self.cfg["env"]["fall_penalty"],
                av_factor=self.cfg["env"]["av_factor"],
            )

        # trifinger - keypoint distance reward
        elif self.reward_type == 'keypoint':
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
                obj_kps=centered_obj_kp_pos,
                goal_kps=centered_goal_kp_pos,
                actions=self.action_buf,
                n_tip_contacts=self.n_tip_contacts,
                lgsk_scale=self.cfg["env"]["lgsk_scale"],
                lgsk_eps=self.cfg["env"]["lgsk_eps"],
                kp_dist_scale=self.cfg["env"]["kp_dist_scale"],
                success_tolerance=self.cfg["env"]["kp_success_tolerance"],
                max_episode_length=self.cfg["env"]["episode_length"],
                fall_reset_dist=self.cfg["env"]["fall_reset_dist"],
                require_contact=self.cfg["env"]["require_contact"],
                contact_reward_scale=self.cfg["env"]["contact_reward_scale"],
                action_penalty_scale=self.cfg["env"]["action_penalty_scale"],
                reach_goal_bonus=self.cfg["env"]["reach_goal_bonus"],
                fall_penalty=self.cfg["env"]["fall_penalty"],
                av_factor=self.cfg["env"]["av_factor"],
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
