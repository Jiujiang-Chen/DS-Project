from isaacgym.torch_utils import unscale
from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate

from smg_gym.tasks.base_hand import BaseShadowModularGrasper
from smg_gym.tasks.reorient.rewards import compute_hybrid_reorient_reward
from smg_gym.tasks.reorient.rewards import compute_keypoint_reorient_reward


class BaseReorient(BaseShadowModularGrasper):

    def __init__(
        self,
        cfg,
        sim_device,
        graphics_device_id,
        headless
    ):

        # reward / termination vars
        self.reward_type = cfg["env"]["reward_type"]
        self.max_episode_length = cfg["env"]["episode_length"]

        if self.reward_type not in ["hybrid", "keypoint"]:
            raise ValueError('Incorrect reward mode specified.')

        # randomisation params
        self.randomize = cfg["rand_params"]["randomize"]
        self.rand_hand_joints = cfg["rand_params"]["rand_hand_joints"]
        self.rand_obj_init_orn = cfg["rand_params"]["rand_obj_init_orn"]

        super(BaseReorient, self).__init__(
            cfg,
            sim_device,
            graphics_device_id,
            headless
        )

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
        self.obs_buf[:, 31:40] = self.action_buf
        self.obs_buf[:, 40:43] = self.tip_object_contacts
        self.obs_buf[:, 43:52] = self.fingertip_pos
        self.obs_buf[:, 52:55] = self.goal_base_pos
        self.obs_buf[:, 55:59] = self.goal_base_orn
        self.obs_buf[:, 59:63] = quat_mul(self.obj_base_orn, quat_conjugate(self.goal_base_orn))
        self.obs_buf[:, 63:81] = (self.obj_kp_positions
                                  - self.obj_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)
        self.obs_buf[:, 81:99] = (self.goal_kp_positions
                                  - self.goal_displacement_tensor).reshape(self.num_envs, self.n_keypoints*3)

        return self.obs_buf

    def compute_reward_and_termination(self):
        """
        Calculate the reward and termination (including goal successes) per env.
        """

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
                obj_kps=self.obj_kp_positions - self.obj_displacement_tensor,
                goal_kps=self.goal_kp_positions - self.goal_displacement_tensor,
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
