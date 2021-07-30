    def apply_grasp_action(self, env_ids):

        grasp_action = torch.zeros((self.num_envs, self.n_hand_dofs), device=self.device)
        grasp_action[env_ids, 1::3] = 1.0

        # apply actions
        self.actions = grasp_action.clone().to(self.device)
        self.cur_targets = self.current_joint_state[:, self.control_joint_dof_indices] + self.dof_speed_scale * self.dt * self.actions

        self.current_joint_state[:, self.control_joint_dof_indices] = self.cur_targets[:, self.control_joint_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)



    def compute_reward(self):
        """
        Reward computed after observation so vars set in compute_obs can
        be used here
        """

        init_obj_pos = self.init_obj_states[..., :3]

        # retrieve environment observations from buffer
        self.rew_buf[:], self.reset_buf[:] = compute_rotate_reward(
            self.obj_base_pos,
            self.obj_base_orn,
            self.obj_base_linvel,
            self.obj_base_angvel,
            init_obj_pos,
            self.n_tip_contacts ,
            self.rew_buf,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.fall_reset_dist
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_rotate_reward(
        obj_base_pos: Tensor,
        obj_base_orn: Tensor,
        obj_base_linvel: Tensor,
        obj_base_angvel: Tensor,
        init_obj_pos: Tensor,
        n_tip_contacts : Tensor,
        rew_buf: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        max_episode_length: float,
        fall_reset_dist: float
    ): # -> Tuple[Tensor, Tensor]

    angvel_reward_weight = 1.0
    angvel_penalty_weight = -0.0
    linvel_penalty_weight = -0.0
    distance_penalty_weight = -1.0

    # calc distance from starting position
    dist_from_start = torch.norm(obj_base_pos - init_obj_pos, p=2, dim=-1)

    # convert obj angvel into object baseframe
    # obj_angvel_objframe = quat_rotate(obj_base_orn, obj_base_angvel)

    # angular velocity around z axis
    angvel_reward = torch.ones_like(rew_buf) * obj_base_angvel[..., 2] * angvel_reward_weight

    # add penalty for velocity in other directions
    linvel_penalty = torch.sum(obj_base_linvel ** 2, dim=-1)
    angvel_penalty = obj_base_angvel[..., 0]**2 + obj_base_angvel[..., 1]**2

    # combine reward
    reward = (
        angvel_reward +
        (angvel_penalty * angvel_penalty_weight) +
        (linvel_penalty * linvel_penalty_weight) +
        (dist_from_start * distance_penalty_weight)
    )

    # zero reward when less than 2 tips in contact
    reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), reward)

    # set envs to terminate when reach criteria

    # end of episode
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # obj droped too far
    reset = torch.where(dist_from_start >= fall_reset_dist, torch.ones_like(reset_buf), reset)

    return reward, reset
