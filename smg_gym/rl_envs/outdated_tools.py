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
