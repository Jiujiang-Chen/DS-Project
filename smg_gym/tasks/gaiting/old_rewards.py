

@torch.jit.script
def compute_manip_reward_conditional(
                obj_base_pos: torch.Tensor,
                pivot_point_pos: torch.Tensor,
                target_pivot_axel: torch.Tensor,
                current_pivot_axel: torch.Tensor,
                object_angvel: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                require_contact: bool,
                pos_reward_scale: float,
                orn_reward_scale: float,
                vel_reward_scale: float,
                contact_reward_scale: float,
                action_penalty_scale: float,
                fall_penalty_scale: float,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
                angvel_buf: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    angle_lim = 30  # degrees
    pos_lim = 0.025
    n_contact_lim = 2
    r_min, r_max = -1.0, 1.0

    # Distance from the pivot point to the object base
    dist_from_pivot = torch.norm(obj_base_pos - pivot_point_pos, p=2, dim=-1)

    # Calculate orientation distance from starting orientation.
    axel_cos_sim = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axel_degree_dist = torch.acos(axel_cos_sim) * (180 / torch.pi)

    # Angular velocity reward
    obj_angvel_about_axis = torch.sum(object_angvel * current_pivot_axel, dim=1)
    angvel_buf += obj_angvel_about_axis

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1) * -action_penalty_scale

    # apply reward conditionals
    total_reward = torch.clamp(obj_angvel_about_axis, min=r_min, max=r_max) * vel_reward_scale
    total_reward = torch.where(n_tip_contacts < n_contact_lim, torch.zeros_like(rew_buf), total_reward)
    total_reward = torch.where(axel_degree_dist > angle_lim, torch.zeros_like(rew_buf), total_reward)
    total_reward = torch.where(dist_from_pivot > pos_lim, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(dist_from_pivot >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'dist_from_pivot': dist_from_pivot,
        'axel_degree_dist': axel_degree_dist,
        'obj_angvel_about_axis': obj_angvel_about_axis,
        'action_penalty': action_penalty,
        'num_tip_contacts': n_tip_contacts,
        'total_reward': total_reward,
    }

    return total_reward, resets, angvel_buf, info


@torch.jit.script
def compute_manip_reward_dense(
                obj_base_pos: torch.Tensor,
                pivot_point_pos: torch.Tensor,
                target_pivot_axel: torch.Tensor,
                current_pivot_axel: torch.Tensor,
                object_angvel: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                require_contact: bool,
                pos_reward_scale: float,
                orn_reward_scale: float,
                vel_reward_scale: float,
                contact_reward_scale: float,
                action_penalty_scale: float,
                fall_penalty_scale: float,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
                angvel_buf: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    n_contact_lim = 2
    r_min, r_max = -1.0, 1.0

    # Distance from the pivot point to the object base
    dist_from_pivot = torch.norm(obj_base_pos - pivot_point_pos, p=2, dim=-1)

    # Calculate orientation distance from starting orientation.
    axel_cos_sim = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axel_cos_dist = torch.ones_like(axel_cos_sim) - axel_cos_sim
    axel_degree_dist = torch.acos(axel_cos_sim) * (180 / torch.pi)

    # Angular velocity reward
    obj_angvel_about_axis = torch.sum(object_angvel * target_pivot_axel, dim=1)

    # bound and scale rewards such that they are in similar ranges
    pos_dist_rew = lgsk_kernel(dist_from_pivot, scale=50., eps=2.) * pos_reward_scale
    # axel_dist_rew = lgsk_kernel(axel_cos_dist, scale=40., eps=2.) * orn_reward_scale
    axel_dist_rew = axel_cos_sim * orn_reward_scale
    angvel_rew = torch.clamp(obj_angvel_about_axis, min=r_min, max=r_max) * vel_reward_scale

    # add reward for maintaining tips in contact
    contact_rew = n_tip_contacts * contact_reward_scale

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1) * -action_penalty_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = pos_dist_rew + axel_dist_rew + angvel_rew + contact_rew + action_penalty

    # zero reward when less than 2 tips in contact
    total_reward = torch.where(n_tip_contacts < n_contact_lim, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(dist_from_pivot >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'dist_from_pivot': dist_from_pivot,
        'axel_degree_dist': axel_degree_dist,
        'obj_angvel_about_axis': obj_angvel_about_axis,
        'action_penalty': action_penalty,
        'num_tip_contacts': n_tip_contacts,
        'total_reward': total_reward,
    }

    return total_reward, resets, info


@torch.jit.script
def compute_manip_reward_simple(
                obj_base_pos: torch.Tensor,
                pivot_point_pos: torch.Tensor,
                target_pivot_axel: torch.Tensor,
                current_pivot_axel: torch.Tensor,
                object_angvel: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                require_contact: bool,
                pos_reward_scale: float,
                orn_reward_scale: float,
                vel_reward_scale: float,
                contact_reward_scale: float,
                action_penalty_scale: float,
                fall_penalty_scale: float,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
                angvel_buf: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    n_contact_lim = 2
    r_min, r_max = -1.0, 1.0

    # Distance from the pivot point to the object base
    dist_from_pivot = torch.norm(obj_base_pos - pivot_point_pos, p=2, dim=-1)

    # Calculate orientation distance from starting orientation.
    axel_cos_sim = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axel_degree_dist = torch.acos(axel_cos_sim) * (180 / torch.pi)

    # bound and scale rewards such that they are in similar ranges
    pos_dist_rew = lgsk_kernel(dist_from_pivot, scale=50., eps=2.) * pos_reward_scale

    angvel_penalty = -1 * torch.clamp(torch.abs(object_angvel[:, 0]), min=r_min, max=r_max) + \
        -1 * torch.clamp(torch.abs(object_angvel[:, 1]), min=r_min, max=r_max)

    angvel_reward = torch.clamp(object_angvel[:, 2], min=r_min, max=r_max) * 2.0

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1) * -action_penalty_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = pos_dist_rew + angvel_reward + angvel_penalty + action_penalty

    # zero reward when less than 2 tips in contact
    total_reward = torch.where(n_tip_contacts < n_contact_lim, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(dist_from_pivot >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'dist_from_pivot': dist_from_pivot,
        'axel_degree_dist': axel_degree_dist,
        'object_angvel_x': object_angvel[:, 0],
        'object_angvel_y': object_angvel[:, 1],
        'object_angvel_z': object_angvel[:, 2],
        'action_penalty': action_penalty,
        'num_tip_contacts': n_tip_contacts,
        'total_reward': total_reward,
    }

    return total_reward, resets, info


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # roll = roll % (2*np.pi)
    # pitch = pitch % (2*np.pi)
    # yaw = yaw % (2*np.pi)

    return torch.cat((torch.unsqueeze(roll, 1), torch.unsqueeze(pitch, 1), torch.unsqueeze(yaw, 1)), dim=1)


@torch.jit.script
def compute_manip_reward_conditional(
                obj_base_pos: torch.Tensor,
                pivot_point_pos: torch.Tensor,
                target_pivot_axel: torch.Tensor,
                current_pivot_axel: torch.Tensor,
                object_angvel: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                require_contact: bool,
                pos_reward_scale: float,
                orn_reward_scale: float,
                vel_reward_scale: float,
                contact_reward_scale: float,
                action_penalty_scale: float,
                fall_penalty_scale: float,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
                angvel_buf: torch.Tensor,
                curr_object_state: torch.Tensor,
                prev_object_state: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    angle_lim = 30  # degrees
    pos_lim = 0.025
    n_contact_lim = 1
    r_min, r_max = -1.0, 1.0

    # Distance from the pivot point to the object base
    dist_from_pivot = torch.norm(obj_base_pos - pivot_point_pos, p=2, dim=-1)

    # Calculate orientation distance from starting orientation.
    axel_cos_sim = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axel_degree_sim = torch.acos(axel_cos_sim) * (180 / torch.pi)

    # Angular velocity reward
    obj_angvel_about_axis = torch.sum(object_angvel * current_pivot_axel, dim=1)
    angvel_buf += obj_angvel_about_axis
    # print(angvel_buf)

    # aproximata angvel from pose delta instead of using angvel from sim
    # curr_obj_quat = curr_object_state[:, 3:7]
    # prev_obj_quat = prev_object_state[:, 3:7]
    # curr_obj_eul = get_euler_xyz(curr_obj_quat)
    # prev_obj_eul = get_euler_xyz(prev_obj_quat)
    #
    # deltas = (curr_obj_eul - prev_obj_eul) / 0.0167
    #
    # print(deltas)

    # wx = deltas[:, 0] + 0 - deltas[:, 2] * torch.sin(curr_obj_eul[:, 1])
    # wy = 0 + deltas[:, 1] * torch.sin(curr_obj_eul[:, 0]) + deltas[:, 2] * \
    #     torch.sin(curr_obj_eul[:, 0]) * torch.cos(curr_obj_eul[:, 1])
    # wz = 0 - deltas[:, 1] * torch.sin(curr_obj_eul[:, 0]) + deltas[:, 2] * \
    #     torch.cos(curr_obj_eul[:, 0]) * torch.cos(curr_obj_eul[:, 1])
    # est_angvel = torch.cat(
    #     (torch.unsqueeze(wx, 1), torch.unsqueeze(wy, 1), torch.unsqueeze(wz, 1)),
    #     dim=1)

    # obj_angvel_about_axis = torch.sum(deltas * current_pivot_axel, dim=1)

    # print(torch.sum(curr_obj_quat * prev_obj_quat, dim=1))
    # diff_quat = curr_obj_quat - prev_obj_quat
    # vel_quat = ((diff_quat * 2) / 0.0167) * quat_conjugate(prev_obj_quat)

    # print(torch.sum(vel_eul * current_pivot_axel, dim=1))

    # approx_angvel = 2 * quat_diff(curr_obj_quat, prev_obj_quat)
    # sim_angvel = curr_object_state[:, 10:13]

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1) * -action_penalty_scale

    # apply reward conditionals
    total_reward = torch.clamp(obj_angvel_about_axis, min=r_min, max=r_max)
    # total_reward = torch.clamp(approx_angvel, min=r_min, max=r_max)
    total_reward = torch.where(n_tip_contacts < n_contact_lim, torch.zeros_like(rew_buf), total_reward)
    total_reward = torch.where(axel_degree_sim > angle_lim, torch.zeros_like(rew_buf), total_reward)
    total_reward = torch.where(dist_from_pivot > pos_lim, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(dist_from_pivot >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'position_rew': dist_from_pivot,
        'axel_orientation_rew': axel_degree_sim,
        'angvel_rew': obj_angvel_about_axis,
        'action_penalty': action_penalty,
        'total_reward': total_reward,
    }

    return total_reward, resets, angvel_buf, info


@torch.jit.script
def compute_manip_reward_dense(
                obj_base_pos: torch.Tensor,
                pivot_point_pos: torch.Tensor,
                target_pivot_axel: torch.Tensor,
                current_pivot_axel: torch.Tensor,
                object_angvel: torch.Tensor,
                actions: torch.Tensor,
                max_episode_length: float,
                fall_reset_dist: float,
                n_tip_contacts: torch.Tensor,
                require_contact: bool,
                pos_reward_scale: float,
                orn_reward_scale: float,
                vel_reward_scale: float,
                contact_reward_scale: float,
                action_penalty_scale: float,
                fall_penalty_scale: float,
                rew_buf: torch.Tensor,
                reset_buf: torch.Tensor,
                progress_buf: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance from the pivot point to the object base
    dist_from_pivot = torch.norm(obj_base_pos - pivot_point_pos, p=2, dim=-1)

    # Calculate orientation distance from starting orientation.
    axel_cos_sim = torch.nn.functional.cosine_similarity(target_pivot_axel, current_pivot_axel, dim=1, eps=1e-12)
    axel_cos_dist = torch.ones_like(axel_cos_sim) - axel_cos_sim

    # Angular velocity reward
    obj_angvel_about_axis = torch.sum(object_angvel * current_pivot_axel, dim=1)

    # bound and scale rewards such that they are in similar ranges
    pos_dist_rew = lgsk_kernel(dist_from_pivot, scale=30., eps=2.) * pos_reward_scale
    axel_dist_rew = lgsk_kernel(axel_cos_dist, scale=30., eps=2.) * orn_reward_scale
    angvel_rew = torch.clamp(obj_angvel_about_axis, min=-1.0, max=1.0) * vel_reward_scale

    # add reward for maintaining tips in contact
    contact_rew = n_tip_contacts * contact_reward_scale

    # add penalty for large actions
    action_penalty = torch.sum(actions ** 2, dim=-1) * -action_penalty_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = pos_dist_rew + axel_dist_rew + angvel_rew + contact_rew + action_penalty

    # zero reward when less than 2 tips in contact
    if require_contact:
        total_reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(dist_from_pivot >= fall_reset_dist, total_reward + fall_penalty_scale, total_reward)

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(dist_from_pivot >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    info: Dict[str, torch.Tensor] = {
        'position_rew': pos_dist_rew,
        'axel_orientation_rew': axel_dist_rew,
        'angvel_rew': angvel_rew,
        'contact_rew': contact_rew,
        'action_penalty': action_penalty,
        'total_reward': total_reward,
    }

    return total_reward, resets, info
