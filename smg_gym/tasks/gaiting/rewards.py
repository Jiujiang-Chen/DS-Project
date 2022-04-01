from typing import Tuple, Dict
import torch
from smg_gym.utils.torch_jit_utils import lgsk_kernel


@torch.jit.script
def compute_conditional_angvel_reward(
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
def compute_dense_angvel_reward(
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

    # Angular velocity reward
    obj_angvel_about_axis = torch.sum(object_angvel * target_pivot_axel, dim=1)

    # bound and scale rewards such that they are in similar ranges
    pos_dist_rew = lgsk_kernel(dist_from_pivot, scale=50., eps=2.) * pos_reward_scale
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
