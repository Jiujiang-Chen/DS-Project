from typing import Tuple, Dict
import torch

from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_conjugate

from smg_gym.utils.torch_jit_utils import lgsk_kernel


@torch.jit.script
def compute_hybrid_reorient_reward(
    rew_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    obj_base_pos: torch.Tensor,
    obj_base_orn: torch.Tensor,
    targ_base_pos: torch.Tensor,
    targ_base_orn: torch.Tensor,
    actions: torch.Tensor,
    n_tip_contacts: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    success_tolerance: float,
    max_episode_length: float,
    fall_reset_dist: float,
    require_contact: bool,
    contact_reward_scale: float,
    action_penalty_scale: float,
    reach_goal_bonus: float,
    fall_penalty: float,
    av_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance from the hand to the object
    goal_dist = torch.norm(obj_base_pos - targ_base_pos, p=2, dim=-1)

    # Orientation alignment for the object in hand and goal object
    quat_diff = quat_mul(obj_base_orn, quat_conjugate(targ_base_orn))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # calc dist and orn rew
    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # add penalty for large actions
    action_penalty = -torch.sum(actions**2, dim=-1) * action_penalty_scale

    # add reward for maintaining tips in contact
    contact_rew = n_tip_contacts * contact_reward_scale

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = dist_rew + rot_rew + action_penalty + contact_rew

    # zero reward when less than 2 tips in contact
    if require_contact:
        total_reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), total_reward)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    total_reward = torch.where(torch.abs(rot_dist) <= success_tolerance, total_reward + reach_goal_bonus, total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(goal_dist >= fall_reset_dist, total_reward + fall_penalty, total_reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(goal_dist >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # find average consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    info: Dict[str, torch.Tensor] = {
        'successes': successes,
        'cons_successes': cons_successes,
        'action_penalty': action_penalty,
        'num_tip_contacts': n_tip_contacts,
        'total_reward': total_reward,
    }

    return total_reward, resets, goal_resets, successes, cons_successes, info


@torch.jit.script
def compute_keypoint_reorient_reward(
    rew_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    obj_kps: torch.Tensor,
    goal_kps: torch.Tensor,
    obj_linvel: torch.Tensor,
    obj_angvel: torch.Tensor,
    actions: torch.Tensor,
    n_tip_contacts: torch.Tensor,
    n_non_tip_contacts: torch.Tensor,
    current_joint_pos: torch.Tensor,
    init_joint_pos: torch.Tensor,
    lgsk_scale: float,
    lgsk_eps: float,
    kp_dist_scale: float,
    hand_pose_penalty_scale: float,
    obj_linvel_penalty_scale: float,
    success_tolerance: float,
    max_episode_length: float,
    fall_reset_dist: float,
    require_contact: bool,
    contact_reward_scale: float,
    bad_contact_penalty_scale: float,
    action_penalty_scale: float,
    reach_goal_bonus: float,
    fall_penalty: float,
    av_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    # Distance from the pivot point to the object base
    kp_deltas = torch.norm(obj_kps - goal_kps, p=2, dim=-1)
    min_kp_dist, _ = kp_deltas.min(dim=-1)
    # max_kp_dist, _ = kp_deltas.max(dim=-1)
    mean_kp_dist = kp_deltas.mean(dim=-1)

    # bound and scale rewards such that they are in similar ranges
    kp_dist_rew = lgsk_kernel(kp_deltas, scale=lgsk_scale, eps=lgsk_eps).mean(dim=-1) * kp_dist_scale

    # add penalty for large actions
    action_penalty = -torch.sum(actions**2, dim=-1) * action_penalty_scale

    # add penalty for large actions
    hand_pose_penalty = -hand_pose_penalty_scale * torch.norm(current_joint_pos - init_joint_pos, p=2, dim=-1)

    # add penalty for linear velocity of obj
    obj_linvel_penalty = -obj_linvel_penalty_scale * torch.norm(obj_linvel, p=2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    total_reward = kp_dist_rew + action_penalty + hand_pose_penalty + obj_linvel_penalty

    # add reward for maintaining tips in contact
    total_reward = torch.where(n_tip_contacts <= 2, total_reward, total_reward + contact_reward_scale)

    # add penalty for contacting with links other than the tips
    total_reward = torch.where(n_non_tip_contacts > 0, total_reward - bad_contact_penalty_scale, total_reward)

    # zero reward when less than 2 tips in contact
    if require_contact:
        total_reward = torch.where(n_tip_contacts < 2, torch.zeros_like(rew_buf), total_reward)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    total_reward = torch.where(mean_kp_dist <= success_tolerance, total_reward + reach_goal_bonus, total_reward)

    # Fall penalty: distance to the goal is larger than a threashold
    total_reward = torch.where(min_kp_dist >= fall_reset_dist, total_reward + fall_penalty, total_reward)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(mean_kp_dist <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Check env termination conditions, including maximum success number
    resets = torch.zeros_like(reset_buf)
    resets = torch.where(min_kp_dist >= fall_reset_dist, torch.ones_like(reset_buf), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # find average consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    info: Dict[str, torch.Tensor] = {
        'successes': successes,
        'cons_successes': cons_successes,
        'action_penalty': action_penalty,
        'num_tip_contacts': n_tip_contacts,
        'num_non_tip_contacts': n_non_tip_contacts,
        'total_reward': total_reward,
        'penalty_hand_pose': hand_pose_penalty,
        'penalty_obj_linvel': obj_linvel_penalty,
    }

    return total_reward, resets, goal_resets, successes, cons_successes, info
