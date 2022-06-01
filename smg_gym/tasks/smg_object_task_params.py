import enum
from types import SimpleNamespace
import numpy as np


class SMGObjectTaskDimensions(enum.Enum):

    # cartesian position + quaternion orientation
    PosDim = 3
    OrnDim = 4
    PoseDim = PosDim + OrnDim

    # linear velocity + angular velcoity
    LinearVelocityDim = 3
    AngularVelocityDim = 3
    VelocityDim = LinearVelocityDim + AngularVelocityDim

    # state: pose + velocity
    StateDim = 13

    # force + torque
    ForceDim = 3
    TorqueDim = 3
    WrenchDim = ForceDim + TorqueDim

    # number of fingers
    NumFingers = 3
    FingertipPosDim = PosDim * NumFingers
    FingertipOrnDim = OrnDim * NumFingers
    FingerContactForceDim = ForceDim * NumFingers
    FingerContactTorqueDim = TorqueDim * NumFingers

    # for three fingers
    ActionDim = 9
    JointPositionDim = 9
    JointVelocityDim = 9
    JointTorqueDim = 9

    # for object
    NumKeypoints = 6
    KeypointPosDim = PosDim * NumKeypoints

    # for target
    VecDim = 3


dims = SMGObjectTaskDimensions

object_properties = {
    "sphere": {
        "radius": 0.035,
        "radius_llim": 0.03,
        "radius_ulim": 0.04,
    },
    "box": {
        "size": [0.075, 0.075, 0.075],
        "size_llims": [0.06, 0.06, 0.06],
        "size_ulims": [0.08, 0.08, 0.08],
    },
}

robot_dof_properties = {
    "max_position_delta_rad": 0.025,
    "max_velocity_radps": np.deg2rad(45.0),
    "max_torque_Nm": 0.4,
    "friction": [0.0, 0.0, 0.0] * dims.NumFingers.value,
}

# The kp and kd gains of the PD control of the fingers.
robot_dof_gains = {
    "p_gains": [0.6, 0.6, 0.2] * dims.NumFingers.value,
    "d_gains": [0.03, 0.01, 0.005] * dims.NumFingers.value,
}

# actuated joints on the hand
control_joint_names = [
    "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
    "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
    "SMG_F3J1", "SMG_F3J2", "SMG_F3J3",
]

# limits of the robot (mapped later: str -> torch.tensor)
robot_limits = {
    "joint_position": SimpleNamespace(
        # matches those on the real robot
        low=np.array([-0.785, -1.396, -1.047] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.785, 1.047, 1.396] * dims.NumFingers.value, dtype=np.float32),
        # default=np.deg2rad([0.0, 7.5, -10.0] * dims.NumFingers.value, dtype=np.float32),
        # rand_lolim=np.deg2rad([-20.0, 7.5, -10.0] * dims.NumFingers.value, dtype=np.float32),
        # rand_uplim=np.deg2rad([20.0, 7.5, -10.0] * dims.NumFingers.value, dtype=np.float32),
        default=np.array([-0.45, 0.35, -0.55, 0.0, 0.35, -0.55, 0.45, 0.35, -0.55], dtype=np.float32),
        rand_lolim=np.array([-0.15, 0.35, -0.55, -0.3, 0.35, -0.55, 0.15, 0.35, -0.55], dtype=np.float32),
        rand_uplim=np.array([-0.75, 0.35, -0.55, 0.3, 0.35, -0.55, 0.75, 0.35, -0.55], dtype=np.float32),
    ),
    "joint_velocity": SimpleNamespace(
        low=np.full(dims.JointVelocityDim.value, -robot_dof_properties["max_velocity_radps"], dtype=np.float32),
        high=np.full(dims.JointVelocityDim.value, robot_dof_properties["max_velocity_radps"], dtype=np.float32),
        default=np.zeros(dims.JointVelocityDim.value, dtype=np.float32),
    ),
    "joint_effort": SimpleNamespace(
        low=np.full(dims.JointTorqueDim.value, -robot_dof_properties["max_torque_Nm"], dtype=np.float32),
        high=np.full(dims.JointTorqueDim.value, robot_dof_properties["max_torque_Nm"], dtype=np.float32),
        default=np.zeros(dims.JointTorqueDim.value, dtype=np.float32),
    ),
    "fingertip_position": SimpleNamespace(
        low=np.array([-0.25, -0.25, -0.25] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.25, 0.25, 0.25] * dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_orientation": SimpleNamespace(
        low=-np.ones(4 * dims.NumFingers.value, dtype=np.float32),
        high=np.ones(4 * dims.NumFingers.value, dtype=np.float32),
    ),
    "fingertip_velocity": SimpleNamespace(
        low=np.full(dims.VelocityDim.value, -0.25, dtype=np.float32),
        high=np.full(dims.VelocityDim.value, 0.25, dtype=np.float32),
    ),
    "latest_action_pos": SimpleNamespace(
        low=np.full(dims.JointPositionDim.value, -robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
        high=np.full(dims.JointPositionDim.value, robot_dof_properties["max_position_delta_rad"], dtype=np.float32),
    ),
    "latest_action_vel": SimpleNamespace(
        low=np.full(dims.JointVelocityDim.value, -robot_dof_properties["max_velocity_radps"], dtype=np.float32),
        high=np.full(dims.JointVelocityDim.value, robot_dof_properties["max_velocity_radps"], dtype=np.float32),
    ),
    "latest_action_eff": SimpleNamespace(
        low=np.full(dims.JointTorqueDim.value, -robot_dof_properties["max_torque_Nm"], dtype=np.float32),
        high=np.full(dims.JointTorqueDim.value, robot_dof_properties["max_torque_Nm"], dtype=np.float32),
    ),
    "bool_tip_contacts": SimpleNamespace(
        low=np.zeros(dims.NumFingers.value, dtype=np.float32),
        high=np.ones(dims.NumFingers.value, dtype=np.float32),
    ),
    "net_tip_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "ft_sensor_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "ft_sensor_contact_torques": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_positions": SimpleNamespace(
        low=np.array([-0.02, -0.02, -0.02] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([0.02, 0.02, 0.02] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_normals": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * dims.NumFingers.value, dtype=np.float32),
    ),
    "tip_contact_force_mags": SimpleNamespace(
        low=np.array([0.0] * dims.NumFingers.value, dtype=np.float32),
        high=np.array([5.0] * dims.NumFingers.value, dtype=np.float32),
    ),
}

object_limits = {
    "position": SimpleNamespace(
        low=np.array([-0.2, -0.2, 0.0], dtype=np.float32),
        high=np.array([0.2, 0.2, 0.4], dtype=np.float32),
    ),
    "orientation": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "keypoint_position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0.0] * dims.NumKeypoints.value, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.5] * dims.NumKeypoints.value, dtype=np.float32),
    ),
    "linear_velocity": SimpleNamespace(
        low=np.array([-0.5, -0.5, -2.0], dtype=np.float32),
        high=np.array([0.5, 0.5, 0.2], dtype=np.float32),
    ),
    "angular_velocity": SimpleNamespace(
        low=np.full(dims.AngularVelocityDim.value, -1.0, dtype=np.float32),
        high=np.full(dims.AngularVelocityDim.value, 1.0, dtype=np.float32),
    ),
}

target_limits = {
    "position": SimpleNamespace(
        low=np.array([-0.2, -0.2, 0.0], dtype=np.float32),
        high=np.array([0.2, 0.2, 0.4], dtype=np.float32),
    ),
    "orientation": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "keypoint_position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0.0] * dims.NumKeypoints.value, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.5] * dims.NumKeypoints.value, dtype=np.float32),
    ),
    "active_quat": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "pivot_axel_vector": SimpleNamespace(
        low=np.full(dims.VecDim.value, -1.0, dtype=np.float32),
        high=np.full(dims.VecDim.value, 1.0, dtype=np.float32),
    ),
    "pivot_axel_position": SimpleNamespace(
        low=np.full(dims.PosDim.value, -0.05, dtype=np.float32),
        high=np.full(dims.PosDim.value, 0.05, dtype=np.float32),
    ),
}
