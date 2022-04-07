from types import SimpleNamespace
import numpy as np


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

# for robot
# number of fingers
NumFingers = 3

# for three fingers
JointPositionDim = 9
JointVelocityDim = 9
JointTorqueDim = 9

# for object
NumKeypoints = 6
KeypointPositionDim = 3

# for target
VecDim = 3

# Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/trifinger_platform.py#L68
# maximum joint torque (in N-m) applicable on each actuator
_max_torque_Nm = 0.36

# maximum joint velocity (in rad/s) on each actuator
_max_velocity_radps = 10

# limits of the robot (mapped later: str -> torch.tensor)
robot_limits = {
    "joint_position": SimpleNamespace(
        # matches those on the real robot
        low=np.array([-0.785, -1.396, -1.047] * NumFingers, dtype=np.float32),
        high=np.array([0.785, 1.047, 1.57] * NumFingers, dtype=np.float32),
        default=np.array([0.0, 0.9, -1.7] * NumFingers, dtype=np.float32),
    ),
    "joint_velocity": SimpleNamespace(
        low=np.full(JointVelocityDim, -_max_velocity_radps, dtype=np.float32),
        high=np.full(JointVelocityDim, _max_velocity_radps, dtype=np.float32),
        default=np.zeros(JointVelocityDim, dtype=np.float32),
    ),
    "joint_torque": SimpleNamespace(
        low=np.full(JointTorqueDim, -_max_torque_Nm, dtype=np.float32),
        high=np.full(JointTorqueDim, _max_torque_Nm, dtype=np.float32),
        default=np.zeros(JointTorqueDim, dtype=np.float32),
    ),
    "fingertip_position": SimpleNamespace(
        low=np.array([-0.4, -0.4, 0] * NumFingers, dtype=np.float32),
        high=np.array([0.4, 0.4, 0.5] * NumFingers, dtype=np.float32),
    ),
    "fingertip_orientation": SimpleNamespace(
        low=-np.ones(4 * NumFingers, dtype=np.float32),
        high=np.ones(4 * NumFingers, dtype=np.float32),
    ),
    "fingertip_velocity": SimpleNamespace(
        low=np.full(VelocityDim, -0.2, dtype=np.float32),
        high=np.full(VelocityDim, 0.2, dtype=np.float32),
    ),
    "bool_tip_contacts": SimpleNamespace(
        low=np.zeros(NumFingers, dtype=np.float32),
        high=np.ones(NumFingers, dtype=np.float32),
    ),
    "tip_contact_forces": SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0] * NumFingers, dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0] * NumFingers, dtype=np.float32),
    ),
}

object_limits = {
    "position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0], dtype=np.float32),
        high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
    ),
    "orientation": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "keypoint_position": SimpleNamespace(
        low=np.array([-0.3, -0.3, 0] * NumKeypoints, dtype=np.float32),
        high=np.array([0.3, 0.3, 0.3] * NumKeypoints, dtype=np.float32),
    ),
    "linear_velocity": SimpleNamespace(
        low=np.full(LinearVelocityDim, -0.5, dtype=np.float32),
        high=np.full(LinearVelocityDim, 0.5, dtype=np.float32),
    ),
    "angular_velocity": SimpleNamespace(
        low=np.full(AngularVelocityDim, -0.5, dtype=np.float32),
        high=np.full(AngularVelocityDim, 0.5, dtype=np.float32),
    ),
}

target_limits = {
    "active_quat": SimpleNamespace(
        low=-np.ones(4, dtype=np.float32),
        high=np.ones(4, dtype=np.float32),
    ),
    "pivot_axel_vector": SimpleNamespace(
        low=np.full(VecDim, -1.0, dtype=np.float32),
        high=np.full(VecDim, 1.0, dtype=np.float32),
    ),
    "pivot_axel_position": SimpleNamespace(
        low=np.full(PosDim, -0.025, dtype=np.float32),
        high=np.full(PosDim, 0.025, dtype=np.float32),
    ),
}

robot_dof_gains = {
    # The kp and kd gains of the PD control of the fingers.
    # Note: This depends on simulation step size and is set for a rate of 250 Hz.
    "stiffness": [10.0, 10.0, 10.0] * NumFingers,
    "damping": [0.1, 0.3, 0.001] * NumFingers,
    # The kd gains used for damping the joint motor velocities during the
    # safety torque check on the joint motors.
    "safety_damping": [0.08, 0.08, 0.04] * NumFingers
}
