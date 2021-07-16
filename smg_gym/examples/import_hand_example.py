import os
import numpy as np
import random
from isaacgym import gymutil
from isaacgym import gymapi
import inspect


from smg_gym.assets import get_assets_path, add_assets_path

from pybullet_object_models import primitive_objects as object_set

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Import Objects Example")

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.always_use_articulations = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    args.physics_engine,
    sim_params
)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Q, "quit")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "esc")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "toggle_viewer_sync")

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 0.0
plane_params.dynamic_friction = 0.0
plane_params.restitution = 0

gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 4
grid_size = int(np.sqrt(num_envs))
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# create ball asset with gravity disabled from pybullet-object_models
def load_hand():

    asset_root = add_assets_path('robot_assets/smg')
    asset_file =  "smg_tactip.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.collapse_fixed_joints = False
    asset_options.armature = 0.00001
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    asset_options.convex_decomposition_from_submeshes = False
    asset_options.vhacd_enabled = False
    asset_options.flip_visual_attachments = False

    hand_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    return hand_asset

def load_objects():

    model_list = object_set.getModelList()
    asset_root = object_set.getDataPath()

    object_assets = []
    for object_name in model_list:
        asset_file = os.path.join(object_name, "model.urdf")
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        obj_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

        object_assets.append(obj_asset)

    return object_assets


control_joint_names = [
    "SMG_F1J1", "SMG_F1J2", "SMG_F1J3",
    "SMG_F2J1", "SMG_F2J2", "SMG_F2J3",
    "SMG_F3J1", "SMG_F3J2", "SMG_F3J3"
]

num_control_dofs = len(control_joint_names)

mins = {
    "J1": -20.0*(np.pi/180),
    "J2": -25.0*(np.pi/180),
    "J3": -20.0*(np.pi/180),
}
maxs = {
    "J1": 20.0*(np.pi/180),
    "J2": -15.0*(np.pi/180),
    "J3": 20.0*(np.pi/180),
}

def init_hand_joints(env, actor_handle):

    init_joint_pos = {}
    for control_joint in control_joint_names:

        # get rand state for init joint pos
        joint_num = control_joint[-2:]
        rand_pos = np.random.uniform(mins[joint_num], maxs[joint_num])
        init_joint_pos[control_joint] = rand_pos

        # set this rand pos as the target
        dof_handle = gym.find_actor_dof_handle(env, actor_handle, control_joint)
        gym.set_dof_target_position(env, dof_handle, rand_pos)
        gym.set_dof_target_velocity(env, dof_handle, 0.0)

    # hard reset to random position
    gym.set_actor_dof_states(env, actor_handle, list(init_joint_pos.values()), gymapi.STATE_POS)
    gym.set_actor_dof_states(env, actor_handle, [0.0]*num_control_dofs, gymapi.STATE_VEL)

    return init_joint_pos

def add_hand_actor(env):

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.025)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    num_hand_bodies = gym.get_asset_rigid_body_count(hand_asset)
    num_hand_shapes = gym.get_asset_rigid_shape_count(hand_asset)
    n_hand_dofs = gym.get_asset_dof_count(hand_asset)

    gym.begin_aggregate(env, num_hand_bodies, num_hand_shapes, False)

    handle = gym.create_actor(env, hand_asset, pose, "hand_actor_{}".format(i), -1, -1)

    # Configure DOF properties
    props = gym.get_actor_dof_properties(env, handle)
    props["driveMode"] = [gymapi.DOF_MODE_POS]*n_hand_dofs
    props["stiffness"] = [5000.0]*n_hand_dofs
    props["damping"] = [100.0]*n_hand_dofs
    gym.set_actor_dof_properties(env, handle, props)

    # create actor handles
    control_handles = {}
    for control_joint in control_joint_names:
        dof_handle = gym.find_actor_dof_handle(env, handle, control_joint)
        control_handles[control_joint] = dof_handle

    init_joint_pos = init_hand_joints(env, handle)

    gym.end_aggregate(env)

    return handle, control_handles, init_joint_pos

def add_object_actor(env):
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.245)
    pose.r = gymapi.Quat(0, 0, 0, 1)

    object_asset = np.random.choice(object_assets)
    handle = gym.create_actor(env, object_asset, pose, "obj_actor_{}".format(i), -1, -1)

    obj_props = gym.get_actor_rigid_body_properties(env, handle)
    obj_props[0].mass = 1.0
    gym.set_actor_rigid_body_properties(env, handle, obj_props)

    # set shape properties (some settings only work on FLEX backend)
    shape_props = gym.get_actor_rigid_shape_properties(env, handle)
    gym.set_actor_rigid_shape_properties(env, handle, shape_props)

    return handle, object_asset


def get_object_state(env, obj_actor_handle):

    obj_state = gym.get_actor_rigid_body_states(env, obj_actor_handle, gymapi.STATE_ALL)
    pos = obj_state['pose']['p']
    orn = obj_state['pose']['r']
    lin_vel = obj_state['vel']['linear']
    ang_vel = obj_state['vel']['angular']

    return pos, orn, lin_vel, ang_vel

def apply_gravity_compensation_object(env, obj_actor_handle):

    obj_props = gym.get_actor_rigid_body_properties(env, obj_actor_handle)
    mass = obj_props[0].mass
    gravity = sim_params.gravity
    force = -(gravity * mass)

    body_names = gym.get_actor_rigid_body_names(env, obj_actor_handle)
    rigid_body_handle = gym.find_actor_rigid_body_handle(env, obj_actor_handle, body_names[0])
    gym.apply_body_forces(env, rigid_body_handle, force=force, torque=None, space=gymapi.ENV_SPACE)


def pre_physics_step():

    act_lim = 0.1
    dof_speed_scale = 10.0
    dt = sim_params.dt

    for i in range(num_envs):

        action = np.random.uniform(-act_lim, act_lim, size=[num_control_dofs])

        current_joint_state = current_joint_states[i]
        targets = current_joint_state + dof_speed_scale * dt * action

        for (j, dof_handle) in enumerate(hand_control_joint_handles[i].values()):
            gym.set_dof_target_position(envs[i], dof_handle, targets[j])

        current_joint_states[i] = targets

def post_physics_step():
    pos, orn, lin_vel, ang_vel = get_object_state(envs[i], object_actor_handles[i])

def apply_grasp_action(current_joint_states):

    act_lim = 0.1
    dof_speed_scale = 10.0
    dt = sim_params.dt

    grasp_action = np.array([
        0.0, act_lim, 0.0,
        0.0, act_lim, 0.0,
        0.0, act_lim, 0.0
    ])

    for i in range(num_envs):
        current_joint_state = current_joint_states[i]
        targets = current_joint_state + dof_speed_scale * dt * grasp_action

        for (j, dof_handle) in enumerate(hand_control_joint_handles[i].values()):
            gym.set_dof_target_position(envs[i], dof_handle, targets[j])

        current_joint_states[i] = targets


def get_tip_contacts(env, hand_actor_handle, obj_actor_handle):

    contacts = gym.get_env_rigid_contacts(env)
    # contact_forces = gym.get_env_rigid_contact_forces(env)

    obj_body_names = gym.get_actor_rigid_body_names(env, obj_actor_handle)
    obj_body_idx = gym.find_actor_rigid_body_index(env, obj_actor_handle, obj_body_names[0], gymapi.DOMAIN_ENV)

    hand_body_names = gym.get_actor_rigid_body_names(env, hand_actor_handle)
    tip_body_names = [name for name in hand_body_names if 'tactip_tip' in name]
    tip_body_idxs = [gym.find_actor_rigid_body_index(env, hand_actor_handle, name, gymapi.DOMAIN_ENV) for name in tip_body_names]

    tip_contacts = [False] * len(tip_body_idxs)

    # iterate through contacts and update tip_contacts list if there is a
    # contact between object and tip
    for contact in contacts:
        if obj_body_idx in [contact['body0'], contact['body1']]:
            current_tip_contacts = [tip_body_idx in [contact['body0'], contact['body1']] for tip_body_idx in tip_body_idxs]

            for i, current_tip_contact in enumerate(current_tip_contacts):
                if current_tip_contact:
                    tip_contacts[i] = True

    return tip_contacts


def initialise_contact(current_joint_states):
    max_steps = 50
    update_envs = list(range(num_envs))

    for i in range(max_steps):

        if update_envs == []: break

        for i in update_envs:
            apply_gravity_compensation_object(envs[i], object_actor_handles[i])
            apply_grasp_action(current_joint_states)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        for i in update_envs:
            tip_contacts = get_tip_contacts(envs[i], hand_actor_handles[i], object_actor_handles[i])

            # if all three tips establish contact then stop appying grasp action
            if sum(tip_contacts) == 3:
                update_envs.remove(i)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

def reset():
    gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # set rand init start state
    init_joint_poses = []
    for env, handle in zip(envs, hand_actor_handles):
        init_joint_pos = init_hand_joints(env, handle)
        init_joint_poses.append(init_joint_pos)

    current_joint_states = []
    for i in range(num_envs):
        current_joint_states.append(np.array(list(init_joint_poses[i].values())))

    initialise_contact(current_joint_states)

    return current_joint_states

# create hand asset
hand_asset = load_hand()
object_assets = load_objects()

# create list to mantain environment and asset handles
envs = []
hand_actor_handles = []
hand_control_joint_handles = []
init_joint_poses = []
object_actor_handles = []
object_asset_list = []
for i in range(num_envs):

    env = gym.create_env(sim, env_lower, env_upper, grid_size)

    hand_actor_handle, hand_control_handles, init_joint_pos = add_hand_actor(env)
    object_handle, object_asset = add_object_actor(env)

    envs.append(env)
    hand_actor_handles.append(hand_actor_handle)
    hand_control_joint_handles.append(hand_control_handles)
    init_joint_poses.append(init_joint_pos)
    object_actor_handles.append(object_handle)
    object_asset_list.append(object_asset)

# print('Envs: ', envs)
# print('Actors: ', hand_actor_handles)
# print('Joints: ', hand_control_joint_hanles)
# print('Init Joints: ', init_joint_poses)

# look at the first env
cam_pos = gymapi.Vec3(2, 2, 2)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# save initial state for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

colors = [gymapi.Vec3(1.0, 0.0, 0.0),
          gymapi.Vec3(1.0, 127.0/255.0, 0.0),
          gymapi.Vec3(1.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 1.0, 0.0),
          gymapi.Vec3(0.0, 0.0, 1.0),
          gymapi.Vec3(39.0/255.0, 0.0, 51.0/255.0),
          gymapi.Vec3(139.0/255.0, 0.0, 1.0)]

current_joint_states = reset()

# clear lines every n steps
clear_step = 25
step_counter = 0
enable_viewer_sync = True

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):

        if evt.action == "reset" and evt.value > 0:
            current_joint_states = reset()

        if ( (evt.action == "quit" and evt.value > 0) or
             (evt.action == "esc" and evt.value > 0) ):
            gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            quit()

        if evt.action == "toggle_viewer_sync" and evt.value > 0:
            enable_viewer_sync = not enable_viewer_sync

    # apply a step before simulating
    pre_physics_step()

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # apply a step after simulating
    post_physics_step()

    # update the viewer
    if enable_viewer_sync:

        # draw contacts for first env
        # gym.draw_env_rigid_contacts(viewer, envs[0], colors[0], 0.25, True)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

        # remove all lines drawn on viewer
        if step_counter % clear_step == 0:
            gym.clear_lines(viewer)

    else:
        gym.poll_viewer_events(viewer)

    step_counter += 1

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
