"""
python check_joint_trajectories.py task=smg_debug headless=false
"""

# need to import isaacgym before torch
import isaacgym

import pandas as pd
from ast import literal_eval
import numpy as np
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver
from isaacgymenvs.utils.utils import set_seed

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from smg_gym.utils.rlgames_utils import get_rlgames_env_creator

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


@hydra.main(config_name="config", config_path="./cfg")
def launch_trajectory_checker(cfg: DictConfig):

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
    # We use the helper function here to specify the environment config.
    create_rlgpu_env = get_rlgames_env_creator(
        omegaconf_to_dict(cfg.task),
        cfg.task_name,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        multi_gpu=cfg.multi_gpu,
    )

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # create an agent and restore parameters
    player = runner.create_player()

    simple_run_agent(player)


def simple_run_agent(player):

    # compare_to = 'real'
    compare_to = 'gazebo'

    data_dir = '/home/alex/Documents/smg_real/smg_real/utils/proprio_data'
    ref_df_name = os.path.join(
        data_dir,
        f'{compare_to}_proprio.csv'
    )
    ref_df = pd.read_csv(ref_df_name)
    ref_df['actions'] = ref_df['actions'].apply(lambda x: literal_eval(x))
    max_steps = len(ref_df)

    df = pd.DataFrame(
        columns=['actions', 'joint_pos', 'joint_vel', 'joint_eff', 'fingertip_pos', 'fingertip_orn']
    )
    row_counter = 0

    player.env_reset(player.env)

    for n in range(max_steps):
        actions = np.array(ref_df.loc[row_counter]['actions'])[np.newaxis, ...]

        obses, r, done, info = player.env_step(player.env, actions)

        df.loc[row_counter] = [
            list(actions[0, :]),
            list(obses[0, 0:9].cpu().numpy()),
            list(obses[0, 9:18].cpu().numpy()),
            list(obses[0, 18:27].cpu().numpy()),
            list(obses[0, 27:36].cpu().numpy()),
            list(obses[0, 36:48].cpu().numpy()),
        ]
        row_counter += 1

    csv_filename = os.path.join(
        data_dir,
        'isaacgym_proprio.csv'
    )
    print(f'Saving Data to {csv_filename}')
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    launch_trajectory_checker()
