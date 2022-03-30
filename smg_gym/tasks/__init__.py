from smg_gym.tasks.reorient.smg_reorient import SMGReorient
from smg_gym.tasks.reorient.smg_rotate import SMGRotate
from smg_gym.tasks.reorient.smg_pivot import SMGPivot
from smg_gym.tasks.gaiting.smg_gaiting import SMGGaiting

# Mappings from strings to environments
task_map = {
    "smg_reorient": SMGReorient,
    "smg_rotate": SMGRotate,
    "smg_pivot": SMGPivot,
    "smg_gaiting": SMGGaiting,
}
