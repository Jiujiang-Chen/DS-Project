from smg_gym.tasks.reorient.smg_reorient import SMGReorient
from smg_gym.tasks.reorient.smg_rotate import SMGRotate
from smg_gym.tasks.reorient.smg_pivot import SMGPivot
from smg_gym.tasks.gaiting.smg_gaiting import SMGGaiting
from smg_gym.tasks.debug.smg_debug import SMGDebug

# Mappings from strings to environments
task_map = {
    "smg_reorient": SMGReorient,
    "smg_rotate": SMGRotate,
    "smg_pivot": SMGPivot,
    "smg_gaiting": SMGGaiting,
    "smg_debug": SMGDebug,
}
