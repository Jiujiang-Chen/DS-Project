from smg_gym.tasks.reorient.smg_manip import SMGManip
from smg_gym.tasks.reorient.smg_rotate import SMGRotate
from smg_gym.tasks.reorient.smg_pivot import SMGPivot
from smg_gym.tasks.gaiting.smg_gaiting import SMGGaiting

# Mappings from strings to environments
task_map = {
    "smg_manip": SMGManip,
    "smg_rotate": SMGRotate,
    "smg_pivot": SMGPivot,
    "smg_gaiting": SMGGaiting,
}