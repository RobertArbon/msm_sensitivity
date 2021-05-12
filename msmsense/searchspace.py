import numpy as np
#
# EXPERIMENT PARAMETERS
#
NAME = '1FME'
INPUT_TRAJ_GLOB = 'Volumes/REA/MD/12FF/DESRES-Trajectory_1FME-*-protein/1FME-*-protein'
NUM_TRIALS = 4
BS_SAMPLES = 1

#
# HYPER-PARAMETERS
# MDTraj deals with nm not angstroms!
#
HP_SPACE = {'feature__value': ['dihedrals', 'distances'],
            'tica__dim': [1, 10], 'tica__lag': [1, 100],
            'cluster__k': [10, 1000], 'cluster__max_iter': [1000],
            'dihedrals__which': ['all'],
            'distances__transform': [ 'logistic', 'linear'],
            'distances__scheme': ['closest-heavy'],
            'distances__steepness': [0.1, 50],
            'distances__centre': [0.3, 1.5]}
# UNITS = {k: 1 for k in HP_SPACE.keys()}
# UNITS['distances__centre'] = 0.01
# UNITS['distances__steepness'] = 0.1

#
# DATA PARAMETERS
#
STRIDE = 1

#
# MSM PARAMETERS
#
LAGS = np.arange(2, 51, 2)
MSM_LAG = 10