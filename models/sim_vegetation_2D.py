import numpy as np
from numba import njit
from numba import prange
from scipy import ndimage

import lib.fun_2D
import lib.fun_vegetation_2D

@njit(parallel = True)
def find_configurations(L, dV, lV, bV = 0, nconf = 1000, nsteps_init = int(1e6), nsteps_sample = int(1e5)):
    '''
    -----------------------------------------------------------------
    Arguments:  - L, int
                - 3 parameters of the models, floats
                - nconf, int
                - nsteps_init, int
                - nsteps_sample, int
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function is used to generate in parallel a given number of
    vegetation configurations. We first initialize the system at the
    stationary state, then from this stationary configuration we run
    in parallel different simulations.
    After a large enough number of steps, so the system has forgotten
    the starting configuration, we save the configuration.
    -----------------------------------------------------------------
    '''
    states_list = np.zeros((nconf, L, L), dtype = np.int8)

    change_matrix = np.zeros((2,2,2,3), dtype = np.float64)
    change_matrix[0,0,1] = np.array([0, 0, +lV/4], dtype = np.float64)
    change_matrix[0,1,0] = np.array([0, 0, -lV/4], dtype = np.float64)

    nn_mat = lib.fun_2D.return_nn(np.zeros((L,L)))

    propensity_matrix = np.zeros((L, L, 3), dtype = np.float64)

    '''
    -----------------------------------------------------------------
    Initialize the system at the stationary conf
    -----------------------------------------------------------------
    '''

    tmax = 100
    mat = np.random.choice(np.array([0,1], dtype = np.int8), size = (L,L))

    for idx, _ in np.ndenumerate(mat):
        propensity_matrix[idx] = lib.fun_vegetation_2D.node_propensity(mat, nn_mat, idx, dV, bV, lV)

    ini_mat, propensity_matrix = lib.fun_vegetation_2D.run_simulation(propensity_matrix, mat.copy(),
                                                                      nn_mat, change_matrix,
                                                                      nsteps_init,
                                                                      tmax, dV, bV, lV)

    '''
    -----------------------------------------------------------------
    Run in parallel different simulations from the stationary conf
    -----------------------------------------------------------------
    '''

    for idx in prange(nconf):
        mat, _ = lib.fun_vegetation_2D.run_simulation(propensity_matrix.copy(), ini_mat.copy(),
                                                      nn_mat, change_matrix,
                                                      nsteps_sample,
                                                      tmax, dV, bV, lV)

        states_list[idx] = mat.copy()

    return states_list


def create_labeled_states(states_list):
    labeled_states_list = np.zeros(states_list.shape)

    for idx_state, mat in enumerate(states_list):
        label_image, _ = ndimage.measurements.label(mat)

        for y in range(label_image.shape[0]):
            if label_image[y, 0] > 0 and label_image[y, -1] > 0:
                label_image[label_image == label_image[y, -1]] = label_image[y, 0]

        for x in range(label_image.shape[1]):
            if label_image[0, x] > 0 and label_image[-1, x] > 0:
                label_image[label_image == label_image[-1, x]] = label_image[0, x]

        n = np.unique(label_image).size

        for old_label, new_label in zip(np.unique(label_image), np.arange(n)):
            label_image[label_image == old_label] = new_label

        labeled_states_list[idx_state] = label_image.copy()

    return labeled_states_list
