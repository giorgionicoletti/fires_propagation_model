import numpy as np
import time
from numba import njit
from numba import prange

import lib.fun_2D

@njit
def run_simulation(propensity_matrix, mat, nn_mat, change_matrix, nsteps, tmax,
                   dF, bF, lF, dV, bV, lV, save_every = 0.05):
    '''
    -----------------------------------------------------------------
    Arguments:  - propensity matrix, LxLx6 array
                - 2D lattice, LxL numpy array
                - neighbors matrix, LxLx4x2 numpy array
                - number of Gillespie steps
                - 6 parameters of the models, floats
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    Order of the reactions:

    Reaction 0:     F --(dF)--> 0
    Reaction 1:     V --(bF)--> F
    Reaction 2: F + V --(lF)--> F + F
    Reaction 3:     V --(dV)--> 0
    Reaction 4:     0 --(bV)--> V
    Reaction 5: 0 + V --(lV)--> V + V

    Initialize the lists to be returned: the densities, the states
    list and the time. Create a flag if there is an absorbing conf
    in the system.
    -----------------------------------------------------------------
    '''

    L = mat.shape[0]
    nreactions = np.int8(6)
    reaction_update = np.array([0, 1, 1, 0, 2, 2])

    if bV == 0:
        absorbing = True
    else:
        absorbing = False
    reached_abs_state = False
    fire_ended = False

    density_fires = []
    density_vegetation = []
    time = [0]

    density_fires.append(lib.fun_2D.find_density(mat, 1))
    density_vegetation.append(lib.fun_2D.find_density(mat, 2))

    '''
    -----------------------------------------------------------------
    Start the simulation loop. Use the Gillespie update.
    The function exits after the max number of steps is reached or
    if the system has reached the absorbing state.

    The propensity matrix is updated by updating the rows that
    correspond to the updated node and its neighbors.
    -----------------------------------------------------------------
    '''

    for idx_loop in range(nsteps):
        r1 = np.random.rand()
        r2 = np.random.rand()

        cumsum = np.cumsum(propensity_matrix)
        a0 = cumsum[-1]

        tau = - 1/a0 * np.log(r1)
        idx = np.searchsorted(cumsum, r2*a0, side = 'right')

        tau, node, reaction = tau, idx//nreactions, idx%nreactions
        node = (int(node/L), int(node%L))

        old_state = mat[node]
        new_state = reaction_update[reaction]

        mat[node] = reaction_update[reaction]

        propensity_matrix[node[0], node[1]] = lib.fun_2D.node_propensity(mat, nn_mat, node, dF, bF, lF, dV, bV, lV)

        for idx in nn_mat[node[0], node[1]]:
            propensity_matrix[idx[0], idx[1]] = propensity_matrix[idx[0], idx[1]] + change_matrix[mat[idx[0], idx[1]],
                                                                                                  old_state,
                                                                                                  mat[node]]

        time.append(time[-1] + tau)
        density_fires.append(lib.fun_2D.find_density(mat, 1))
        density_vegetation.append(lib.fun_2D.find_density(mat, 2))

        if density_fires[-1] == 0:
            fire_ended = True

        if density_fires[-1] + density_vegetation[-1] == 0:
            reached_abs_state = True

        if fire_ended or reached_abs_state:
            return time, density_fires, density_vegetation

    return time, density_fires, density_vegetation
