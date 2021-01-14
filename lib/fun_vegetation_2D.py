import numpy as np
from numba import njit
import lib.fun_2D

@njit
def node_propensity(mat, nn_mat, node, dV, bV, lV):
    '''
    -----------------------------------------------------------------
    Arguments:  - 2D lattice, LxL numpy array
                - neighbors matrix, LxLx4x2 numpy array
                - node, 2x1 numpy array
                - 6 parameters of the models, floats

    Returns the new propensity rates fore the given node, considering
    its state and the ones of the neighbors
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    Order of the reactions:

    Reaction 0:     V --(dV)--> 0
    Reaction 1:     0 --(bV)--> V
    Reaction 2: 0 + V --(lV)--> V + V
    -----------------------------------------------------------------
    '''
    state = mat[node[0], node[1]]
    nn_list = nn_mat[node[0], node[1]]

    if state == 0:
        nnV = lib.fun_2D.sum_nn(mat, nn_list, 1)
        res = np.array([0, bV, nnV*lV/4], dtype = np.float64)

    elif state == 1:
        res = np.array([dV, 0, 0], dtype = np.float64)

    return res


@njit
def run_simulation(propensity_matrix, mat, nn_mat, change_matrix, nsteps, tmax,
                   dV, bV, lV):
    '''
    -----------------------------------------------------------------
    Arguments:  - propensity matrix, LxLx3 array
                - 2D lattice, LxL numpy array
                - neighbors matrix, LxLx4x2 numpy array
                - change matrix, 3x3x3x6 numpy array
                - number of Gillespie steps
                - maximum time
                - 3 parameters of the models, floats
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    Order of the reactions:

    Reaction 0:     V --(dV)--> 0
    Reaction 1:     0 --(bV)--> V
    Reaction 2: 0 + V --(lV)--> V + V

    Initialize the lists to be returned: the densities, the states
    list and the time. Create a flag if there is an absorbing conf
    in the system.
    -----------------------------------------------------------------
    '''

    L = mat.shape[0]
    nreactions = np.int8(3)
    reaction_update = np.array([0, 1, 1])

    if bV == 0:
        absorbing = True
    else:
        absorbing = False

    density_vegetation = []
    time = [0]

    density_vegetation.append(lib.fun_2D.find_density(mat, 1))

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

        propensity_matrix[node[0], node[1]] = node_propensity(mat, nn_mat, node, dV, bV, lV)

        for idx in nn_mat[node[0], node[1]]:
            propensity_matrix[idx[0], idx[1]] = propensity_matrix[idx[0], idx[1]] + change_matrix[mat[idx[0], idx[1]], old_state, mat[node]]

        time.append(time[-1] + tau)
        density_vegetation.append(lib.fun_2D.find_density(mat, 1))

        if density_vegetation[-1] == 0:
            if absorbing:
                return mat, propensity_matrix

    return mat, propensity_matrix
