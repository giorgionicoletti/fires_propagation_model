import numpy as np
from numba import njit

@njit
def return_nn(mat):
    '''
    -----------------------------------------------------------------
    Arguments:  - 2D lattice, LxL numpy array

    Creates the nearest neighbors matrix, a LxLx4x2 array so each
    site of the LxL sublattice is a 4x2 array that contains the
    indexes of the nns.
    -----------------------------------------------------------------
    '''
    L = mat.shape[0]
    nnlist = np.zeros((L, L, np.int64(4), np.int64(2)), dtype = np.int64)

    moves_list = [np.array([-1,0]), np.array([0,+1]),
                  np.array([+1,0]), np.array([0,-1])]

    for site, _ in np.ndenumerate(mat):
        nn = np.zeros((4,2))
        for idx, move in enumerate(moves_list):
            nn[idx] = (np.array(site) + move)%L
        nnlist[site] = nn

    return nnlist

@njit
def sum_nn(mat, nn_list, state):
    '''
    -----------------------------------------------------------------
    Arguments:  - 2D lattice, LxL numpy array
                - list of neighboring sites, 4x2 array
                - state to search in the nn sites, either 1 or 2

    Counts the number of neighboring sites that are in a given
    state, in a numba-friendly fashion
    -----------------------------------------------------------------
    '''
    s = np.int64(0)
    for i in range(4):
        nn_state = mat[nn_list[:,0][i], nn_list[:,1][i]]
        if nn_state == state:
            s = s + 1
    return s

@njit
def find_density(mat, state):
    '''
    -----------------------------------------------------------------
    Arguments:  - 2D lattice, LxL numpy array
                - state, either 0, 1 or 2

    Counts the number of sites in the lattice that are in the given
    state and returns the density.
    -----------------------------------------------------------------
    '''
    return np.where(mat == state)[0].size/mat.size

@njit
def node_propensity(mat, nn_mat, node, dF, bF, lF, dV, bV, lV):
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

    Reaction 0:     F --(dF)--> 0
    Reaction 1:     V --(bF)--> F
    Reaction 2: F + V --(lF)--> F + F
    Reaction 3:     V --(dV)--> 0
    Reaction 4:     0 --(bV)--> V
    Reaction 5: 0 + V --(lV)--> V + V
    -----------------------------------------------------------------
    '''
    state = mat[node[0], node[1]]
    nn_list = nn_mat[node[0], node[1]]

    if state == 0:
        nnV = sum_nn(mat, nn_list, 2)
        res = np.array([0, 0, 0, 0, bV, nnV*lV/4], dtype = np.float64)

    elif state == 1:
        res = np.array([dF, 0, 0, 0, 0, 0], dtype = np.float64)

    elif state == 2:
        nnF = sum_nn(mat, nn_list, 1)
        res = np.array([0, bF, nnF*lF/4, dV, 0, 0], dtype = np.float64)

    return res

@njit
def node_propensity_nn(state, nn_new_state, nn_old_state, dF, bF, lF, dV, bV, lV):
    '''
    -----------------------------------------------------------------
    Arguments:  - state of the node, int
                - new state of the nn that was changed by Gillespie
                  algorithm, int
                - old state of the nn before Gillespie, int
                - 6 parameters of the models, floats

    Returns the change in the propensity rate for the node in the
    given state, considering that one nn changed state.

    This is actually NOT used, we use instead a 4D matrix to encode
    directly the possible changes using [state, new_state, old_state]
    as indexing.
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    Order of the reactions:

    Reaction 0:     F --(dF)--> 0
    Reaction 1:     V --(bF)--> F
    Reaction 2: F + V --(lF)--> F + F
    Reaction 3:     V --(dV)--> 0
    Reaction 4:     0 --(bV)--> V
    Reaction 5: 0 + V --(lV)--> V + V
    -----------------------------------------------------------------
    '''

    if state == 0:
        if nn_old_state == 2:
            change = np.array([0, 0, 0, 0, 0, -lV/4], dtype = np.float64)
        elif nn_new_state == 2:
            change = np.array([0, 0, 0, 0, 0, +lV/4], dtype = np.float64)
        else:
            change = np.array([0, 0, 0, 0, 0, 0], dtype = np.float64)

    if state == 1:
        change = np.array([0, 0, 0, 0, 0, 0], dtype = np.float64)

    if state == 2:
        if nn_old_state == 1:
            change = np.array([0, 0, -lF/4, 0, 0, 0], dtype = np.float64)
        elif nn_new_state == 1:
            change = np.array([0, 0, +lF/4, 0, 0, 0], dtype = np.float64)
        else:
            change = np.array([0, 0, 0, 0, 0, 0], dtype = np.float64)

    return change

@njit
def bisect_right(a, x):
    '''
    -----------------------------------------------------------------
    Arguments:  - array
                - parameters to be inserted

    Alternative to numpy searchsorted. Finds the index to insert x in
    a, provided that a is sorted.
    -----------------------------------------------------------------
    '''
    lo = 0
    hi = len(a)

    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1

    return lo

@njit
def init_system(mat, dF, bF, lF, dV, bV, lV):
    '''
    -----------------------------------------------------------------
    Arguments:  - 2D lattice, LxL numpy array
                - 6 parameters of the models, floats

    Build the nearest-neighbors matrix and initialize the propensity
    matrix
    -----------------------------------------------------------------
    '''
    L = mat.shape[0]
    nn_mat = return_nn(mat)

    propensity_matrix = np.zeros((L, L, 6))

    for idx, _ in np.ndenumerate(mat):
        propensity_matrix[idx] = node_propensity(mat, nn_mat, idx, dF, bF, lF, dV, bV, lV)

    return nn_mat, propensity_matrix

@njit
def run_simulation(propensity_matrix, mat, nn_mat, change_matrix, nsteps, tmax,
                   dF, bF, lF, dV, bV, lV, save_every = 0.1):
    '''
    -----------------------------------------------------------------
    Arguments:  - propensity matrix, LxLx6 array
                - 2D lattice, LxL numpy array
                - neighbors matrix, LxLx4x2 numpy array
                - change matrix, 3x3x3x6 numpy array
                - number of Gillespie steps
                - maximum time
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

    density_fires = []
    density_vegetation = []

    states_list = []
    time = [0]
    states_list_time = []

    density_fires.append(find_density(mat, 1))
    density_vegetation.append(find_density(mat, 2))
    states_list.append(mat.copy())
    states_list_time.append(np.float64(0))

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
        if idx_loop % 10000 == 0:
            print('Step: ' + str(idx_loop) + ', real time:', time[-1])
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

        propensity_matrix[node[0], node[1]] = node_propensity(mat, nn_mat, node, dF, bF, lF, dV, bV, lV)

        for idx in nn_mat[node[0], node[1]]:
            propensity_matrix[idx[0], idx[1]] = propensity_matrix[idx[0], idx[1]] + change_matrix[mat[idx[0], idx[1]], old_state, mat[node]]


        time.append(time[-1] + tau)
        density_fires.append(find_density(mat, 1))
        density_vegetation.append(find_density(mat, 2))

        if density_fires[-1] + density_vegetation[-1] == 0:
            if absorbing:
                reached_abs_state = True

        if time[-1]//save_every - time[-2]//save_every != 0:
            states_list.append(mat.copy())
            states_list_time.append(time[-1])

        if reached_abs_state or time[-1] > tmax:
            return states_list_time, states_list, time, density_fires, density_vegetation

    return states_list_time, states_list, time, density_fires, density_vegetation
