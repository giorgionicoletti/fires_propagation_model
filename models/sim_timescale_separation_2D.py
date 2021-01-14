import numpy as np
from numba import njit
from numba import prange

import lib.fun_2D
import lib.fun_timescale_separation_2D

@njit(parallel=True)
def find_avalanches(states_list, labeled_states_list, dF, lF, subclusters = 100):
    Nstates = states_list.shape[0]
    L = states_list.shape[1]

    avalanches_per_state = np.zeros(Nstates)

    # compute the number of clusters in this state
    # and select a subsample (if needed)

    for idx_state in prange(Nstates):
        labeled_state = labeled_states_list[idx_state]
        nclusters = np.max(labeled_state)
        if nclusters > subclusters:
            avalanches_per_state[idx_state] = subclusters
        else:
            avalanches_per_state[idx_state] = nclusters

    total_number_avalanches = np.int64(np.sum(avalanches_per_state))
    av_size = np.zeros(total_number_avalanches)
    av_time = np.zeros(total_number_avalanches)

    change_matrix = np.zeros((3,3,3,6), dtype = np.float64)

    change_matrix[2,1,:] = np.array([0, 0, -lF/4, 0, 0, 0], dtype = np.float64)
    for i in [0,2]:
        change_matrix[2,i,1] = np.array([0, 0, +lF/4, 0, 0, 0], dtype = np.float64)

    nn_mat = lib.fun_2D.return_nn(np.zeros((L,L)))

    tmax = 100
    for idx_state in prange(Nstates):
        state = states_list[idx_state]*2
        labeled_state = labeled_states_list[idx_state]
        n_avalanches = np.int64(avalanches_per_state[idx_state])

        # avoid prange bug
        previous_clusters = 0
        for l in range(idx_state):
            previous_clusters = previous_clusters + avalanches_per_state[l]

        nclusters = np.max(labeled_state)

        if nclusters > subclusters:
            choose_clusters = np.random.choice(np.arange(1, nclusters + 1), size = subclusters)
        else:
            choose_clusters = np.arange(1, nclusters + 1)

        Nempty = L**2 - np.where(state == 2)[0].size

        for idx_av in range(n_avalanches):
            idx_cluster = choose_clusters[idx_av]
            vidx_list = np.where(labeled_state == idx_cluster)
            Nvegetation = vidx_list[0].size

            idx_node = np.random.randint(Nvegetation)
            node = [vidx_list[0][idx_node], vidx_list[1][idx_node]]
            mat = state.copy()

            mat[node[0], node[1]] = 1
            mat = mat.astype(np.int8)

            propensity_matrix = np.zeros((L, L, 6), dtype = np.float64)
            propensity_matrix[node[0], node[1]] = np.array([dF, 0, 0, 0, 0, 0], dtype = np.float64)

            for idx_nn in nn_mat[node[0], node[1]]:
                if mat[idx_nn[0], idx_nn[1]] == 2:
                    propensity_matrix[idx_nn[0], idx_nn[1]] = np.array([0, 0, lF/4, 0, 0, 0], dtype = np.float64)

            time, _, _ = lib.fun_timescale_separation_2D.run_simulation(propensity_matrix, mat, nn_mat, change_matrix,
                                                                        int(1e8), tmax, dF, 0, lF, 0, 0, 0)

            current_idx = np.int64(previous_clusters + idx_av)
            burned_area = np.sum((mat == 0)) - Nempty
            av_size[current_idx] = burned_area
            av_time[current_idx] = time[-1]

    return av_size, av_time
