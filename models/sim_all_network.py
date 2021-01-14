import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

import lib.fun_plotting as fplot


class Fires_Simulation_Network():

    def __init__(self, N, graph_type, fires_parameters, vegetation_parameters,
                 prob_V = 0.3, save_every = 0.1):
        '''
        -----------------------------------------------------------------
        Arguments:  - N, int
                    - graph_type, networkx graph
                    - fires_parameters, iterable of len 3
                    - vegetation_parameters, iterable of len 3
                    - prob_V, float
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Initialize the system on a graph.
        -----------------------------------------------------------------
        '''
        # Initialize graph parameters
        self.N = N
        self.graph = graph_type
        self.pos = nx.spring_layout(self.graph)

        # Initialize model parameters
        self.dF, self.bF, self.lF = fires_parameters
        self.dV, self.bV, self.lV = vegetation_parameters

        # Note: currently save_every is not used, but it is
        # useful to make animation or save the network config
        # at given times
        self.save_every = save_every
        self.time = [0]

        if self.bV == 0:
            self.absorbing = True

        # Network initialization
        for idx in self.graph:
            self.graph.nodes[idx]['state'] = np.random.choice(['V', '0'],
                                                              p = [prob_V, 1 - prob_V])
            self.graph.nodes[idx]['nn'] = list(self.graph.neighbors(idx))

        self.density_void = []
        self.density_fires = []
        self.density_vegetation = []
        self.update_density()

        # Initialize Gillespie variables
        self.reaction_update = ['0', 'F', 'F', '0', 'V', 'V']
        self.propensity_matrix = np.zeros((len(self.graph), 6))

        for idx in self.graph:
            self.propensity_matrix[idx] = self.node_propensity(idx)

        self.nreactions = self.propensity_matrix.shape[1]


    def update_density(self):
        '''
        -----------------------------------------------------------------
        Update the densities of the system.
        -----------------------------------------------------------------
        '''
        states_list = np.array([*nx.get_node_attributes(self.graph, 'state').values()])

        self.density_void.append(np.sum(states_list == '0')/self.N)
        self.density_fires.append(np.sum(states_list == 'F')/self.N)
        self.density_vegetation.append(np.sum(states_list == 'V')/self.N)


    def node_propensity(self, idx):
        '''
        -----------------------------------------------------------------
        Arguments:  - idx, int
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Gets the propensity of a given node, given its current state and
        the states of the neares neighbors.

        The order of the reaction is given by:

        Reaction 1:     F --(dF)--> 0
        Reaction 2:     V --(bF)--> F
        Reaction 3: F + V --(lF)--> F + F
        Reaction 4:     V --(dV)--> 0
        Reaction 5:     0 --(bV)--> V
        Reaction 6: 0 + V --(lV)--> V + V
        -----------------------------------------------------------------
        '''
        state = self.graph.nodes[idx]['state']
        nn_list = self.graph.nodes[idx]['nn']

        if state == '0':
            nnV = len([idx_nn for idx_nn in nn_list if self.graph.nodes[idx_nn]['state'] == 'V'])
            nnV /= len(nn_list)
            return [0, 0, 0, 0, self.bV, nnV*self.lV]

        if state == 'F':
            return [self.dF, 0, 0, 0, 0, 0]

        if state == 'V':
            nnF = len([idx_nn for idx_nn in nn_list if self.graph.nodes[idx_nn]['state'] == 'F'])
            nnF /= len(nn_list)
            return [0, self.bF, nnF*self.lF, self.dV, 0, 0]


    def gillespie_step(self):
        '''
        -----------------------------------------------------------------
        Performs a single step using a network version of the Gillespie
        algorithm.
        -----------------------------------------------------------------
        '''
        # Check if there are allowed reactions
        if np.all(self.propensity_matrix == 0):
            return False

        # Algorithm step
        r1 = np.random.rand()
        r2 = np.random.rand()

        cumsum = np.cumsum(self.propensity_matrix)
        a0 = cumsum[-1]

        tau = - 1/a0 * np.log(r1)
        idx = np.searchsorted(cumsum, r2*a0)

        tau, node, reaction = tau, idx//self.nreactions, idx%self.nreactions

        # Update network state
        self.time.append(self.time[-1] + tau)
        self.graph.nodes[node]['state'] = self.reaction_update[reaction]
        self.update_density()

        self.propensity_matrix[node] = self.node_propensity(node)

        for idx in self.graph:
            self.propensity_matrix[idx] = self.node_propensity(idx)

        return True

    def Gillespie(self, nsteps, tmax):
        '''
        -----------------------------------------------------------------
        Arguments:  - nsteps, int
                    - tmax, float
                    - save_every, float
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Run the Gillespie algorithm for nsteps or up until tmax is reached.
        The save_every parameter is used to save the configuration each
        save_every time.
        -----------------------------------------------------------------
        '''

        for idx_loop in range(nsteps):
            if idx_loop % 1000 == 0:
                print('Step: ' + str(idx_loop) + ', real time:', self.time[-1])

            keep_going = self.gillespie_step()

            if (not keep_going) or (self.time[-1] > tmax):
                self.time = np.array(self.time)
                self.density_void = np.array(self.density_void)
                self.density_vegetation = np.array(self.density_vegetation)
                self.density_fires = np.array(self.density_fires)
                return None

    def node_colors(self):
        '''
        -----------------------------------------------------------------
        Assign the colors that are later used for plotting the graph.
        -----------------------------------------------------------------
        '''
        ncol = []

        for i in self.graph.nodes():
            if self.graph.nodes[i]['state'] == '0':
                ncol.append('gray')
            if self.graph.nodes[i]['state'] == 'V':
                ncol.append('green')
            if self.graph.nodes[i]['state'] == 'F':
                ncol.append('red')

        return ncol

    def draw_graph(self):
        '''
        -----------------------------------------------------------------
        Draw the graph.
        -----------------------------------------------------------------
        '''
        nx.draw(self.graph, pos = self.pos, node_size = 50, width=0.1,
                node_color = self.node_colors())
        plt.show()

    def MF_equations(self, t, y):
        '''
        -----------------------------------------------------------------
        Arguments:  - t, float
                    - y, len 2 iterable
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Define the MF equations to be integrated.
        -----------------------------------------------------------------
        '''
        pF, pV = y

        # the model equations
        return [-self.dF*pF + (self.lF*pF + self.bF)*pV,
                -pV*(self.dV + self.bF + self.lF*pF) + (self.bV + self.lV*pV)*(1 - pV - pF)]

    def plot_probabilities(self, plot_MF = True, save = True, gtype = 'fully_connected',
                           xl1 = None, xl2 = None):
        '''
        -----------------------------------------------------------------
        Arguments:  - plot_MF, bool
                    - save, bool
                    - gtype, string
                    - xl1, float
                    - xl2, float
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Plot the densities over time.
        If requested, integrate numerically the MF equations and plot
        the solutions.
        -----------------------------------------------------------------
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))

        colors = ["lightgreen", "limegreen", "forestgreen", "green", "darkgreen"]
        nodes = [0.0, 1/6, 1/3, 1/2, 1]
        cmap_vegetation = LinearSegmentedColormap.from_list("cmap_vegetation",
                                                            list(zip(nodes, colors)))
        colors = ["indianred", "firebrick", "brown", "darkred", "maroon"]
        cmap_fires = LinearSegmentedColormap.from_list("cmap_fires",
                                                       list(zip(nodes, colors)))
        colors = ["lightgray", "silver", "darkgray", "gray", "black"]
        cmap_void = LinearSegmentedColormap.from_list("cmap_void",
                                                       list(zip(nodes, colors)))

        if plot_MF == True:
            time = np.linspace(0,np.max(self.time),1000)

            soln = solve_ivp(self.MF_equations,
                             [0,self.time[-1]],
                             [self.density_fires[0], self.density_vegetation[0]],
                             dense_output = True)

            prob = soln.sol(time)

            line_MF = ax.plot(time, prob[0], ls = '--', color = 'darkred')
            ax.plot(time, prob[1], ls = '--', color = 'darkgreen')
            ax.plot(time, 1 - prob[0]-prob[1], ls = '--', color = 'gray')
        else:
            line_MF = None

        line_sim = fplot.plot_section2d(ax, plt.Normalize(0, self.density_fires.max()),
                                        cmap_fires, self.time, self.density_fires,
                                        lw = 1.1)
        fplot.plot_section2d(ax, plt.Normalize(0, self.density_vegetation.max()),
                             cmap_vegetation, self.time, self.density_vegetation,
                             lw = 1.1)
        fplot.plot_section2d(ax, plt.Normalize(0, self.density_void.max()),
                             cmap_void, self.time, self.density_void,
                             lw = 1.1)

        ax.set_xlabel('Time', fontsize = 20)
        ax.set_ylabel('Density', fontsize = 20, labelpad = 10)
        ax.tick_params(labelsize=17)
        ax.set_xlim(0, self.time.max()*1.01)
        ax.set_ylim(-0.01, 1.01)

        ax.set_title(r'$d_F=$' + str(self.dF) + ', ' + r'$b_F=$' + str(self.bF) + ', ' +
                     r'$\lambda_F=$'+ str(self.lF) + ', ' + r'$d_V=$' + str(self.dV) + ', ' +
                     r'$b_V=$' + str(self.bV) + ', ' + r'$\lambda_V=$'+ str(self.lV),
                     fontsize = 20, pad = 10)
        if line_MF == None:
            legend_elements = [line_sim]
            handler_maps = [fplot.HandlerColorLineCollection(numpoints=len(colors))]
            labels = [r"Simulation"]
        else:
            legend_elements = [line_sim, line_MF[0]]
            handler_maps = [fplot.HandlerColorLineCollection(numpoints=len(colors)),
                            matplotlib.legend_handler.HandlerLine2D()]
            labels = [r"Simulation", r'MF solution']

        fplot.create_legend(ax, legend_elements, labels, handler_maps)

        if xl1 != None and xl2 != None:
            plt.xlim(xl1, xl2)

        if save:
            title = 'plot_N'+ str(self.N) + '_' + gtype + '_'
            title += '_dF' + str(self.dF).replace('.', '-')
            title += '_bF' + str(self.bF).replace('.', '-')
            title += '_lF' + str(self.lF).replace('.', '-')
            title += '_dV' + str(self.dV).replace('.', '-')
            title += '_bV' + str(self.bV).replace('.', '-')
            title += '_lV' + str(self.lV).replace('.', '-')
            title += '.jpg'
            plt.savefig(title, dpi = 300, bbox_inches = 'tight')
            return fix, ax
        else:
            return fig, ax
