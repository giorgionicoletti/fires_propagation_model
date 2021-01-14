import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

import lib.fun_2D as fun
import lib.fun_plotting as fplot

class Fires_Simulation_2D():

    def __init__(self, L, fires_parameters, vegetation_parameters, prob_V = 0.3):
        '''
        -----------------------------------------------------------------
        Arguments:  - L, int
                    - fires_parameters, iterable of len 3
                    - vegetation_parameters, iterable of len 3
                    - prob_V, float
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Initialize the system randomly with a density p_V of vegetation
        sites.
        Initialize the change_matrix that is later used to update the
        propensity matrix in an efficient way based on the reaction
        that happened.
        -----------------------------------------------------------------
        '''
        self.N = int(L**2)
        self.L = int(L)

        self.cmap = matplotlib.colors.ListedColormap(['grey', 'darkred', 'darkgreen'])

        self.mat = np.random.choice([2, 0], p = [prob_V, 1 - prob_V],
                                    size = (self.L, self.L)).astype('int8')

        '''
        -----------------------------------------------------------------
        Init model parameters. If bV = 0 the model has an absorbing
        state.
        -----------------------------------------------------------------
        '''
        self.dF, self.bF, self.lF = fires_parameters
        self.dV, self.bV, self.lV = vegetation_parameters

        self.nn_mat, self.propensity_matrix = fun.init_system(self.mat,
                                                              self.dF, self.bF, self.lF,
                                                              self.dV, self.bV, self.lV)

        # state, nn_old_state, nn_new_state
        self.change_matrix = np.zeros((3,3,3,6), dtype = np.float64)
        self.change_matrix[0,2,:] = np.array([0, 0, 0, 0, 0, -self.lV/4])

        for i in [0,1]:
            self.change_matrix[0,i,2] = np.array([0, 0, 0, 0, 0, +self.lV/4])

        self.change_matrix[2,1,:] = np.array([0, 0, -self.lF/4, 0, 0, 0])

        for i in [0,2]:
            self.change_matrix[2,i,1] = np.array([0, 0, +self.lF/4, 0, 0, 0])

    def Gillespie(self, nsteps, tmax, save_every = 0.1):
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
        self.save_every = save_every

        res = fun.run_simulation(self.propensity_matrix, self.mat,
                                 self.nn_mat, self.change_matrix,
                                 nsteps, tmax,
                                 self.dF, self.bF, self.lF,
                                 self.dV, self.bV, self.lV,
                                 save_every = save_every)

        self.states_list_time, self.states_list, self.time, self.density_fires, self.density_vegetation = res

        self.states_list_time = np.array(self.states_list_time)
        self.time = np.array(self.time)

        self.density_fires = np.array(self.density_fires)
        self.density_vegetation = np.array(self.density_vegetation)
        self.density_void = 1 - self.density_fires - self.density_vegetation

    def draw_graph(self, t_index):
        '''
        -----------------------------------------------------------------
        Arguments:  - t_index, int
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Draw the lattice with the predefined colormap at a given time.
        -----------------------------------------------------------------
        '''
        fig, ax = plt.subplots(figsize = (5,5))
        im = ax.imshow(self.states_list[t_index], cmap = self.cmap, interpolation = 'none')
        ax.axis('off')
        ax.set_title(r'$t=' + str(round(self.states_list_time[t_index], 4)) + '$')
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

    def plot_probabilities(self, plot_MF = True, save = True, xl1 = None, xl2 = None):
        '''
        -----------------------------------------------------------------
        Arguments:  - plot_MF, bool
                    - save, bool
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

        if xl1 != None and xl2 != None:
            plt.xlim(xl1, xl2)

        if save:
            title = 'plot_L'+str(self.L)
            title += '_dF' + str(self.dF).replace('.', '-')
            title += '_bF' + str(self.bF).replace('.', '-')
            title += '_lF' + str(self.lF).replace('.', '-')
            title += '_dV' + str(self.dV).replace('.', '-')
            title += '_bV' + str(self.bV).replace('.', '-')
            title += '_lV' + str(self.lV).replace('.', '-')
            title += '.jpg'
            plt.savefig(title, dpi = 300, bbox_inches = 'tight')
        else:
            return fig, ax

    def animate(self, title = None):
        '''
        -----------------------------------------------------------------
        Arguments:  - title, string or None
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Function used to save the animation of the time-evolution of the
        system.
        -----------------------------------------------------------------
        '''
        fig, ax = plt.subplots(figsize = (5,5))
        plt.tight_layout()

        im = ax.imshow(self.states_list[0], cmap = self.cmap, interpolation = 'None')
        ax.axis('off')

        def init():
            im.set_data(self.states_list[0])
            ax.set_title(r"$t = " + str(round(0, 2)) + "$")
            return (im,)

        def animate(i, data):
            im.set_data(data[i])
            ax.set_title(r"$t = " + str(round(i*self.save_every, 2)) + "$")
            return (im,)

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.states_list), interval=500, blit=False,
                                       fargs = (self.states_list,))

        FFMpegWriter = animation.writers['ffmpeg']
        writer = FFMpegWriter(fps=10, bitrate=1000, codec="libx264")

        if title == None:
            title = 'movie_L'+str(self.L)
            title += '_dF' + str(self.dF).replace('.', '-')
            title += '_bF' + str(self.bF).replace('.', '-')
            title += '_lF' + str(self.lF).replace('.', '-')
            title += '_dV' + str(self.dV).replace('.', '-')
            title += '_bV' + str(self.bV).replace('.', '-')
            title += '_lV' + str(self.lV).replace('.', '-')
            title += '.mp4'

        anim.save(title, writer = writer, dpi = 300)
        plt.close()

    def save_states(self):
        title = 'states_L'+str(self.L)
        title += '_dF' + str(self.dF).replace('.', '-')
        title += '_bF' + str(self.bF).replace('.', '-')
        title += '_lF' + str(self.lF).replace('.', '-')
        title += '_dV' + str(self.dV).replace('.', '-')
        title += '_bV' + str(self.bV).replace('.', '-')
        title += '_lV' + str(self.lV).replace('.', '-')
        title += '.npy'

        np.save(title, self.states_list)
