import matplotlib
from matplotlib import cm
import matplotlib.colors as pltcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLineCollection
import numpy as np

class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize, trans):

        x = np.linspace(0, width, self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1) + height/2. - ydescent

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=artist.cmap,
                            transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth()+1)
        return [lc]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = pltcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_section2d(ax, norm, cmap, x, y,
                   ls = 'solid', lw = 1, zorder = 1, alpha = 1):
    points = np.array([x, y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segs, cmap = cmap, norm = norm, linestyles = ls,
                        linewidths = lw, alpha = alpha, zorder = zorder)
    lc.set_array(y)
    ax.add_collection(lc)

    return lc


def create_legend(ax, legend_elements, labels, handler_maps = None,
                  fontsize = 17, loc = 'upper right'):

    if handler_maps == None:
        handler_maps = [matplotlib.legend_handler.HandlerLine2D()]*len(legend_elements)

    handler_dict = dict(zip(legend_elements,handler_maps))

    ax.legend(legend_elements, labels, handler_map = handler_dict,
              framealpha=1, fontsize = 17, loc = loc)
