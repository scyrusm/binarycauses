import matplotlib.pyplot as plt
import numpy as np


def plot_adjacencies(ax, adjacencies, binary_adjacencies=None, plot_title=''):
    """
    A handy plotting function
    Arguments
    -------
    ax:  a matplotlib axis
    adjaciences:  a unary adjacency matrix
    binary_adjacencies:  a binary adjacency dictionary
    plot_title:  the desired plot title

    Returns
    -------
    a (now-filled) matplotlib axis
    """
    assert adjacencies.shape[0] == adjacencies.shape[1]
    n_nodes = adjacencies.shape[0]
#    fig, ax = plt.subplots()
    bounds = [-n_nodes * 0.5 - 0.5, n_nodes * 0.5 + 0.5]
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    node_x_pos = {'cause': -n_nodes * 0.25, 'effect': n_nodes * 0.25}
    label_x_pos = {'cause': -n_nodes * 0.35, 'effect': n_nodes * 0.35}
    node_y_pos = {
        j: (-j + n_nodes * 0.5 - 0.5)
        for j in np.arange(adjacencies.shape[1])
    }
    cause_nodes = {}
    effect_nodes = {}
    for i in range(adjacencies.shape[0]):
        cause_nodes[i] = plt.Circle(
            (node_x_pos['cause'], node_y_pos[i]),
            0.2,
            facecolor='y',
            edgecolor='k')
        effect_nodes[i] = plt.Circle(
            (node_x_pos['effect'], node_y_pos[i]),
            0.2,
            facecolor='y',
            edgecolor='k')
        ax.add_artist(cause_nodes[i])
        ax.add_artist(effect_nodes[i])
        ax.text(
            label_x_pos['cause'],
            node_y_pos[i],
            'C{}'.format(i),
            fontsize=17,
            horizontalalignment='center',
            verticalalignment='center')
        ax.text(
            label_x_pos['effect'],
            node_y_pos[i],
            'E{}'.format(i),
            fontsize=17,
            horizontalalignment='center',
            verticalalignment='center')

        for j in range(adjacencies.shape[1]):
            if adjacencies[i, j] == 1:
                ax.annotate(
                    "",
                    xy=(node_x_pos['effect'], node_y_pos[i]),
                    xytext=(node_x_pos['cause'], node_y_pos[j]),
                    arrowprops=dict(width=2))

    if binary_adjacencies != None:
        for key in binary_adjacencies.keys():
            for ieffect, adj in enumerate(binary_adjacencies[key]):
                if adj == 1:
                    ax.annotate(
                        "",
                        xy=(node_x_pos['effect'], node_y_pos[ieffect]),
                        xytext=(node_x_pos['cause'], node_y_pos[key[0]]),
                        arrowprops=dict(
                            arrowstyle="fancy",
                            fc="0.6",
                            ec="none",
                            patchB=effect_nodes[ieffect],
                            connectionstyle="angle3,angleA=85,angleB=5"))
                    ax.annotate(
                        "",
                        xy=(node_x_pos['effect'], node_y_pos[ieffect]),
                        xytext=(node_x_pos['cause'], node_y_pos[key[1]]),
                        arrowprops=dict(
                            arrowstyle="fancy",
                            fc="0.6",
                            ec="none",
                            patchB=effect_nodes[ieffect],
                            connectionstyle="angle3,angleA=85,angleB=5"))
    ax.set_title(plot_title, fontsize=17)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
#    plt.show()
