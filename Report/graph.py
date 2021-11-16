import matplotlib.pyplot as plt
import os
import errno
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import colors as mcolors

LINE_WIDTH = 0.5

def init():
    plt.clf()
    plt.close()

def show_figure():
    '''
    Show current figure
    '''
    plt.show()

def save_figure(file_name, dpi=600):
    plt.tight_layout(2)
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print('Directory exists')
    plt.savefig(file_name, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    return os.path.abspath(file_name)

def add_patch(patch):
    '''
    Add patch to figure
    '''
    ax = plt.gca()
    ax.add_patch(patch)

def add_line(x_coords, y_coords, color, linewidth=LINE_WIDTH):
    '''
    Add line to figure

    x_coords - tuple(start_x, end_x)
    y_coords - tuple(start_y, end_y)
    '''
    plt.plot(x_coords, y_coords, color=color, linewidth=linewidth)

def add_text(position, content, size, alignment='left', color=mcolors.GRAPH_EDGE_DARK):
    '''
    Add text to figure

    position - tuple (x, y)
    content - string
    '''
    x, y = position
    plt.text(x, y, content, horizontalalignment=alignment, verticalalignment='center', 
        fontsize=size, color=color)

def add_path(path, ec=mcolors.DATA_EDGE_COOL, fc=mcolors.TRANSPARENT, poly=True):
    ax = plt.gca()
    path_params = []
    for p in path:
        path_params.append(mpath.Path.LINETO)
    path_params[0] = mpath.Path.MOVETO
    if poly:
        path_params[ len(path_params) - 1 ] = mpath.Path.CLOSEPOLY
    else:
        path_params[ len(path_params) - 1 ] = mpath.Path.STOP
    path = mpatches.PathPatch(mpath.Path(path,
        path_params), fc=fc, ec=ec, lw=LINE_WIDTH, transform=ax.transData)
    add_patch(path)


def add_circle(center, radius, ec, fc, dotted=False):
    '''
    Create circle patch
    '''
    if dotted:
        circle = plt.Circle(center, radius=radius, ec=ec, fc=fc, linestyle='--')
    else:
        circle = plt.Circle(center, radius=radius, ec=ec, fc=fc)
    add_patch(circle)
