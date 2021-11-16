import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import math

import graph as mgraph
import content as mcontent
import colors as mcolors

GRAPH_RADIUS = 100
LABEL_SIZE = 8
LEGEND_SIZE = 5
MARK_LENGTH = 2
MARK_SCALE = 10

def content(figure_name, data, pos, center=(0, 0)):
    '''
    Creates & adds graph base to figure

    center - tuple(x, y) of center of figure
    data - dict of data to draw
    '''
    mgraph.init()
    ax = plt.gca().axes

    # print(GRAPH_RADIUS, LABEL_SIZE, LEGEND_SIZE, MARK_LENGTH, MARK_SCALE)

    # Titles
    if pos == "OD":
        mgraph.add_text((center[0]-10, center[1]+130), 'OD', 25)
    elif pos == "OS":
        mgraph.add_text((center[0] - 10, center[1] + 130), 'OS', 25)
    elif pos == "N":
        mgraph.add_text((center[0] - 40, center[1] + 130), 'Normal', 25, color=mcolors.DATA_GREEN)
    elif pos == "B":
        mgraph.add_text((center[0] - 70, center[1] + 130), 'Benign disorder', 25, color=mcolors.DATA_YELLOW)
    elif pos == "D":
        mgraph.add_text((center[0] - 100, center[1] + 130), 'Dangerous disorder', 25, color=mcolors.DATA_RED)

    x = center[0]
    # Circles
    mgraph.add_circle(center, GRAPH_RADIUS, mcolors.GRAPH_EDGE_LIGHT, mcolors.GRAPH_BG, True) 
    mgraph.add_circle(center, GRAPH_RADIUS * 0.3, mcolors.GRAPH_EDGE_LIGHT, mcolors.GRAPH_FACE) 
    mgraph.add_circle(center, GRAPH_RADIUS * 0.1, mcolors.GRAPH_EDGE_LIGHT, mcolors.GRAPH_FACE) 
    mgraph.add_text((x, GRAPH_RADIUS), str(GRAPH_RADIUS) + '(%)', LABEL_SIZE)
    mgraph.add_text((x, GRAPH_RADIUS * 0.3), str(int(GRAPH_RADIUS * 0.3)), LABEL_SIZE)
    mgraph.add_text((x, GRAPH_RADIUS * 0.1), str(int(GRAPH_RADIUS * 0.1)), LABEL_SIZE)

    add_data(center, data, ax)

    # draw_legend(center)

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.axis('scaled')

    return mgraph.save_figure(figure_name)

def add_data(center, data, ax):
    '''
    Add data parsed from records to figure

    center - tuple(x, y) of center of figure
    data - dict of data to add
    '''
    test_num = len(mcontent.SUMMARY_TYPE)
    for i in range(test_num):
        angle = i * (360 / test_num)
        test = mcontent.SUMMARY_TYPE[i]
        x, y = center

        # line
        end_x, end_y = get_end(center, angle, GRAPH_RADIUS)
        color = mcolors.GRAPH_EDGE_DARK if test in data else mcolors.GRAPH_EDGE_LIGHT
        mgraph.add_line( (x, end_x), (y, end_y), color )

        # text labels
        if end_x > x:
            alignment = 'left'
        elif end_x < x:
            alignment = 'right'
        else:
            alignment = 'center'
        mgraph.add_text( get_end(center, angle, GRAPH_RADIUS + 10), test, LABEL_SIZE, alignment, color )

        # marks
        if test in data:
            test_data = data[test]
            
            v = test_data['Vertical']
            h = test_data['Horizontal']

            d_x, d_y = get_end( center, angle, test_data['Value'] )

            # POS -> RED
            if test_data['Vertical'][2] >= 0:
                ec = mcolors.DATA_EDGE_HOT
                fc = mcolors.DATA_FACE_HOT
            else:
                ec = mcolors.DATA_EDGE_COOL
                fc = mcolors.DATA_FACE_COOL

            draw_mark((d_x, d_y), v, h, ec, fc)
    return

def draw_mark(position, v, h, ec, fc):
    x, y = position
    # print(position)
    h_min = x + h[0] * MARK_SCALE
    h_1 = x + h[1] * MARK_SCALE
    h_3 = x + h[3] * MARK_SCALE
    h_max = x + h[4] * MARK_SCALE

    v_min = y + v[0] * MARK_SCALE
    v_1 = y + v[1] * MARK_SCALE
    v_3 = y + v[3] * MARK_SCALE
    v_max = y + v[4] * MARK_SCALE

    path_points = [ (h_3, v_3), (h_3, v_1), (h_1, v_1), (h_1, v_3), (h_3, v_3) ]
    # print(path_points)
    mgraph.add_path(path_points, ec=ec, fc=fc)
            
    mgraph.add_line( (h_1, h_min), (y, y), ec )
    mgraph.add_line( (h_3, h_max), (y, y), ec )
    mgraph.add_line( (h_max, h_max), (y + MARK_LENGTH, y - MARK_LENGTH), ec )
    mgraph.add_line( (h_min, h_min), (y + MARK_LENGTH, y - MARK_LENGTH), ec )
            
    mgraph.add_line( (x, x), (v_1, v_min), ec )
    mgraph.add_line( (x, x), (v_3, v_max), ec )
    mgraph.add_line( (x + MARK_LENGTH, x - MARK_LENGTH), (v_max, v_max), ec )
    mgraph.add_line( (x + MARK_LENGTH, x - MARK_LENGTH), (v_min, v_min), ec )

    mgraph.add_line((x - MARK_LENGTH, x + MARK_LENGTH), (y, y), ec)
    mgraph.add_line((x, x), (y - MARK_LENGTH, y + MARK_LENGTH), ec)

def draw_legend(center):
    '''
    Draw legend for graph

    center - center of graph
    '''
    l_x, l_y = get_end(center, 60, GRAPH_RADIUS * 1.5)
    l_array = [ -1, -0.5, 0, 0.5, 1 ]
    mgraph.add_text((l_x, l_y), 'Horizontal Nystagmus', LEGEND_SIZE)
    mgraph.add_text((l_x + 61, l_y + 17), 'Vertical Nystagmus', LEGEND_SIZE)
    mgraph.add_text((l_x + 63, l_y - 17), 'Positive', LEGEND_SIZE)
    mgraph.add_text((l_x + 90, l_y - 17), 'Negative', LEGEND_SIZE)
    draw_mark((l_x + 75, l_y), l_array, l_array, mcolors.DATA_EDGE_HOT, mcolors.DATA_FACE_HOT)
    draw_mark((l_x + 103, l_y), l_array, l_array, mcolors.DATA_EDGE_COOL, mcolors.DATA_FACE_COOL)
    mgraph.add_line((l_x + 61, l_x + 115), (l_y - 22, l_y - 22), mcolors.GRAPH_EDGE_DARK, linewidth=0.25)
    mgraph.add_text((l_x + 65, l_y - 27), 'Match Percentage', LEGEND_SIZE)

def get_end(point, angle, length):
    '''
     Get point starting from point at designated angle & length; returns start, end coordinates

     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.
     '''
    x, y = point
    return (x + length * math.cos(math.radians(angle))), (y + length * math.sin(math.radians(angle)))