import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import graph as mgraph
import content as mcontent
import colors as mcolors

TEXT_H_PADDING = 0.1
TEXT_H_PADDING = 0.1

def content(figure_name, data, type, dynamic=True):
    mgraph.init()
    if type == "pur":
        fig, axs = plt.subplots(3, 2, figsize=(10,12))
        types = mcontent.DYNAMIC_DETAIL_TYPE[0:6]
    elif type == "sin":
        fig, axs = plt.subplots(3, 2, figsize=(10,12))
        types = mcontent.DYNAMIC_DETAIL_TYPE[6:12]
    elif type == "sac":
        fig, axs = plt.subplots(3, 2, figsize=(10,12))
        types = mcontent.DYNAMIC_DETAIL_TYPE[12:18]
    elif type == "okn":
        fig, axs = plt.subplots(2, 2, figsize=(10,12))
        types = mcontent.DYNAMIC_DETAIL_TYPE[18:22]
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10,12))
        types = mcontent.STATIC_DETAIL_TYPE
    for i in range( len(types) ):
        d_name = types[i]
        # if dynamic:
        #     subplot = axs[int(i/2)][i%2]
        # else:
        #     subplot = axs[i]
        subplot = axs[int(i / 2)][i % 2]
        subplot.set_title(d_name, size=12)
        if d_name in data:
            # print(d_name)
            subplot.yaxis.set_ticks([30, 20, 10, 0, -10, -20, -30])
            for t in data[d_name]:
                # print(t)
                # if dynamic:
                x, y = data[d_name][t]
                if ('Target' in t) and ('Horizontal' in t):
                    subplot.plot(x, y, alpha=0.25, color='blue', linewidth=0.8)
                elif ('Target' in t) and ('Vertical' in t):
                    subplot.plot(x, y, alpha=0.25, color='red', linewidth=0.8)
                elif 'Horizontal' in t:
                    subplot.plot(x, y, color='blue', linewidth=0.6)
                elif 'Vertical' in t:
                    subplot.plot(x, y, color='red', linewidth=0.6)
                # else:
                #     x, y = data[d_name][t]
                #     if ('Target' in t) and ('Horizontal' in t):
                #         subplot.plot(x, y, label="T_H", alpha=0.25, color='blue', linewidth=0.8)
                #     elif ('Target' in t) and ('Vertical' in t):
                #         subplot.plot(x, y, label="T_V", alpha=0.25, color='red', linewidth=0.8)
                #     elif 'Horizontal' in t:
                #         subplot.plot(x, y, label="H", color='blue', linewidth=0.6)
                #     elif 'Vertical' in t:
                #         subplot.plot(x, y, label="V", color='red', linewidth=0.6)

            subplot.set_xlim(xmin=0)
            subplot.set_ylim([-30, 30])
            subplot.set_xlabel(mcontent.DETAIL_GRAPH_XY_LABELS[0])
            subplot.set_ylabel(mcontent.DETAIL_GRAPH_XY_LABELS[1])
            subplot.text(subplot.get_xlim()[0], 30, mcontent.DETAIL_GRAPH_CORNER_LABELS[0], size=8, ha='left', va='top')
            subplot.text(subplot.get_xlim()[0], -30, mcontent.DETAIL_GRAPH_CORNER_LABELS[1], size=8, ha='left', va='bottom')
            subplot.text(subplot.get_xlim()[1], 30, mcontent.DETAIL_GRAPH_CORNER_LABELS[2], size=8, ha='right', va='top')
            subplot.text(subplot.get_xlim()[1], -30, mcontent.DETAIL_GRAPH_CORNER_LABELS[3], size=8, ha='right', va='bottom')
            subplot.grid(linestyle='--')
            # if not dynamic:
            #     subplot.legend(loc='best')
        else:
            subplot.get_yaxis().set_visible(False)
            subplot.get_xaxis().set_visible(False)
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)
            subplot.spines['bottom'].set_visible(False)
            subplot.spines['left'].set_visible(False)
            subplot.text(0.5, 0.5, 'N/A', fontsize=18, ha='center')
    return mgraph.save_figure(figure_name)
