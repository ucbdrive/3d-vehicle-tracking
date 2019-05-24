import matplotlib

matplotlib.use('Agg')
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

# from mpl_toolkits.mplot3d import Axes3D

sns.set(style="darkgrid")


class RandomColor():

    def __init__(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a 
        distinct 
        RGB color; the keyword argument name must be a standard mpl colormap 
        name.'''
        self.cmap = plt.cm.get_cmap(name, n)
        self.n = n

    def get_random_color(self, scale=1):
        ''' Using scale = 255 for opencv while scale = 1 for matplotlib '''
        return tuple(
            [scale * x for x in self.cmap(np.random.randint(self.n))[:3]])


def fig2data(fig, size=(1920, 1080)):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # canvas.tostring_argb give pixmap in ARGB mode.
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    buf.shape = (h, w, 4)  # last dim: (alpha, r, g, b)

    # Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)

    buf = cv2.resize(buf[:, :, 1:], size)

    # Get BGR from RGB
    buf = buf[:, :, ::-1]

    return buf


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval,
                                               b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_depth(epoch, session, targets, inputs, outputs):
    fig = plt.figure(dpi=100)
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)
    plt.title('Depth estimation', fontsize=30)

    # Plot only valid locations
    valid = (targets != 0)
    targets = targets[valid]
    inputs = inputs[valid]
    outputs = outputs[valid]
    t = np.arange(targets.shape[0])

    plt.plot(t,
             targets,
             color='g', marker='o', linewidth=2.0, label='GT')
    plt.plot(t,
             inputs,
             color='b', marker='o', linewidth=1.0, label='INPUT')
    plt.plot(t,
             outputs,
             color='r', marker='o', linewidth=1.0, label='OUTPUT')

    plt.legend()
    plt.savefig('output/lstm/{}_{}_depth.eps'.format(session, epoch), format='eps')
    plt.close()


def plot_3D(epoch, session, cam_loc, targets, inputs, outputs):
    fig = plt.figure(dpi=100)
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (15, 5), }
    # 'aaxes.labelsize': 'x-large',
    # 'axes.titlesize':'x-large',
    # 'xtick.labelsize':'x-large',
    # 'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    min_color = 0.5
    max_color = 1.0
    show_cam_loc = False
    show_dist = False

    cm_cam = truncate_colormap(cm.get_cmap('Purples'), min_color, max_color)
    cm_gt = truncate_colormap(cm.get_cmap('Greens'), min_color, max_color)
    cm_in = truncate_colormap(cm.get_cmap('Blues'), min_color, max_color)
    cm_out = truncate_colormap(cm.get_cmap('Oranges'), min_color, max_color)

    plt.title(
        'Linear motion estimation', fontsize=30)
    ax = plt.axes(projection='3d')

    # Plot only valid locations
    valid = (np.sum(targets != 0, axis=1) > 0)
    cam_loc = cam_loc[valid]
    targets = targets[valid]
    inputs = inputs[valid]
    outputs = outputs[valid]
    t = np.linspace(0.0, 1.0, targets.shape[0])

    plt.plot(targets[:, 0],
             targets[:, 1],
             zs=targets[:, 2],
             color='g', linewidth=2.0, label='_nolegend_')
    plt.plot(inputs[:, 0],
             inputs[:, 1],
             zs=inputs[:, 2],
             color='b', linewidth=1.0, label='_nolegend_')
    plt.plot(outputs[:, 0],
             outputs[:, 1],
             zs=outputs[:, 2],
             color='r', linewidth=1.0, label='_nolegend_')
    ax.scatter(targets[:, 0],
               targets[:, 1],
               zs=targets[:, 2],
               c=t,
               cmap=cm_gt, marker='o', linewidth=4.0, label='GT')
    ax.scatter(inputs[:, 0],
               inputs[:, 1],
               zs=inputs[:, 2],
               c=t,
               cmap=cm_in, marker='o', linewidth=2.0, label='INPUT')
    ax.scatter(outputs[:, 0],
               outputs[:, 1],
               zs=outputs[:, 2],
               c=t,
               cmap=cm_out, marker='o', linewidth=2.0, label='OUTPUT')

    if show_cam_loc:
        plt.plot(cam_loc[:, 0],
                 cam_loc[:, 1],
                 zs=cam_loc[:, 2],
                 color='c', linewidth=1.0, label='_nolegend_')
        ax.scatter(cam_loc[:, 0],
                   cam_loc[:, 1],
                   zs=cam_loc[:, 2],
                   c=t,
                   cmap=cm_cam, marker='^', linewidth=2.0, label='CAM')

    if show_dist:
        for i in range(0, len(targets), 3):
            ax.text(targets[i, 0],targets[i,1],targets[i,2], 
                '{:.2f}'.format(np.sqrt(np.sum((targets[i] - cam_loc[i])**2))), size=20, zorder=1, color='k') 
            ax.text(inputs[i, 0],inputs[i,1],inputs[i,2], 
                '{:.2f}'.format(np.sqrt(np.sum((inputs[i] - cam_loc[i])**2))), size=20, zorder=1, color='k') 
            ax.text(outputs[i, 0],outputs[i,1],outputs[i,2], 
                '{:.2f}'.format(np.sqrt(np.sum((outputs[i] - cam_loc[i])**2))), size=20, zorder=1, color='k') 

    plt.legend()
    plt.savefig('output/lstm/{}_{}_3D.eps'.format(session, epoch), format='eps')
    plt.close()
