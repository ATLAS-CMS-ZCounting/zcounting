import matplotlib as mpl
import matplotlib.pyplot as plt

def set_matplotlib_style():

    textsize = 16
    markersize = 4.0
    labelsize =  12.5

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino",],
        "font.size": textsize,
        'text.latex.preamble': r"\usepackage{bm}"
    })

    mpl.rcParams.update({
        "legend.fontsize" : 14,
        "legend.frameon" : False,
        "legend.handletextpad" : 0.1,
        "legend.columnspacing" : 0.5,
        "axes.labelsize" : "medium",
        "axes.titlesize" : "medium",
        "xtick.labelsize" : "medium",
        "ytick.labelsize" : "medium",
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
    })
    colors = ["darkorange","cornflowerblue", "limegreen"]

    return colors, textsize, labelsize, markersize