from cmap import Color, Colormap
import matplotlib.pyplot as plt 
import scienceplots # type: ignore

cmap = Colormap('colorcet:cet_cbtc2').to_mpl()
cmap2 = Colormap([cmap(0.5), Color('lightcoral'), cmap(0.5), Color('lightsteelblue'), cmap(0.5)]).to_mpl()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def apply():
    plt.rcdefaults()
    plt.style.use(['science', 'light'])
    plt.rcParams['figure.figsize'] = (5,4)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    # set_matplotlib_formats('svg')

apply()