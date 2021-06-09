import glob
import pickle
import matplotlib.pyplot as plt

for f in glob.glob('*.pkl'):
    print(f)
    with open(f, 'rb') as fig_obj:
        fig=pickle.load(fig_obj)
        ax_master = fig.axes[0]
        for ax in fig.axes:
            if ax is not ax_master:
                ax_master.get_shared_y_axes().join(ax_master, ax)

plt.show()