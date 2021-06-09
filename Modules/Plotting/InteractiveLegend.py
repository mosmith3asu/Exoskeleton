import numpy as np
import matplotlib.pyplot as plt
#from Modules.Plotting import InteractiveLegend
def create(legends):
    LEGENDS = []
    for legend in legends:
        LEGENDS.append(InteractiveLegend(legend))
    return LEGENDS

class InteractiveLegend(object):
    def __init__(self, legend=None):
        if legend == None:
            legend = plt.gca().get_legend()
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.update()
    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))
        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist
        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))
        return lookup_artist, lookup_handle
    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()
    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return
        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()
    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

if __name__ == '__main__':
    for i in range(20):
        plt.plot(np.random.randn(1000), label=i)
    plt.legend()
    leg = InteractiveLegend()
    plt.show()