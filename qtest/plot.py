import numpy as num
import matplotlib as mpl
plt = mpl.pyplot


class VisualModel():

    def __init__(self, values):
        '''
        :param values: matrix with three dimensions
        '''
        self.values = values

    def plot_zslize(self, index, ax=None, show=False, saveas=None):
        d = self.values[:,:,index]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.pcolormesh(d)
        fig.colorbar(im)

        if saveas:
            fig.savefig(saveas)

        if show:
            plt.show()


class UniqueColor():
    def __init__(self, color_map=mpl.cm.coolwarm, tracers=None):
        self.tracers = tracers
        self.color_map = color_map
        self.mapping = dict(zip(self.tracers, num.linspace(0, 1, len(self.tracers))))

    def __getitem__(self, tracer):
        return self.color_map(self.mapping[tracer])


class TracerColor():
    def __init__(self, attr, color_map=mpl.cm.coolwarm, tracers=None):
        self.tracers = tracers
        self.attr = attr
        self.color_map = color_map
        self.min_max = None

        self.set_range()

    def __getitem__(self, tracer):
        v = getattr_dot(tracer, self.attr)
        return self.color_map(self.proj(v))

    def proj(self, v):
        minv, maxv = self.min_max
        return (v-minv)/(maxv-minv)

    def set_range(self):
        vals = [getattr_dot(trs, self.attr) for trs in self.tracers]
        self.min_max = (min(vals), max(vals))

