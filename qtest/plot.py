import numpy as num
import matplotlib as mpl
plt = mpl.pyplot


class VisualModel():

    def __init__(self, values, x=None, y=None, z=None):
        '''
        :param values: matrix with three dimensions
        '''
        if any([x is not None or y is not None or z is not None]):
            assert(values.shape == (len(x), len(y), len(z)))
        else:
            nx, ny, nz = values.shape
            x = num.arange(nx)
            y = num.arange(ny)
            z = num.arange(nz)

        self.x = x
        self.y = y
        self.z = z

        self.values = values

    def plot_slize(self, direction, index, ax=None, show=False, saveas=None, vminmax=None):
        if vminmax is None:
            absmax = num.max(num.abs(self.values))
            vmin, vmax = -absmax, absmax
        else:
            vmin, vmax = vminmax

        if direction == 'NS':
            d = self.values[:, :, index]
            x, y = num.meshgrid(self.y, self.z)

        elif direction == 'EW':
            d = self.values[:, index, :]
            x, y = num.meshgrid(self.z, self.x)

        elif direction == 'Z':
            d = self.values[index, :, :]
            x, y = num.meshgrid(self.x, self.y)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()

        im = ax.pcolormesh(x, y, d, vmin=vmin, vmax=vmax, cmap='RdBu')

        cb = fig.colorbar(im)
        cb.set_label('1/Q')

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

