import numpy as num
import math

from lassie import grid
from pyrocko.cake import Ray, d2m
from functools import reduce


def evenize_path(x, y, z, t, delta):
    distance = num.zeros(x.size)
    distance[1:] = num.cumsum(num.sqrt(
        (x[1:]-x[:-1])**2 + (y[1:]-y[:-1])**2 + (z[1:]-z[:-1])**2))
    n = int(math.floor(distance[-1]/delta))
    distance_even = num.arange(n)*delta + 0.5*delta
    x = num.interp(distance_even, distance, x)
    y = num.interp(distance_even, distance, y)
    z = num.interp(distance_even, distance, z)
    t = num.interp(distance_even, distance, t)
    return x, y, z, t


class Couple():
    def __init__(self, ray1, ray2):
        self.ray1 = ray1
        self.ray2 = ray2

        self.slope = None

    @property
    def dd_ray_segment(self):
        ''' The TD - part of one of the rays'''


class Ray3D(Ray):
    ''' Inherts from cake.RayPath. Extended by third dimension.'''
    def __init__(self, lat=0., lon=0., azimuth=0., *args, **kwargs):
        Ray.__init__(self, *args, **kwargs)

    def set_carthesian_coordinates(self, nshift=0., eshift=0., zshift=0., azimuth=0.):
        ''' Set the carthesian departure point of the ray'''
        self.nshift = nshift
        self.eshift = eshift
        self.zshift = zshift
        self.azimuth = azimuth

    @property
    def nezt(self):
        z, x, t = self.zxt_path_subdivided()
        az = self.azimuth/180.*num.pi
        x[0] *= d2m
        n = num.cos(az) * x[0]
        e = num.sin(az) * x[0]

        n += self.nshift
        e += self.eshift
        z = z[0] + self.zshift
        return n, e, z, t[0]

    @classmethod
    def from_RayPath(cls, ray):
        o = cls(0., 0., 0., ray.path, ray.p, ray.x, ray.t, ray.endgaps, ray.draft_pxt)
        return o

    def evenized_path(self, delta):
        ''' Return n, e, z and t of ray subsampled with *delta* in meters.
        Distances between points are regular spaced.'''
        return evenize_path(*self.nezt, delta=delta)


class Ray3DDiscretized(Ray3D):

    def __init__(self, n, e, z, t):
        self.n = n
        self.e = e
        self.z = z
        self.t = t
        assert self.t.shape == self.n.shape

    @property
    def nezt(self):
        return self.n, self.e, self.z, self.t


def distance3d_broadcast(location_array1, location_array2):
    ''' both arrays must have shape (3, n) where n is number of locations.'''
    return num.sqrt(num.sum((location_array1-location_array2)**2, axis=0))


def distance3d(x1, y1, z1, x2, y2, z2):
    ''' both arrays must have shape (3, n) where n is number of locations.'''
    return num.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def corner_index_permutations(size=1):
    n = num.tile(num.repeat([0, 1], 4), size)
    e = num.tile(num.tile(num.repeat([0, 1], 2), 2), size)
    d = num.tile(num.tile([0, 1], 4), size)

    return n, e, d


class RayCastModel(grid.Carthesian3DGrid):
    ''' Cartesian discretized model including some tomography functionalities'''
    def __init__(self, *args, **kwargs):
        grid.Carthesian3DGrid.__init__(self, *args, **kwargs)
        self.path_discretization_delta = 2.
        #self.weights = num.zeros(self._shape())

    def cast_ray(self, ray):
        ''' Get weightings from ray passing through volume. Subclass
        implementation'''
        pass

    @classmethod
    def from_rays(cls, rays, dx, dy, dz):
        '''setup a model from list of *rays* to ensure that the model covers
        all ray segments.'''
        mins = num.empty((len(rays), 3))
        maxs = num.empty((len(rays), 3))
        for ir, r in enumerate(rays):
            mins[ir] = (num.min(r.n), num.min(r.e), num.min(r.z))
            maxs[ir] = (num.max(r.n), num.max(r.e), num.max(r.z))

        mins = num.min(mins, axis=0)
        maxs = num.max(maxs, axis=0)

        o = cls(
            xmin=mins[0], xmax=maxs[0],
            ymin=mins[1], ymax=maxs[1],
            zmin=mins[2], zmax=maxs[2],
            dx=dx, dy=dy, dz=dz)
        return o

    @classmethod
    def from_model(cls, m):
        return cls(
            xmin=m.xmin,
            xmax=m.xmax,
            ymin=m.ymin,
            ymax=m.ymax,
            zmin=m.zmin,
            zmax=m.zmax,
            dx=m.dx,
            dy=m.dy,
            dz=m.dz,
        )

    def vtk_actors(self):
        values = num.ones(self._shape())*0.1
        from qtest.vtk_graph import grid_actors
        return grid_actors(values, *self._get_coords())


class DiscretizedModelNNInterpolation(RayCastModel):
    ''' Cartesian discretized model including some tomography functionalities
    '''
    def __init__(self, *args, **kwargs):
        RayCastModel.__init__(self, *args, **kwargs)
        self.max_dist_possible = num.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    def cast_ray(self, ray):
        '''return a vector with model indeces of volumes a ray was passing
        through and the distance the ray traversed within that value.

        This will be used to construct the Matrix_TD as input for the kernel.

        It seems that the others call this the derivative weighted sum (DWS)
        (http://www.diss.fu-berlin.de/diss/servlets/MCRFileNodeServlet/FUDISS_derivate_000000001580
        '''

        # Ray coordinates
        n, e, d, t = ray.evenized_path(self.path_discretization_delta)

        # clip to coordinates inside volume
        i_n = num.ravel(num.where(num.logical_and(n>=self.xmin, n<=self.xmax)))
        i_e = num.ravel(num.where(num.logical_and(e>=self.ymin, e<=self.ymax)))
        i_d = num.ravel(num.where(num.logical_and(d>=self.zmin, d<=self.zmax)))

        # combine indices above
        iwant_ray = reduce(num.intersect1d, (i_n, i_e, i_d))

        n = n[iwant_ray[:-1]]
        e = e[iwant_ray[:-1]]
        d = d[iwant_ray[:-1]]

        # indices of the corner closest to origin of the 8 nearest neighbours
        i_n = (n-self.xmin) / self.dx
        i_e = (e-self.ymin) / self.dy
        i_d = (d-self.zmin) / self.dz

        # repeat for the cubes other 7 corners
        n_permus, e_permus, d_permus = corner_index_permutations(len(i_n))
        i_n = num.repeat(i_n.astype(dtype=num.int, copy=False) , 8) + n_permus
        i_e = num.repeat(i_e.astype(dtype=num.int, copy=False) , 8) + e_permus
        i_d = num.repeat(i_d.astype(dtype=num.int, copy=False) , 8) + d_permus

        # Intersect ii_x indices to find model indices
        n = num.repeat(n, 8)
        e = num.repeat(e, 8)
        d = num.repeat(d, 8)

        # Grid coordinates
        # construct array from model points to calculate distances
        x, y, z = self._get_coords()

        xyz_array = num.vstack((x[i_n], y[i_e], z[i_d]))

        ned_array = num.vstack((n, e, d))

        # calculate distances
        w = distance3d_broadcast(ned_array, xyz_array)
        #w = distance3d(n, e, d, x[i_n], y[i_n], z[i_n])
        # normalize with maximum possible distance
        w /= self.max_dist_possible
        w = 1. - w
        weights = num.zeros(self._shape())
        for i, (ix, iy, iz) in enumerate(zip(i_n, i_e, i_d)):
            weights[ix, iy, iz] += w[i]

        return weights


class DiscretizedVoxelModel(RayCastModel):
    ''' Cartesian discretized model including some tomography functionalities'''
    def __init__(self, *args, **kwargs):
        RayCastModel.__init__(self, *args, **kwargs)

    def cast_ray(self, ray, return_quantity='distances'):
        '''return a vector with model indeces of volumes a ray was passing
        through and the distance the ray traversed within that value.

        This will be used to construct the Matrix_TD as input for the kernel.
        :param quantity: distances|times
        '''
        # Ray coordinates
        n, e, d, t = ray.evenized_path(self.path_discretization_delta)

        # clip to coordinates inside volume
        i_n = num.ravel(num.where(num.logical_and(n>=self.xmin, n<=self.xmax)))
        i_e = num.ravel(num.where(num.logical_and(e>=self.ymin, e<=self.ymax)))
        i_d = num.ravel(num.where(num.logical_and(d>=self.zmin, d<=self.zmax)))

        # combine indices above
        iwant_ray = reduce(num.intersect1d, (i_n, i_e, i_d))

        n = n[iwant_ray[:-1]]
        e = e[iwant_ray[:-1]]
        d = d[iwant_ray[:-1]]
        t = t[iwant_ray]
        t = t[1:] - t[:-1]

        if return_quantity == 'distances':
            t[:] = self.path_discretization_delta

        # indices of the corner closest to origin of the 8 nearest neighbours
        i_n = (n-self.xmin) / self.dx
        i_e = (e-self.ymin) / self.dy
        i_d = (d-self.zmin) / self.dz

        i_n = i_n.astype(dtype=num.int, copy=False)
        i_e = i_e.astype(dtype=num.int, copy=False)
        i_d = i_d.astype(dtype=num.int, copy=False)

        weights = num.zeros(self._shape())

        for i in iwant_ray[:-1]:
            weights[i_n[i], i_e[i], i_d[i]] += t[i]

        return weights


class ModelWithValues(DiscretizedVoxelModel):
    def __init__(self, *args, **kwargs):

        DiscretizedVoxelModel.__init__(self, *args, **kwargs)
        self.values = num.empty(self._shape())
        #self.values = num.zeros(self._shape())

    def vtk_actors(self):
        from qtest.vtk_graph import grid_actors
        return grid_actors(self.values, *self._get_coords())


class CheckerboardModel(ModelWithValues):

    def __init__(self, *args, **kwargs):
        ModelWithValues.__init__(self, *args, **kwargs)

    def setup(self, nx, ny, nz, vmin=1., vmax=10.):
        ''' number of check board patches in 3 dimensions'''
        i = 0
        mnx, mny, mnz = self._shape()

        xvals = vmin + ((num.sin(num.linspace(0, nx*2*num.pi, mnx))+1.)/2.) * (vmax-vmin)
        yvals = vmin + ((num.sin(num.linspace(0, ny*2*num.pi, mny))+1.)/2.) * (vmax-vmin)
        zvals = vmin + ((num.sin(num.linspace(0, nz*2*num.pi, mnz))+1.)/2.) * (vmax-vmin)

        for ix in range(mnx):
            for iy in range(mny):
                for iz in range(mnz):
                    self.values[ix, iy, iz] = (xvals[ix] + yvals[iy] + zvals[iz])/3

    @property
    def values_range(self):
        return num.min(self.values), num.max(self.values)

    @classmethod
    def from_model(cls, m):
        i = cls(
            xmin=m.xmin,
            xmax=m.xmax,
            ymin=m.ymin,
            ymax=m.ymax,
            zmin=m.zmin,
            zmax=m.zmax,
            dx=m.dx,
            dy=m.dy,
            dz=m.dz,
        )
        i.setup(3, 3, 3)
        return i


if __name__ == '__main__':

    model = DiscretizedModel()


