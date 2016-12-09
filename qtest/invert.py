import time
import numpy as num
from scipy.optimize import basinhopping

from lassie import grid
from pyrocko.cake import evenize_path, Ray, d2m
from pyrocko.orthodrome import azimuth_numpy
from qtest.distance_point2line import project2enz
from functools import reduce


def kernel(Matrix_TD_i, Matrix_TD_j, model_vector, slopes):
    '''
    Matrix_TD_i/j = Traversing distance matrix.
    This is the Distance each ray spent within one voxel.

    model_vector = 1/Q*1/v = q*s

    slopes = slopes from spectral ratios
    '''
    return (Matrix_TD_i - Matrix_TD_j ) * model_vector - slopes



class Ray3D(Ray):
    ''' Inherts from cake.RayPath. Extended by third dimension.'''
    def __init__(self, lat=0., lon=0., azimuth=0., *args, **kwargs):
        Ray.__init__(self, *args, **kwargs)
        self.set_coordinates(lat, lon, azimuth)

    def set_coordinates(self, lat, lon, azimuth):
        self.lat_origin = lat
        self.lon_origin = lon
        self.azimuth = azimuth

    def set_carthesian_coordinates(self, nshift=0., eshift=0., zshift=0., azimuth=0.):
        ''' Set the carthesian departure point of the ray'''
        self.nshift = nshift
        self.eshift = eshift
        self.zshift = zshift
        self.azimuth = azimuth

    @property
    def lat_target(self):
        pass

    @property
    def lon_target(self):
        pass

    def orientate3d(self):
        z, x, t = self.zxt_path_subdivided()
        az = self.azimuth/180.*num.pi
        x[0] *= d2m
        n = num.cos(az) * x[0]
        e = num.sin(az) * x[0]

        n += self.nshift
        e += self.eshift
        z = z[0] + self.zshift
        return n, e, z, t[0]

    def evenized_path(self, delta):
        return evenize_path(*self.orientate3d(), delta=delta)

    @classmethod
    def from_RayPath(cls, ray):
        o = cls(0., 0., 0., ray.path, ray.p, ray.x, ray.t, ray.endgaps, ray.draft_pxt)
        return o


def distance3d_broadcast(location_array1, location_array2):
    ''' both arrays must have shape (3, n) where n is number of locations.'''
    return num.sqrt(num.sum((location_array1-location_array2)**2, axis=0))


class DiscretizedModel(grid.Carthesian3DGrid):
    ''' Cartesian discretized model.'''
    def __init__(self, *args, **kwargs):
        grid.Carthesian3DGrid.__init__(self, *args, **kwargs)
        self.path_discretization_delta = 10
        self.weights = num.zeros(self._shape())
        self.max_dist_possible = num.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    def cast_ray(self, ray):
        '''return a vector with model indeces of volumes a ray was passing
        through and the distance the ray traversed within that value.

        This will be used to construct the Matrix_TD as input for the kernel.
        '''
        # Ray coordinates
        n, e, d, t = ray.evenized_path(self.path_discretization_delta)

        # Grid coordinates
        x, y, z = self._get_coords()

        # indices of the corner closest to origin of the 8 nearest neighbours
        i_n = (n-self.xmin) / self.dx
        i_e = (e-self.ymin) / self.dy
        i_d = (d-self.zmin) / self.dz

        #i_n = num.repeat(i_n, 8)
        #i_e = num.repeat(i_e, 8)
        #i_d = num.repeat(i_d, 8)

        i_n = i_n.astype(dtype=num.int, copy=True)
        i_e = i_e.astype(dtype=num.int, copy=True)
        i_d = i_d.astype(dtype=num.int, copy=True)

        # model indices
        nx, ny, nz = self._shape()
        ii_n = num.ravel(num.where(num.logical_and(i_n>=0, i_n<=nx-1)))
        ii_e = num.ravel(num.where(num.logical_and(i_e>=0, i_e<=ny-1)))
        ii_d = num.ravel(num.where(num.logical_and(i_d>=0, i_d<=nz-1)))
        iwant = reduce(num.intersect1d, (ii_n, ii_e, ii_d))

        #iwant = num.repeat(iwant, 8)

        #shift_base = num.zeros(2)
        #shift_base[0] = 1

        #dn = num.repeat(shift_base, 8*len(iwant))
        #de = num.repeat(shift_base, len(iwant))

        xyz_array = num.vstack((x[i_n[iwant]], y[i_e[iwant]], z[i_d[iwant]]))

        # ray indices
        istart_ray = max(iwant)
        istop_ray = min(iwant)

        ned_array = num.vstack((n[iwant], e[iwant], d[iwant]))

        # calculate distances
        w = distance3d_broadcast(ned_array, xyz_array)

        # normalize with maximum possible distance
        w /= self.max_dist_possible
        w = 1. - w

        for i, (ix, iy, iz) in enumerate(zip(i_n[iwant], i_e[iwant], i_d[iwant])):
            self.weights[ix, iy, iz] += w[i]

        return self.weights

    def iter_nearest_corners(self, ix, iy, iz):
        for dx in range(2):
            for dy in range(2):
                for dz in range(2):
                    yield ix+dx, iy+dy, iz+dz

    def get_starting_model(self, initial_guess):
        m = self.get_model_vector()
        m[:] = initial_guess
        return m

def process():
    model = DiscretizedModel()

    start = model.get_starting_model(initial_guess=100)
    basinhopping(kernel, x0=starting_model)

if __name__ == '__main__':

    model = DiscretizedModel()


