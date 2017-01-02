import numpy as num
import unittest
from pyrocko import cake
from qtest import invert, vtk_graph
import scipy.optimize as optimize


km = 1000.

class TomoTestCase(unittest.TestCase):

    def test_distance3d_broadcast(self):
        ned = num.zeros((3, 10))
        xyz = num.ones((3, 10))
        d = num.empty(ned.shape[1])
        d.fill(num.sqrt(3.))
        num.testing.assert_array_almost_equal(
            invert.distance3d_broadcast(ned, xyz), d)

        ned = num.random.random((3, 10))-0.5
        xyz = num.random.random((3, 10))-0.5

    def test_ray_casting(self):
        #model = invert.DiscretizedVoxelModel(
        model = invert.DiscretizedVoxelModel(
            xmin=-1*km,
            xmax=1*km,
            ymin=-1*km,
            ymax=1*km,
            zmin=12*km,
            zmax=15*km,
            dx=0.4*km,
            dy=0.4*km,
            dz=0.4*km
        )
        weights = None

        model.path_discretization_delta = 10.

        cake_model = cake.load_model('webnet_model1d.nd')
        ray = cake_model.arrivals(phases=[cake.PhaseDef('p')],
                                  distances=[33000.*cake.m2d],
                                  zstart=13.3*km)[0]


        ray_3d = invert.Ray3D.from_RayPath(ray)
        ray_3d.set_carthesian_coordinates(-101, -101., azimuth=40.)
        actors = []
        weights = model.cast_ray(ray_3d)
        if False:
            print 'size of weight matrix', weights.nbytes
            xg, yg, zg = model._get_coords()
            actors.extend(vtk_graph.vtk_grid(weights, xg, yg, zg, normalize=True))

            x, y, z, t = ray_3d.evenized_path(model.path_discretization_delta)

            inp = num.vstack((x, y, z))
            actors.append(vtk_graph.vtk_ray(inp))
            actors.append(vtk_graph.vtk_points(inp, cube_size=3.))
            vtk_graph.render_actors(actors)

    def test_index_repetitions(self):
        n, e, d = invert.corner_index_permutations(2)
        num.testing.assert_equal(n[0].dtype, num.int)
        num.testing.assert_array_equal(n, num.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                                                  0, 0, 1, 1, 1, 1],
                                                 dtype=num.int))

        num.testing.assert_equal(n.shape, e.shape)
        num.testing.assert_equal(n.shape, d.shape)

    def test_application(self):
        model = invert.DiscretizedVoxelModel(
            xmin=-1*km,
            xmax=1*km,
            ymin=-1*km,
            ymax=1*km,
            zmin=12*km,
            zmax=15*km,
            dx=0.4*km,
            dy=0.4*km,
            dz=0.4*km
        )

        model.path_discretization_delta = 10.

        cake_model = cake.load_model('webnet_model1d.nd')

        phases = [cake.PhaseDef('p')]

        # number of measurements (slopes/tstars)
        n_measurements = 2

        # right side gives on slope (aka t-star)
        ray_r_1 = cake_model.arrivals(phases=phases,
                                      distances=[0.*cake.m2d],
                                      zstart=15.*km)[0]

        ray_r_1 = invert.Ray3D.from_RayPath(ray_r_1)
        ray_r_1.set_carthesian_coordinates(0, -105., azimuth=0.)

        ray_r_2 = cake_model.arrivals(phases=phases,
                                      distances=[0.*cake.m2d],
                                      zstart=11.*km)[0]

        ray_r_2 = invert.Ray3D.from_RayPath(ray_r_2)
        ray_r_2.set_carthesian_coordinates(0, -105., azimuth=0.)


        # left side gives on slope (aka t-star)
        ray_l_1 = cake_model.arrivals(phases=phases,
                                      distances=[0.*cake.m2d],
                                      zstart=15.*km)[0]

        ray_l_1 = invert.Ray3D.from_RayPath(ray_l_1)
        ray_l_1.set_carthesian_coordinates(0, 105., azimuth=0.)

        ray_l_2 = cake_model.arrivals(phases=phases,
                                      distances=[0.*cake.m2d],
                                      zstart=11.*km)[0]

        ray_l_2 = invert.Ray3D.from_RayPath(ray_l_2)
        ray_l_2.set_carthesian_coordinates(0, 105., azimuth=0.)

        cr = invert.Couple(ray_r_1, ray_r_2)
        cl = invert.Couple(ray_l_1, ray_l_2)

        cr.slope = 0.1
        cl.slope = 0.2

        couples = [cr, cl]

        # intermediate step required: ray path geometry
        G = []
        slopes = []

        for c in couples:
            ti_1 = model.cast_ray(c.ray1, return_quantity='times')
            ti_2 = model.cast_ray(c.ray2, return_quantity='times')

            G.append(num.ravel(ti_1 - ti_2))
            slopes.append(c.slope)

        # Fuer jedes couple gibt es ein deltat*
        # delta_t* = sum(qi*ti)_1 - sum(qi*ti)_2 = slope
        # m = qi        (i: index of voxel)
        # G = ti        (i: index of voxel)
        # d = slopej    (j: index of couple)


        # measurements:
        # has to have dimensions of model!
        #tstars = num.array([0.01, 0.02])

        def search(m_test):
            n_G = len(G)
            tstars = num.zeros(n_G)
            errors = num.zeros(n_G)
            for ig in range(n_G):
                delta_tstar = G[ig] * m_test
                errors[ig] = num.sum(delta_tstar) - slopes[ig]

            # L1 norm
            return num.sum(num.abs(errors))

        m_ref = invert.CheckerboardModel.from_model(model)
        result = optimize.basinhopping(search, m_ref.values)
        print result

if __name__=='__main__':
    unittest.main()


