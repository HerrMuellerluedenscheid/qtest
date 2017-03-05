import numpy as num
import unittest
from pyrocko import cake, model as pyrocko_model
from pyrocko.gf import seismosizer
from qtest import invert, vtk_graph, distance_point2line, config, util, plot
import scipy.optimize as optimize
import matplotlib.pyplot as plt


km = 1000.


class TomoTestCase():

    def test_application(self):
        qvmin = 1./500.
        qvmax = 1./100.

        passing_factor = 3.
        min_distance = 400.   # min traveled distance
        phase = cake.PhaseDef('p')
        magnitude_delta_max = 0.1

        conf = config.QConfig.load(filename='config.yaml')
        cake_model = cake.load_model('webnet_model1d.nd')

        sources = [seismosizer.SourceWithMagnitude.from_pyrocko_event(e) for e in
                   conf.filtered_events]
        stations = pyrocko_model.load_stations(conf.stations)
        print stations
        targets = [util.s2t(s, conf.channel) for s in stations]
        coupler = distance_point2line.Coupler()
        coupler.process(sources, targets, cake_model, [phase], ignore_segments=False)

        candidates = coupler.filter_pairs(
                passing_factor,
                min_distance,
                data=coupler.filtrate,
                max_mag_diff=magnitude_delta_max)

        rays = [c[-1] for c in candidates]

        model = invert.DiscretizedVoxelModel.from_rays(rays, 300, 300, 400.)

        ts = num.zeros((len(candidates), num.product(model._shape())))

        # setup the model:
        checkerboard = invert.CheckerboardModel.from_model(model)
        checkerboard.setup(2, 2, 2, vmin=qvmin, vmax=qvmax)

        visual_model = plot.VisualModel(values=checkerboard.values)
        for zslize in range(6):
            visual_model.plot_zslize(zslize, show=False, saveas='%s.png' % zslize)

        q_model = num.ravel(checkerboard.values)
        print q_model

        model.path_discretization_delta = 10.

        # dtstar_theos are the "measured" tstar values
        dtstar_theos = num.zeros(len(candidates))

        # Setting up the "standard tomography method"
        for ic, candidate in enumerate(candidates):
            ray_segment = candidate[-1]
            t_i = num.ravel(model.cast_ray(ray_segment, return_quantity='times'))

            # times in segments (aka ti):
            ts[ic] = t_i

            # theoretical delta-t*:
            # The data vector, so to speak
            dtstar_theos[ic] = num.sum(t_i * q_model)


        # Setting up the "new differential tomography method"
        ncandidates = len(candidates)
        counter = 0
        diff_ts = num.zeros(((ncandidates**2 - ncandidates)/2, num.product(model._shape())))
        diff_dtstar_theos = num.zeros(diff_ts.shape[0])
        for i in range(ncandidates):
            for j in range(ncandidates):
                if j <= i:
                    continue
                else:
                    diff_ts[counter, :] = ts[i] -  ts[j]
                    diff_dtstar_theos[counter] = dtstar_theos[i] - dtstar_theos[j]
                    counter += 1

        ray_actors = [vtk_graph.vtk_ray(r) for r in rays]
        actors = []
        actors.extend(ray_actors)
        actors.extend(checkerboard.vtk_actors())

        if False:
            def search(test_model):
                return num.sqrt(num.sum((num.sum(ts*test_model, axis=1) - dtstar_theos)**2))

            m_ref = invert.ModelWithValues.from_model(model)
            m_ref.values[:] = 1./300.

            bounds = num.array((num.ravel(1./(num.ones(model._shape())*1000.)),
                                num.ravel(1./(num.ones(model._shape())*10.)))).T

            if True:
                #result, norm = optimize.nnls(ts, dtstar_theos)
                result, norm = optimize.nnls(diff_ts, diff_dtstar_theos)
                best_match = invert.ModelWithValues.from_model(model)
                best_match.values = result.reshape(best_match._shape())

            else:
                result = optimize.minimize(search, x0=num.ravel(m_ref.values), bounds=bounds)
                best_match = invert.ModelWithValues.from_model(model)
                best_match.values = result.x.reshape(best_match._shape())
            visual_model = plot.VisualModel(values=best_match.values)

            for zslize in range(6):
                visual_model.plot_zslize(zslize, show=True, saveas='minimize_%sres.png' % zslize)

            print 'best result...'
            vtk_graph.render_actors(best_match.vtk_actors())

            difference = invert.ModelWithValues.from_model(model)
            difference.values = checkerboard.values - best_match.values
            vtk_graph.render_actors(difference.vtk_actors())

        else:
            # )))))))))))))))))))))))))))))))))))))))))))))))))))))))))
            # "old" inversion
            # pseudo inverse mit cutoff:
            # _________________________________________________________
            #tspinv = num.linalg.pinv(ts, rcond=1e-10)
            #mtheo = num.dot(tspinv, dtstar_theos)
            print ' Solving pseudo inverse'
            tspinv = num.linalg.pinv(diff_ts, rcond=1e-10)
            mtheo = num.dot(tspinv, diff_dtstar_theos)

            ginverse_solution = invert.ModelWithValues.from_model(model)
            ginverse_solution.values = num.reshape(mtheo, ginverse_solution._shape())

            visual_model = plot.VisualModel(values=ginverse_solution.values)
            for zslize in range(6):
                visual_model.plot_zslize(zslize, show=True, saveas='%sres.png' % zslize)

            vtk_graph.render_actors(ginverse_solution.vtk_actors())

            # resolution matrix:
            #res = tspinv * ts.T



if __name__=='__main__':
    #unittest.main()
    t = TomoTestCase()
    t.test_application()


