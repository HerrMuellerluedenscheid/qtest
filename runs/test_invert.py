import numpy as num
import unittest
import glob
from pyrocko import cake, pile, model as pyrocko_model
from pyrocko.gf import seismosizer
from qtest import invert, vtk_graph, distance_point2line, config, util, plot
import scipy.optimize as optimize
import matplotlib.pyplot as plt


km = 1000.



class TomoTestCase():

    def test_application(self):
        '''
        :param inversion_algorithm: pinv | minimize | nnls
        :param inversion_method: diff_ds | ds
        '''
        inversion_algorithm = 'nnls'
        inversion_method = 'ds'

        qvmin = 1./500.
        qvmax = 1./100.
        p_factor = 0.
        #p_factor = 0.5

        vminmax = qvmax * 1.
        passing_factor = 4.
        min_distance = 800.   # min traveled distance
        phase = cake.PhaseDef('p')

        conf = config.QConfig.load(filename='config.yaml')
        cake_model = cake.load_model('webnet_model1d.nd')

        sources = [seismosizer.SourceWithMagnitude.from_pyrocko_event(e) for e in
                   conf.filtered_events]

        # decimation: it is assumed, that about half are removed due to snr
        print'''
.................................................

WARN: remove every second event (assuming bad SNR)
---------------------------------------------------'''
        #sources = sources[::2]
        stations = pyrocko_model.load_stations(conf.stations)
        targets = [util.s2t(s, conf.channel) for s in stations]

        fn_mseed = glob.glob(conf.traces)
        data_pile = pile.make_pile(
            fn_mseed, fileformat=conf.file_format or 'mseed')
        #data_pile.snuffle()
        fn_out_prefix = '%s_%s' % (inversion_algorithm, inversion_method)

        load_coupler = True

        if load_coupler:
            print '''
.................................................

WARN: load coupler. Reprocess if guts changed!
---------------------------------------------------'''
            filtrate = distance_point2line.Filtrate.load_pickle(filename=conf.fn_couples)
            coupler = distance_point2line.Coupler(filtrate)
        else:
            coupler = distance_point2line.Coupler()
            coupler.magdiffmax = conf.magdiffmax
            coupler.process(sources, targets, cake_model, [phase],
                            dump_to=conf.fn_couples,
                            check_relevance_by=data_pile,
                            ignore_segments=False)

        candidates = coupler.filter_pairs(
                passing_factor,
                min_distance,
                data=coupler.filtrate,
                min_magnitude=conf.min_magnitude,
                max_magnitude=conf.max_magnitude,
                max_mag_diff=conf.magdiffmax)

        rays = [c[-1] for c in candidates]

        model = invert.DiscretizedVoxelModel.from_rays(rays, 200, 4000, 200.)

        all_z = []
        for c in candidates:
            all_z.extend(c[-1].z)
        max_z = num.max(num.array(all_z))
        min_z = num.min(num.array(all_z))
        print 'average depth', min_z + (max_z-min_z)/2.
        import pdb
        pdb.set_trace()
        ts = num.zeros((len(candidates), num.product(model._shape())))

        # setup the model:
        checkerboard = invert.CheckerboardModel.from_model(model)
        checkerboard.setup(2, 2, 2, vmin=qvmin, vmax=qvmax)

        visual_model = plot.VisualModel(values=checkerboard.values)
        for zslize in range(2):
            visual_model.plot_slize(
                direction='EW', index=zslize, show=False,
                saveas=fn_out_prefix + '_%s.png' % zslize,
            vminmax=(-vminmax, vminmax))

        q_model = num.ravel(checkerboard.values)
        model.path_discretization_delta = 2.

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

        ts += (num.random.random()-0.5)*2.*num.std(ts) * p_factor
        dtstar_theos_perturbation = (num.random.random(dtstar_theos.shape)-0.5) * 2. * num.std(dtstar_theos)*p_factor
        dtstar_theos += dtstar_theos_perturbation
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.hist(dtstar_theos)
        ax = fig.add_subplot(212)
        ax.hist(dtstar_theos_perturbation)
        # Setting up the "new differential tomography method"
        ncandidates = len(candidates)
        counter = 0
        print ncandidates
        diff_ts = num.zeros(((ncandidates**2 - ncandidates)/2, num.product(model._shape())))
        diff_dtstar_theos = num.zeros(diff_ts.shape[0])

        # equations to be minimized in sequential least squares:
        eqcons = []
        for i in range(ncandidates):
            for j in range(ncandidates):
                if j <= i:
                    continue
                else:
                    diff_ts[counter, :] = ts[i] -  ts[j]
                    diff_dtstar_theos[counter] = dtstar_theos[i] - dtstar_theos[j]
                    counter += 1
        print 'ray_actors start'
        ray_actors = [vtk_graph.vtk_ray(r) for r in rays]
        print 'ray_actors stop'
        actors = []
        actors.extend(ray_actors)
        actors.extend(checkerboard.vtk_actors())

        if inversion_method == 'ds':
            _G = ts
            _D = dtstar_theos
        elif inversion_method == 'diff_ds':
            _G = diff_ts
            _D = diff_dtstar_theos

        def search(test_model):
            return num.sqrt(num.sum((num.sum(_G*test_model, axis=1) - _D)**2))

        print 'solving %s with algorithm %s' % (inversion_method, inversion_algorithm)
        if inversion_algorithm == 'minimize':
            ''' ------------------ MINIMIZE ---------------------'''
            m_ref = invert.ModelWithValues.from_model(model)
            m_ref.values[:] = 1./300.

            bounds = num.array((num.ravel(1./(num.ones(model._shape())*1000.)),
                                num.ravel(1./(num.ones(model._shape())*10.)))).T

            result = optimize.minimize(search, x0=num.ravel(m_ref.values), bounds=bounds).x

        elif inversion_algorithm == 'nnls':
            ''' ------------------   NNLS   ---------------------'''
            result, norm = optimize.nnls(_G, _D)

        elif inversion_algorithm == 'pinv':
            ''' ------------------   PINV---------------------'''
            tspinv = num.linalg.pinv(_G, rcond=1e-5)
            result = num.dot(tspinv, _D)

        result = num.ma.masked_equal(result, 0.0)

        best_match = invert.ModelWithValues.from_model(model)
        best_match.values = result.reshape(best_match._shape())
        residuals = checkerboard.values - best_match.values

        #Jx =num.arange(model.xmin, model.xmax, model.dx)
        #Jy =num.arange(model.ymin, model.ymax+model.dy, model.dy)   #### the little lassie hack!
        #Jz =num.arange(model.zmin, model.zmax, model.dz)
        print 'done'
        visual_model = plot.VisualModel(values=best_match.values)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.text(0., 1., 'residual mean=%1.4f, std=%1.4f' % (residuals.mean(), residuals.std()),
#                transform=ax.transAxes, fontsize=14,
#                verticalalignment='top', horizontalalignment='left')
#
        yslize = 0
        visual_model.plot_slize(
            direction='EW', index=yslize, show=False,
            saveas=fn_out_prefix+'_slize_EW%s.png' % yslize,
            ax=ax,
            vminmax=(-vminmax, vminmax))

        print residuals
        fig = plt.figure()
        ax = fig.add_subplot(111)
        residual_model = plot.VisualModel(values=residuals)
        #residual_model = plot.VisualModel(x=x, y=y, z=z, values=residuals)
        residual_model.plot_slize(
            direction='EW', index=yslize, show=False,
            saveas=fn_out_prefix+'_residuals_slize_EW%s.png' % yslize,
            ax=ax, vminmax=(-vminmax, vminmax))

        actors.extend(best_match.vtk_actors())
        #vtk_graph.render_actors(best_match.vtk_actors())
        vtk_graph.render_actors(actors)

if __name__=='__main__':
    #unittest.main()
    t = TomoTestCase()
    t.test_application()


