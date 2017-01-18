import matplotlib as mpl
mpl.use('Agg')
mpl.rc('ytick', labelsize=10)
mpl.rc('xtick', labelsize=10)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as num
import os
import glob
import progressbar
import logging
from pyrocko.gf import meta, Target, LocalEngine
from pyrocko import orthodrome
from pyrocko.gui_util import PhaseMarker
from pyrocko import util
from pyrocko import cake, model
from pyrocko import pile
from pyrocko import moment_tensor
from pyrocko.fomosto import qseis
from pyrocko.trace import nextpow2
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import signal, interpolate
from qtest.micro_engine import DataTracer, Tracer, Builder
from qtest.micro_engine import Noise, Chopper
from qtest.micro_engine import associate_responses, UniformTTPerturbation
from qtest.distance_point2line import Coupler, Animator, Filtrate, fresnel_lambda
from qtest.q import SyntheticCouple, QInverter, QInverter3D
from qtest.plot import UniqueColor
from qtest.util import Magnitude2Window, Magnitude2fmin, fmin_by_magnitude
from qtest.util import e2extendeds, e2s, s2t, konnoohmachi
from qtest.config import QConfig
from qtest.sources import DCSourceWid
from qtest.vtk_graph import render_actors, vtk_ray
from qtest.invert import DiscretizedVoxelModel, ModelWithValues

from autogain.autogain import PhasePie, PickPie


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')
logger.setLevel('DEBUG')
pjoin = os.path.join
km = 1000.

def getattr_dot(obj, attr):
    v = reduce(getattr, attr.split('.'), obj)
    return v


def pb_widgets(message=''):
    return [message, progressbar.Percentage(), progressbar.Bar()]


def xy2targets(x, y, o_lat, o_lon, channels, **kwargs):
    assert len(x) == len(y)
    targets = []
    for istat, xy in enumerate(zip(x, y)):
        for c in channels:
            lat, lon = orthodrome.ne_to_latlon(o_lat, o_lon, *xy)
            kwargs.update({'lat': float(lat), 'lon': float(lon),
                           'codes': ('', '%i' % istat, '', c)})
            targets.append(Target(**kwargs))
    return targets


def legend_clear_duplicates(ax):
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def ax_if_needed(ax):
    if not ax:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
    return ax


def flatten_list(a):
    return [c for e in a for c in e]


def plot_traces(tr, t_shift=0, ax=None, label='', color='r'):
    ax = ax_if_needed(ax)
    ydata = tr.get_ydata()
    if tr:
        tr.normalize()
        #tr.shift(-tr.tmin)
        ax.plot(tr.get_xdata()+t_shift, tr.get_ydata(), label=label, color=color)


def plot_model(mod, ax=None, label='', color=None, parameters=['qp']):
    ax = ax_if_needed(ax)
    z = mod.profile('z')
    #colors = 'rgbcy'
    colors = ['black']
    label_mapping = {'qp': 'Q$_p$',
                     'qs': 'Q$_s$',
                     'vp': 'v$_p$',
                     'vs': 'v$_s$'}
    for ip, parameter in enumerate(parameters):
        profile = mod.profile(parameter)
        if ip>=1:
            ax = ax.twiny()
        if parameter in ['vp', 'vs']:
            profile /= 1000.
        ax.plot(profile, z/1000., label=label, c=colors[ip])
        ax.set_xlabel(label_mapping[parameter], color=colors[ip])
        ax.margins(0.02)
        ax.invert_yaxis()
        ax.set_ylabel('depth [km]')
    #minz = min(z)
    #maxz = max(z)
    #zrange = maxz-minz

    #ax.set_ylim([minz-0.1*zrange, maxz+0.1zrange])


def infos(ax, info_string):
    ax.axis('off')
    ax.text(0., 0, info_string, transform=ax.transAxes)



def apply_filter(tr, order, flow, fhigh, demean=True):
    '''Apply butterworth highpass to tr.
       from pyrocko.trace.Trace

    Mean is removed before filtering.
    '''
    for corner, btype in [(flow, 'low'), (fhigh, 'high')]:
        tr.nyquist_check(corner, 'Corner frequency of highpass',
                            warn=True, raise_exception=False)
        (b,a) = _get_cached_filter_coefs(order, [corner*2.0*tr.deltat],
                                         btype=btype)
        data = tr.ydata.astype(num.float64)
        if len(a) != order+1 or len(b) != order+1:
            logger.warn('Erroneous filter coefficients returned by scipy.signal.butter')
        if demean:
            data -= num.mean(data)
        tr.drop_growbuffer()
        #self.ydata = signal.lfilter(b,a, data)
        tr.ydata = signal.filtfilt(b,a, data)


def slope_histogram(ax, spectra, colors):
    slopes = defaultdict(list)
    for tracer, spectra in spectra.get_slopes():
        x, y, y_off, slope = spectra
        slopes[tracer].append(slope)
    ax.set_title('Spectral Slopes')
    for tracer, sl in slopes.items():
        ax.hist(sl, color=colors[tracer], alpha=1., bins=20)


def plot_response(response, ax=None):
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    freqs = num.exp(num.linspace(num.log(0.05), num.log(100.), 200))
    #freqs = num.linspace(0.1, 200., 200)
    a = response.evaluate(freqs)
    ax.plot(freqs, a)
    ax.set_xscale('log')
    ax.set_yscale('log')


def wanted_q(mod, z):
    q = mod.layer(z).material(z).qp
    return q


def dbtest(config):
    noise_level=1E-8
    print '-------------------db test-------------------------------'
    use_real_shit = True
    use_extended_sources = False
    use_responses = True
    load_coupler = False
    fn_coupler = 'dummy_coupling.p'
    quantity = 'velocity'
    dump_coupler = False
    #fn_coupler = 'dummy_coupling.yaml'
    #fn_coupler = 'pickled_couples.p'
    #fn_coupler = None
    test_scenario = True
    normalize_waveforms = True
    #want_station = ('cz', 'nkc', '')
    #want_station = ('cz', 'kac', '')
    want_station = 'all'
    lat = 50.2059
    lon = 12.5152
    sources = []

    if config.method == 'filter':
        delta_f = 3.
        fcs = num.arange(30, 85, delta_f)
        fwidth = 3
        filters = [(f, fwidth) for f in fcs]
    else:
        filters = None
    #use_common = False
    use_common = True
    fmin = 35.
    fmin = Magnitude2fmin.setup(lim=config.fmin_lim)
    #window_by_magnitude = Magnitude2Window.setup(0.8, 1.)
    window_by_magnitude = Magnitude2Window.setup(0.08, 5.)
    print config
    test_config = config.synthetic_config
    store_id = test_config.store_id

    # setting the dc components:

    strikemin = 160
    strikemax = 180
    dipmin = -60
    dipmax = -80
    rakemin = 20
    rakemax = 40
    #strikemin = 170
    #strikemax = 170
    #dipmin = -70
    #dipmax = -70
    #rakemin = 30
    #rakemax = 30

    #engine = LocalEngine(store_superdirs=['/data/stores', '/media/usb/stores'])
    engine = LocalEngine(store_superdirs=test_config.store_superdirs)
    #engine = LocalEngine(store_superdirs=['/media/usb/stores'])
    store = engine.get_store(store_id)
    gf_config = engine.get_store_config(store_id)
    mod = gf_config.earthmodel_1d

    gf_padding = 50
    zmin = gf_config.source_depth_min + gf_padding
    zmax = gf_config.source_depth_max - gf_padding
    dist_min = gf_config.distance_min
    dist_max = gf_config.distance_max
    tt_mu = 0.0
    tt_sigma = 0.2
    nucleation_radius = 0.1

    # distances used if not real sources:
    if test_scenario:
        distances = num.linspace(gf_config.distance_min+gf_padding,
                                 gf_config.distance_max-gf_padding, 12)
        source_depths = num.linspace(zmin, zmax, 12)
    else:
        distances = num.arange(gf_config.distance_min+gf_padding,
                               gf_config.distance_max-gf_padding, 200)
        source_depths = num.arange(zmin, zmax, 200)

    perturbation = UniformTTPerturbation(mu=tt_mu, sigma=tt_sigma)
    #perturbation = None
    perturbation.plot()
    plt.show()
    p_chopper = Chopper('first(p)', phase_position=0.5,
                        by_magnitude=window_by_magnitude,
                        phaser=PhasePie(mod=mod))
    stf_type = 'brunes'
    #stf_type = 'halfsin'
    #stf_type =  None
    #stf_type =  'gauss'
    tracers = []
    channels = {'P': 'SHZ', 'S': 'SHN' }
    channel = channels[config.want_phase.upper()]
    #fn_noise = '/home/marius/src/qtest/noise' #/*.mseed'
    fn_records = '/media/usb/webnet/mseed'
    #if use_real_shit:
    if test_config.noise_files:
        noise = Noise(files=test_config.noise_files, scale=noise_level,
                      selectby=['station', 'channel'])
        data_pile = pile.make_pile(fn_records)
    else:
        #noise = RandomNoiseConstantLevel(noise_level)
        noise = None
        data_pile = None

    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    all_depths = [e.depth for e in events]
    some_depths = [d/1000. for d in all_depths if d>8500]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('$q_p$ model')
    ax.invert_yaxis()
    ax.set_ylim(0, 12.)
    plot_model(mod, ax=ax, parameters=['qp'])
    ax.axhspan(min(some_depths), max(some_depths), alpha=0.1)

    ax = fig.add_subplot(1, 2, 2, sharey=ax)
    ax.set_title('source depths')
    ax.hist(some_depths, bins=17, orientation='horizontal')
    ax.set_xlabel('count')
    ax.set_ylim(0, 12.)
    ax.axes.get_yaxis().set_visible(False)
    #ax.yaxis.tick_right()
    ax.axhspan(min(some_depths), max(some_depths), alpha=0.1)
    #ax.set_ylim(0, 12000)
    #plt.gca().xaxis.set_major_locator(mpl.ticker.maxnlocator(prune='lower'))
    fig.subplots_adjust(wspace=0.11, right=0.98, top=0.94)
    ax.set_xticks(num.arange(0, 120., 30.))
    #plt.tight_layout()
    ax.invert_yaxis()
    fig.savefig('model_event_depths.png')

    average_depth = num.mean(all_depths)
    want_q = wanted_q(mod, average_depth)
    vp = mod.layer(average_depth).material(average_depth).vp
    #lat = float(num.mean([e.lat for e in events]))
    #lon = float(num.mean([e.lon for e in events]))
    stations = model.load_stations('/data/meta/stations.cz.pf')
    if config.whitelist:
        stations = filter(
            lambda x: util.match_nslcs(
                '%s.%s.%s.*' % x.nsl(), config.whitelist),  stations)

    if load_coupler:
        print 'load coupler'
        filtrate = Filtrate.load_pickle(filename=fn_coupler)
        sources = filtrate.sources
        targets = filtrate.targets
        for t in targets:
            t.store_id = store_id
        coupler = Coupler(filtrate)
        print 'done'
    else:
        coupler = Coupler()
        if use_real_shit is False:
            target_kwargs = {
                'elevation': 0., 'codes': ('cz', 'vac', '', channel), 'store_id': store_id}
            targets = [Target(lat=lat, lon=lon, **target_kwargs)]
            sources = []
            for d in distances:
                d = num.sqrt(d**2/2.)
                for sd in source_depths:
                    #mag = float(1.+num.random.random()*0.2)
                    mag = 1.
                    strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
                                                                             dipmin, dipmax,
                                                                             rakemin, rakemax)
                    mt = moment_tensor.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)
                    e = model.Event(lat=lat, lon=lon, depth=float(sd), moment_tensor=mt)
                    if use_extended_sources is True:
                        sources.append(e2extendeds(e, north_shift=float(d),
                                                   east_shift=float(d),
                                                   nucleation_radius=nucleation_radius,
                                                   stf_type=stf_type))
                    else:
                        sources.append(e2s(e, north_shift=float(d),
                                           east_shift=float(d),
                                           stf_type=stf_type))
            #fig, ax = Animator.get_3d_ax()
            #Animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
            #Animator.plot_sources(sources=targets, reference=coupler.hookup, ax=ax)

        elif use_real_shit is True:
            targets = [s2t(s, channel, store_id=store_id) for s in stations]
            events = filter(lambda x: x.depth>zmin and x.depth<zmax, events)
            events = filter(lambda x: x.magnitude>=config.min_magnitude, events)
            events = filter(lambda x: x.magnitude<=config.max_magnitude, events)
            events = filter(lambda x: x.depth<=zmax, events)
            for e in events:
                strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
                                                                         dipmin, dipmax,
                                                                         rakemin, rakemax)
                mt = moment_tensor.MomentTensor(
                    strike=strike, dip=dip, rake=rake, magnitude=e.magnitude)
                #mt.magnitude = e.magnitude
                e.moment_tensor = mt
            if use_extended_sources is True:
                sources = [e2extendeds(
                    e, nucleation_radius=nucleation_radius, stf_type=stf_type)
                           for e in events]
            else:
                sources = [e2s(e, stf_type=stf_type)
                           for e in events]
        if use_responses:
            #associate_responses(
            #    glob.glob('responses/resp*'),
            #    targets,
            #    time=util.str_to_time('2012-01-01 00:00:00.'))
            associate_responses(glob.glob('/home/marius/src/qtest/responses/*pz'),
                                targets,
                                time=util.str_to_time('2012-01-01 00:00:00.'),
                                type='polezero')

        #plot_response(response=targets[0].filter.response)
        logger.info('number of sources: %s' % len(sources))
        logger.info('number of targets: %s' % len(targets))
        if dump_coupler:
            dump_to = fn_coupler
        else:
            dump_to = None
        coupler.process(sources, targets, mod, [config.want_phase, config.want_phase.lower()],
                        ignore_segments=False, dump_to=dump_to,
                        check_relevance_by=data_pile)
    #fig, ax = animator.get_3d_ax()
    #animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
    pairs_by_rays = coupler.filter_pairs(2, config.traversing_distance_min,
                                         data=coupler.filtrate,
                                         max_mag_diff=config.magdiffmax)
    paired_sources = []
    in_actors = []
    actors = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1, ray_segments = p
        #print ray_segments
        if not (s1, s2, t) in in_actors:
            actors.append(vtk_ray(num.array(ray_segments.nezt[:3])))
        else:
            in_actors.append((s1, s2, t))

        paired_sources.extend([s1, s2])
    used_mags = [s.magnitude for s in paired_sources]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(used_mags)

    testcouples = []
    b = Builder()
    for r in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1, ray_segment = r

        fmax = min(vp/fresnel_lambda(totald, td, pd), config.fmax_lim)
        fmin1 = fmin_by_magnitude(s1.magnitude)
        #if config.want_phase.upper()=="S":
        #    # accounts for fc changes: Abstract
        #    # http://www.geologie.ens.fr/~madariag/Programs/Mada76.pdf
        #    fmin1 /= 1.5
        tracer1 = Tracer(s1, t, p_chopper, channel=channel, fmin=fmin1,
                         fmax=fmax, want=quantity,
                         perturbation=perturbation.perturb(0),
                         engine=engine, noise=noise)

        fmin2 = fmin_by_magnitude(s2.magnitude)
        tracer2 = Tracer(s2, t, p_chopper, channel=channel, fmin=fmin1,
                         fmax=fmax, want=quantity, 
                         perturbation=perturbation.perturb(0),
                         engine=engine, noise=noise)
        if fmax-fmin1<config.fminrange or fmax-fmin2<config.fminrange:
            print s1.magnitude, s2.magnitude
            print 'skip because of fminrange', fmax-fmin1, fmax-fmin2
            continue
        else:
            pair = [tracer1, tracer2]
            b.build(pair)
            testcouple = SyntheticCouple(master_slave=pair,
                                         method=config.method, use_common=use_common,
                                         ray_segment=ray_segment)
            testcouple.normalize_waveforms = True
            testcouple.ray = r
            testcouple.filters = filters
            testcouple.process()
            if testcouple.good:
                testcouples.append(testcouple)
    #render_actors(actors)

    colors = UniqueColor(tracers=tracers)
    #inverter = QInverter(couples=testcouples, cc_min=config.cc_min, onthefly=True,
    #                     snr_min=config.snr_min)
    grid = DiscretizedVoxelModel.from_rays([c.ray_segment for c in testcouples], dx=200., dy=200., dz=200)
    d = num.zeros(grid._shape())
    for c in testcouples:
        d += grid.cast_ray(c.ray_segment)
    mwv = ModelWithValues.from_model(grid)
    mwv.values = d

    #actors.extend(grid.vtk_actors())
    actors.extend(mwv.vtk_actors())
    render_actors(actors)
    print 'grid shape: ', grid._shape()
    inverter = QInverter3D(couples=testcouples, discretized_grid=grid)
    inverter.invert()

    # Good old stuff: ----------------------------------------------------------
    # good_results = filter(lambda x: x.invert_data is not None, testcouples)
    # for i, tc in enumerate(num.random.choice(good_results, 30)):
    #     fn = util.ensuredirs('%s/example_%s.png' % (config.output,
    #                                                    str(i).zfill(2)))
    #     tc.plot(infos=infos, colors=colors, savefig=fn)
    # inverter.plot()
    # fn_results = '%s/results.txt' % (config.output)
    # util.ensuredirs(fn_results)
    # inverter.dump_results(fn_results)

    # fn_hist = '%s/hist_application.png' % (config.output)
    # util.ensuredirs(fn_hist)
    # fig = plt.gcf()
    # fig.savefig(fn_hist, dpi=600)

    # fn_analysis = "%s/q_fit_analysis" % (config.output)
    # util.ensuredirs(fn_analysis)
    # inverter.analyze(fnout_prefix=fn_analysis)
    # --------------------------------------------------------------------------

    #testcouples = []
    #pairs = []
    #for p in pairs_by_rays:
    #    s1, s2, t, td, pd, totald, i1 = p
    #    fmin2 = None
    #    pair = []
    #    for sx in [s1, s2]:
    #        fmin1 = fmin_by_magnitude(sx.magnitude)
    #        #if want_phase.upper()=="S":
    #            # accounts for fc changes: Abstract
    #            # http://www.geologie.ens.fr/~madariag/Programs/Mada76.pdf
    #            #fmin1 /= 1.5

    #        fmax = min(fmax_lim, vp/fresnel_lambda(totald, td, pd))
    #        #print 'test me, change channel code id to lqt'
    #        #t.dip = -90. + i1
    #        #t.azimuth = t.azibazi_to(sx)[1]
    #        tracer1 = Tracer(sx, t, p_chopper, channel=channel, fmin=fmin1,
    #                         fmax=fmax, want=quantity, 
    #                         perturbation=perturbation.perturb(0))
    #        tracer1.engine = engine
    #        dist1, depth1 = tracer1.get_geometry()
    #        if dist1< dist_min or dist1>dist_max:
    #            break
    #        if fmax-fmin1<fminrange:
    #            break
    #        pair.append(tracer1)
    #        tracers.extend(pair)
    #        pairs.append(pair)

    #    if len(pair)==2:
    #        testcouple = SyntheticCouple(master_slave=pair, method=method, use_common=use_common)
    #        testcouple.normalize_waveforms = normalize_waveforms
    #        testcouple.ray = p
    #        testcouple.filters = filters
    #        #testcouple.process(noise=noise)
    #        #if len(testcouple.spectra.spectra)!=2:
    #        #   logger.warn('not 2 spectra in test couple!!!! why?')
    #        #   continue
    #        testcouples.append(testcouple)
    #if len(tracers)==0:
    #    raise Exception('no tracers survived the assessment')

    ##builder = Builder()
    ##tracers = builder.build(tracers, engine=engine, snuffle=False)
    #colors = UniqueColor(tracers=tracers)
    #inverter = QInverter(couples=testcouples, onthefly=True, cc_min=0.8)
    #inverter.invert()
    #for i, testcouple in enumerate(num.random.choice(testcouples, 30)):
    #    fn = 'synthetic_tests/%s/example_%s_%s.png' % (want_phase, store_id, str(i).zfill(2))
    #    print fn
    #    testcouple.plot(infos=infos, colors=colors, noisy_q=False, savefig=fn)
    #inverter.plot()
    ##inverter.plot(q_threshold=800, relative_to='median', want_q=want_q)
    #fig = plt.gcf()
    #plt.tight_layout()
    #fig.savefig('synthetic_tests/%s/hist_db%s.png' %(want_phase, store_id), dpi=200)
    ##location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #inverter.analyze()
    #plt.show()


def reset_events(markers, events):
    for e in events:
        marks = filter(lambda x: x.get_event_time()==e.time, markers)
        for phase in set([m.get_phasename() for m in marks]):
            work = filter(lambda x: x.get_phasename() == phase, marks)
            map(lambda x: x.set_event(e), work)
            # set NKCN to NKC where NKC is not picked
            nkc_picks = filter(lambda x: x.one_nslc()[1]=='NKC', work)
            nkcn_picks = filter(lambda x: x.one_nslc()[1]=='NKCN', work)
            if len(nkc_picks) == 0:
                if len(nkcn_picks) == 1:
                    newid = list(nkcn_picks[0].nslc_ids[0])
                    newid[1] = 'NKC'
                    nkcn_picks[0].nslc_ids = [tuple(newid)]


def run(config):
    print '-------------------apply  -------------------------------'

    load_coupler = False
    dump_coupler = False
    fn_coupler = 'couplings/webnet_dd_pickle_P.yaml'

    if config.method == 'filter':
        fwidth = 2.
        delta_f = 3.
        fcs = num.arange(40, 70, delta_f)
        filters = [(f, fwidth) for f in fcs]
    else:
        filters = None
    use_common = True

    vp = 5000.
    fmin_by_magnitude = Magnitude2fmin.setup(lim=config.fmin_lim)

    mod = cake.load_model(config.earthmodel)
    markers = PhaseMarker.load_markers(config.markers)
    events = list(model.Event.load_catalog(config.events))
    fn_mseed = glob.glob(config.traces)
    data_pile = pile.make_pile(fn_mseed, fileformat='gse2')
    print "nevents ", len(events)
    print "nmarkers ", len(markers)
    print 'initially %s events' , len(events)
    events = filter(lambda x: x.time>data_pile.tmin and x.time<data_pile.tmax, events)
    print 'after time filtering %s ' , len(events)
    events = filter(lambda x: x.magnitude >= config.min_magnitude, events)
    events = filter(lambda x: x.magnitude<= config.max_magnitude, events)
    print '%s events'% len(events)
    reset_events(markers, events)

    stations = model.load_stations(config.stations)
    if config.whitelist:
        stations = filter(
            lambda x: util.match_nslcs(
                '%s.%s.%s.*' % x.nsl(), config.whitelist),  stations)

    pie = PickPie(markers=markers, mod=mod, event2source=e2s, station2target=s2t)

    window_by_magnitude = {'P': Magnitude2Window.setup(0.08, 3.),
                           'S': Magnitude2Window.setup(0.12, 3.1)}[config.want_phase]
    phase_position = {'S': 0.2, 'P': 0.4}

    # in order to rotate into lqt system
    rotate_channels = {'in_channels': ('SHZ', 'SHN', 'SHE'),
                       'out_channels': ('L', 'Q', 'T')}

    channels = {'P': 'SHZ', 'S': 'SHN' }
    channel = channels[config.want_phase.upper()]
    pie.process_markers(phase_selection=config.want_phase, stations=stations, channel=channel)
    p_chopper = Chopper(
        startphasestr=config.want_phase, by_magnitude=window_by_magnitude,
        phase_position=phase_position[config.want_phase.upper()], phaser=pie)
    tracers = []

    if load_coupler:
        logger.warn('LOAD COUPLER')

    ignore = ['*.STC.*.SHZ']

    if data_pile.tmax==None or data_pile.tmin == None:
        raise Exception('failed reading mseed')
    # webnet Z targets:
    targets = [s2t(s, channel) for s in stations]
    if load_coupler:
        filtrate = Filtrate.load_pickle(filename=fn_coupler)
        sources = filtrate.sources
        sources = filter(lambda x: x.magnitude>=config.min_magnitude, sources)
        coupler = Coupler(filtrate)
    else:
        coupler = Coupler()
        coupler.magdiffmax = config.magdiffmax
        coupler.whitelist = config.whitelist
        sources = [e2s(e) for e in events]
        if dump_coupler:
            dump_to = fn_coupler
        else:
            dump_to = None

        coupler.process(
            sources, targets, mod, [config.want_phase, config.want_phase.lower()],
            check_relevance_by=data_pile,
            ignore_segments=False, dump_to=dump_to)

    pairs_by_rays = coupler.filter_pairs(2., config.traversing_distance_min,
                                         data=coupler.filtrate,
                                         ignore=ignore,
                                         max_mag_diff=config.magdiffmax)

    testcouples = []
    actors = []
    in_actors = []
    b = Builder()
    for r in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1, ray_segment = r

        if not (s1, s2, t) in in_actors:
            actors.append(vtk_ray(ray_segments))
        else:
            in_actors.append((s1, s2, t))
        fmax = min(vp/fresnel_lambda(totald, td, pd), config.fmax_lim)
        fmin1 = fmin_by_magnitude(s1.magnitude)
        #if config.want_phase.upper()=="S":
        #    # accounts for fc changes: Abstract
        #    # http://www.geologie.ens.fr/~madariag/Programs/Mada76.pdf
        #    fmin1 /= 1.5
        tracer1 = DataTracer(data_pile=data_pile, source=s1, target=t,
                             chopper=p_chopper, want_channel=channel, fmin=fmin1,
                             fmax=fmax, incidence_angle=i1)
                             #rotate_channels=rotate_channels)

        fmin2 = fmin_by_magnitude(s2.magnitude)
        tracer2 = DataTracer(data_pile=data_pile, source=s2, target=t,
                             chopper=p_chopper, want_channel=channel, fmin=fmin2,
                             fmax=fmax, incidence_angle=i1)
                             #rotate_channels=rotate_channels)
        if fmax-fmin1<config.fminrange or fmax-fmin2<config.fminrange:
            print fmax-fmin1, fmax-fmin2
            continue
        else:
            pair = [tracer1, tracer2]
            b.build(pair)
            testcouple = SyntheticCouple(master_slave=pair,
                                         method=config.method, use_common=use_common,
                                         ray_segment=ray_segment)
            testcouple.normalize_waveforms = True
            testcouple.ray = r
            testcouple.filters = filters
            testcouple.process()
            if testcouple.good:
                testcouples.append(testcouple)

            #render_actors(actors)
            #testcouple = SyntheticCouple(master_slave=pair,
            #                             method=config.method, use_common=use_common)

    #render_actors(actors)

    colors = UniqueColor(tracers=tracers)
    #inverter = QInverter(couples=testcouples, cc_min=config.cc_min, onthefly=True,
    #                     snr_min=config.snr_min)
    #inverter.invert()
    grid = DiscretizedVoxelModel.from_rays([c.ray_segment for c in testcouples], dx=200., dy=200., dz=200)
    print 'grid shape: ', grid._shape()
    actors.append(grid.vtk_actors())
    render_actors(actors)
    inverter = QInverter3D(couples=testcouples, discretized_grid=grid)
    result, result_model = inverter.invert()
    actors = result_model.vtk_actors()
    import pdb
    pdb.set_trace()
    #good_results = filter(lambda x: x.invert_data is not None, testcouples)
    #for i, tc in enumerate(num.random.choice(good_results, 30)):
    #    fn = '%s/example_%s.png' % (config.output, str(i).zfill(2))
    #    util.ensuredirs(fn)
    #    tc.plot(infos=infos, colors=colors, savefig=fn)
    #inverter.plot()

    #fn_results = '%s/results.txt' % config.output
    #util.ensuredirs(fn_results)
    #inverter.dump_results(fn_results)

    #fig = plt.gcf()
    #fn_hist = '%s/hist_application.png' % config.output
    #util.ensuredirs(fn_hist)
    #fig.savefig(fn_hist, dpi=400)

    #fn_analysis = "%s/q_fit_analysis" % config.output
    #util.ensuredirs(fn_analysis)
    #inverter.analyze(fnout_prefix=fn_analysis)


if __name__=='__main__':
    import sys

    fn = sys.argv[1]
    conf = QConfig.load(filename=fn)
    if conf.type == 'synthetic':
        dbtest(conf)
    elif conf.type == 'real':
        run(conf)
    else:
        raise Exception('unknoen test type')
    plt.show()


