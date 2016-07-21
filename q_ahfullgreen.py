import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rc('ytick', labelsize=10)
mpl.rc('xtick', labelsize=10)

import matplotlib.pyplot as plt
import numpy as num
import os
import glob
import logging
import progressbar
from pyrocko.gf import Target
from pyrocko import util, model, pile, moment_tensor, cake
from micro_engine import AhfullgreenTracer
from micro_engine import Noise, RandomNoiseConstantLevel, Chopper
from micro_engine import associate_responses, UniformTTPerturbation
from autogain.autogain import PhasePie
from distance_point2line import Coupler, Animator, Filtrate
from util import Magnitude2Window, Magnitude2fmin, fmin_by_magnitude
from q import Builder, e2s, UniqueColor, SyntheticCouple, QInverter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pjoin = os.path.join
km = 1000.


def ahfullgreen_test(noise_level=1e-14):
    print '-------------------db test-------------------------------'
    use_real_shit = False
    use_responses = True                            # 2 test
    load_coupler = False
    want_station = 'all'
    lat = 50.2059
    lon = 12.5152
    sources = []
    method = 'mtspec'
    deltat = 0.0005
    distance_min = 1.*km
    distance_max = 12.*km
    distance_delta = 1.*km

    depth_min = 8.*km
    depth_max = 12.*km
    depth_delta = 1.*km
    gf_padding = 50.

    min_magnitude = 2.
    max_magnitude = 6.
    fminrange = 20.
    use_common = True
    fmax_lim = 80.
    zmax = 10700
    fmax = 90.
    window_by_magnitude = Magnitude2Window.setup(0.1, 0.02)
    strikemin = 160
    strikemax = 180
    dipmin = -60
    dipmax = -80
    rakemin = 20
    rakemax = 40

    channel = 'Z'
    tt_mu = 0.
    tt_sigma = 0.0001
    save_figs = True
    vp = 6000.
    vs = 3000.
    density = 3000.
    qp = 200.
    qs = 100.


    material = cake.Material(vp=vp, vs=vs, rho=density, qp=qp, qs=qs)
    layer = cake.HomogeneousLayer(
        ztop=0., zbot=30*km, m=material, name='fullspace')
    mod = cake.LayeredModel()
    mod.append(layer)

    distances = num.arange(distance_min, distance_max, distance_delta)
    source_depths = num.arange(depth_min, depth_max, depth_delta)

    perturbation = UniformTTPerturbation(mu=tt_mu, sigma=tt_sigma)
    p_chopper = Chopper('first(p)', phase_position=0.5,
                        by_magnitude=window_by_magnitude,
                        phaser=PhasePie(mod=mod))
    stf_type = 'brunes'
    tracers = []
    want_phase = 'p'
    fn_coupler = 'dummy_coupling.yaml'
    fn_noise = '/media/usb/webnet/mseed/noise.mseed'
    fn_records = '/media/usb/webnet/mseed'
    if use_real_shit:
        noise = Noise(files=fn_noise, scale=noise_level)
        noise_pile = pile.make_pile(fn_records)
    else:
        noise = RandomNoiseConstantLevel(noise_level)
        noise_pile = None

    stations = model.load_stations('/data/meta/stations.cz.pf')
    if not want_station=='all':
        print 'warning: only using station: %s' %'.'.join(want_station)
        stations = filter(lambda x: want_station == x.nsl(), stations)

    if load_coupler:
        print 'load coupler'
        filtrate = Filtrate.load(filename=fn_coupler)
        sources = filtrate.sources
        coupler = Coupler(filtrate)
        print 'done'
    else:
        coupler = Coupler()
        target_kwargs = {
            'elevation': 0., 'codes': ('cz', 'vac', '', channel),
            'quantity': 'displacement'}
        targets = [Target(lat=lat, lon=lon, **target_kwargs)]
        sources = []
        for d in distances:
            d = num.sqrt(d**2/2.)
            for sd in source_depths:
                mag = float(2.+num.random.random()*0.0001)
                strike, dip, rake = moment_tensor.random_strike_dip_rake(
                    strikemin, strikemax, dipmin, dipmax, rakemin, rakemax)

                mt = moment_tensor.MomentTensor(
                    strike=strike, dip=dip, rake=rake, magnitude=mag)
                e = model.Event(
                    lat=lat, lon=lon, depth=float(sd), moment_tensor=mt)
                sources.append(e2s(e, north_shift=float(d), east_shift=0.,
                                   stf_type=stf_type))
        #fig, ax = Animator.get_3d_ax()
        #Animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
        #Animator.plot_sources(sources=targets, reference=coupler.hookup, ax=ax)

        if use_responses:
            associate_responses(
                glob.glob('responses/resp*'),
                targets,
                time=util.str_to_time('2012-01-01 00:00:00.'))
        logger.info('number of sources: %s' % len(sources))
        logger.info('number of targets: %s' % len(targets))
        coupler.process(sources, targets, mod, [want_phase, want_phase.lower()],
                        ignore_segments=True, dump_to=fn_coupler, check_relevance_by=noise_pile)

    pairs_by_rays = coupler.filter_pairs(4, 1200, data=coupler.filtrate, max_mag_diff=0.1)

    widgets = ['plotting segments: ', progressbar.Percentage(), progressbar.Bar()]

    pairs = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1 = p
        fmin2 = None
        pair = []
        for sx in [s1, s2]:
            fmin = fmin_by_magnitude(sx.magnitude)
            tracer = AhfullgreenTracer(
                source=sx, target=t, chopper=p_chopper, channel=channel, fmin=fmin, fmax=fmax,
                want='displacement', perturbation=perturbation.perturb(0),
                material=material, deltat=deltat)
            pair.append(tracer)
            tracers.extend(pair)
            pairs.append(pair)
    if len(tracers)==0:
        raise exception('no tracers survived the assessment')

    builder = Builder()
    tracers = builder.build(tracers, engine=None, snuffle=False)
    colors = UniqueColor(tracers=tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('location_model_db1.png', dpi=200)
    #plt.show()
    testcouples = []
    widgets = ['processing couples: ', progressbar.Percentage(), progressbar.Bar()]
    pb = progressbar.ProgressBar(len(pairs)-1, widgets=widgets).start()
    for i_p, pair in enumerate(pairs):
        pb.update(i_p)
        testcouple = SyntheticCouple(
            master_slave=pair, method=method, use_common=use_common)
        testcouple.process(noise=noise)
        if len(testcouple.spectra.spectra)!=2:
            logger.warn('not 2 spectra in test couple!')
            continue
        testcouples.append(testcouple)
    testcouples = filter(lambda x: x.good, testcouples)
    pb.finish()
    #outfn = 'testimage'
    #plt.gcf().savefig('output/%s.png' % outfn)
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    for i, testcouple in enumerate(num.random.choice(testcouples, 10)):
        fn = 'synthetic_tests/%s/example_%s_%s.png' % (want_phase,
                                                       'ahfullgreen', str(i).zfill(2))
        testcouple.plot(infos="%i"%i, colors=colors, noisy_q=False, savefig=fn)
    inverter.plot(q_threshold=800, relative_to='median', want_q=qp)
    fig = plt.gcf()
    plt.tight_layout()
    fig.savefig('synthetic_tests/%s/hist_db%s.png' %(want_phase, 'ahfullgreen'), dpi=200)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    inverter.analyze()
    plt.show()


if __name__=='__main__':
    ahfullgreen_test()
    plt.show()
