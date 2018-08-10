#!/usr/bin/env python3
import matplotlib as mpl

mpl.use('PDF')
from functools import partial
from qtest.distance_point2line import Coupler, Filtrate, filename_hash, fresnel_lambda, Hookup, project2enz, process_source
from pyrocko import cake
from pyrocko import model
from autogain.autogain import PickPie
from pyrocko.gui.marker import PhaseMarker, Marker, associate_phases_to_events
from pyrocko import util
from pyrocko import orthodrome as ortho
from pyrocko.gf.meta import Location
from qtest.util import e2s, s2t, reset_events, get_spectrum, read_blacklist, Counter
import logging
import os
import numpy as num
from qtest.vtk_graph import render_actors, vtk_ray
import matplotlib.pyplot as plt


logger = logging.getLogger('plot-rays')


traversing_distance_min= 400
traversing_ratio = 4.

try:
    import cPickle as pickle
except ImportError:
    import pickle


class TooManyItems(Exception):
    pass


colors = dict(
    LBC='#1f77b4',
    NKC='#ff7f0e',
    POC='#2ca02c',
)


def run(config):
    data_pile = config.get_pile()
    velocity_model = config.load_velocity_model()
    ptmax = data_pile.tmax
    ptmin = data_pile.tmin

    whitelist = config.whitelist or list(data_pile.stations.keys())

    phases = [cake.PhaseDef(config.want_phase.lower())]

    events_load = model.load_events(config.events)
    markers_load = PhaseMarker.load_markers(config.markers)
    reset_events(markers_load, events_load)

    stations = model.load_stations(config.stations)
    stations = [s for s in stations if '.'.join(s.nsl()) in whitelist]
    hookup = Hookup.from_locations(stations)

    events = []
    blacklist_events = read_blacklist(config.blacklist_events_fn)
    config.tstart = '2008-10-06 00:00:00.'
    config.tstop = '2008-10-13 23:59:59.'
    for e in events_load:
        if not e:
            nskipped += 1
            continue
        if (config.tstart and e.time < util.stt(config.tstart)) or \
                (config.tstop and e.time > util.stt(config.tstop)) or \
                (config.mag_min and e.magnitude < config.mag_min):
            continue

        events.append(e)

    print(len(events))
    nskipped = 0
    logger.warn('nskipped because event was none: %s' % nskipped)
    sources = [e2s(e) for e in events]

    candidates = []
    for station in stations:
        # for ievent, kevent in enumerate(sources):
        #     for jevent, levent in enumerate(sources[ievent+1:]):
        #         candidates.append((kevent, levent, station))
        print('processing station: %s' % station)
        markers = []
        for m in markers_load:
            if not isinstance(m, PhaseMarker):
                continue
            if not data_pile.relevant(
                    m.tmin, m.tmin,
                    trace_selector=lambda x: m.match_nslc(x.nslc_id)):
                continue

            if m.match_nsl(station.nsl()):
                m._phasename = m._phasename.upper()
                markers.append(m)

        targets = [s2t(station)]

        pie = PickPie(
            markers=markers,
            mod= cake.load_model(config.earthmodel),
            event2source=e2s,
            station2target=s2t)

        pie.process_markers(
            config.want_phase.upper(),
            stations=[station],
            channel=config.channel)

        fn_cache = os.path.join(
            config.fn_couples, 'coupler_fix_' + filename_hash(
                sources, targets, config.earthmodel, phases))

        logger.info('phase cache filename: %s' % fn_cache)

        try:
            coupler = Coupler(Filtrate.load_pickle(fn_cache))
        except (IOError) as e: 
            if not os.path.isdir(config.fn_couples):
                os.mkdir(config.fn_couples)
            coupler = Coupler()
            coupler.process(
                sources,
                targets,
                velocity_model,
                phases,
                check_relevance_by=data_pile,
                incl_segments=True,
                fn_cache=fn_cache)

        candidates.extend(
            coupler.filter_pairs(
                config.traversing_ratio,
                config.traversing_distance_min,
                coupler.filtrate,
                max_mag_diff=config.mag_delta_max,
                includes_segments=False,
            )
        )
    plot_candiates(candidates, config, hookup)


def first(x):
    if len(x) == 1:
        return x[0]
    else:
        raise TooManyItems('%s does not have length 1' % x)


def plot_candiates(candidates, config, hookup):

    in_actors = []
    actors = []
    phases = [cake.PhaseDef('p')]
    max_rays = None
    every_nth_ray = 100

    fig = plt.figure()
    ax_n = fig.add_subplot(121)
    ax_e = fig.add_subplot(122, sharey=ax_n)

    sources = []
    refloc = Location(lon=12.3, lat=50.)
    earthmodel = config.load_velocity_model()
    color_iterator = ['red', 'blue', 'green', 'yellow']
    scatter_size = 0.5
    scatter_alpha = 0.1
    azis = []
    for ic, p in enumerate(candidates[::every_nth_ray]):
        print('%s / %s' % ((ic+1)*every_nth_ray, len(candidates)))

        if max_rays is not None and ic == max_rays:
            break
        s1, s2, t, td, pd, totald, i1, ttsegment = p
        if td < traversing_distance_min:
            print('too short: %1.f2' % td)
            continue
        if td/ pd < traversing_ratio:
            print('too little ratio: %1.2f' % (td/pd))
            continue

        arrivals = earthmodel.arrivals(
            [s1.distance_to(t)*cake.m2d],
            phases=phases,
            zstart=s1.depth,
            zstop=s2.depth)

        if not len(arrivals):
            continue
        try:
            arrivals = first(arrivals)
        except TooManyItems as e:
            print(e, 'skip')
            continue
        z, x, _t = arrivals.zxt_path_subdivided(points_per_straight=200)
        z = first(z)
        x = first(x)
        x = num.array(x) * cake.d2m
        z = num.array(z)

        ifilt = num.where(_t<=ttsegment)
        ifilt = ifilt[1]
        x = x[ifilt]
        z = z[ifilt]

        # apply azimuth
        azi, _ = s1.azibazi_to(t)
        azi += 10.
        azis.append(azi)
        n = x * num.cos(azi/180. * num.pi)
        e = x * num.sin(azi/180. * num.pi)

        # apply NE-shifts
        north_shift, east_shift = ortho.latlon_to_ne(refloc, s1)
        n += north_shift
        e += east_shift
        print('xxx', t.codes)
        north_shift_2, east_shift_2 = ortho.latlon_to_ne(refloc, s2)
        color = colors[t.codes[1]]
        ax_n.scatter(north_shift, s1.depth, c=color, alpha=scatter_alpha, s=scatter_size)
        ax_e.scatter(east_shift, s1.depth, c=color, alpha=scatter_alpha, s=scatter_size)
        ax_n.scatter(north_shift_2, s2.depth, c=color, alpha=scatter_alpha, s=scatter_size)
        ax_e.scatter(east_shift_2, s2.depth, c=color, alpha=scatter_alpha, s=scatter_size)
        ax_n.plot(n, z, c=color, alpha=0.1)
        ax_e.plot(e, z, c=color, alpha=0.1, label=t.codes[1])

    
    ax_e.set_aspect('equal')
    ax_n.invert_yaxis()
    ax_e.set_xlabel('East [m]')
    ax_n.set_xlabel('North [m]')
    ax_n.invert_xaxis()
    plt.legend()
    # ax_e.set_xlabel('y')
    fig.savefig('rays_x.pdf', dpi=240)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(azis)
    fig.savefig('azis.pdf', dpi=240)


if __name__ == '__main__':

    from qtest import config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    c = config.QConfig.load(filename=args.config)
    run(c)
