#!/usr/bin/env python3
# encoding=utf8  
import matplotlib

matplotlib.use('Agg')
import os
from pyrocko import orthodrome as ortho
from pyrocko.gui import marker
import numpy as num
import sys
from collections import defaultdict
from pyrocko.gf import *
from pyrocko.pile import make_pile
from pyrocko.model.station import load_stations
from pyrocko.model.event import Event
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle
from qtest.util import make_marker_dict, find_nearest_indx, reset_events
from qtest.util import subtract
from qtest.config import QConfig
pjoin = os.path.join


def make_northing_combination_plot(matrices, fn_out=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(
        matrices['northing_1'],
        matrices['northing_2'],
        c=matrices['qs'])

    if fn_out is not None:
        fig.savefig(fn_out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--slope-ratios')
    args = parser.parse_args()

    stations = load_stations('/data/meta/stations.pf')
    station_by_nsl = {}
    for s in stations:
        station_by_nsl[s.nsl()] = s
    station_nkc = station_by_nsl[('', 'NKC','')]

    point_size = 0.3

    if args.slope_ratios:
        with open(args.slope_ratios, 'rb') as f:
            slopes_ratios = pickle.load(f, encoding='latin1')
    else:
        slopes_ratios = {}

    want_phase = 'P'

    print('Processing phase %s' % want_phase)
    if args.config:
        config = QConfig.load(filename=args.config)
        if len(config.whitelist) == 1:
            station = config.whitelist[0]
        else:
            print('config.whitelist contains more than one station. need to specify') 
            sys.exit()

    fn_stats = 'qstats.cpickle'
    data_pile = config.get_pile()

    with open(fn_stats, 'rb') as f:
        stats = pickle.load(f)
        all_combinations = False
        n = len(stats)

        # source parameters
        matrices = dict(
            event_name_1 = num.ndarray(n),
            event_name_2 = num.ndarray(n),
            depth_1 = num.ndarray(n),
            depth_2 = num.ndarray(n),
            lateral_distance = num.ndarray(n),
            magnitude_1 = num.ndarray(n),
            magnitude_2 = num.ndarray(n),
            origin_time_1 = num.ndarray(n),
            origin_time_2 = num.ndarray(n),
            azimuth_e1_e2 = num.ndarray(n),
            azimuth_NKC_1 = num.ndarray(n),
            azimuth_NKC_2 = num.ndarray(n),
            azimuth_NKC_delta = num.ndarray(n),
            northing_1 = num.ndarray(n),
            northing_2 = num.ndarray(n),
            slope_ratio_1 = num.ndarray(n),
            slope_ratio_2 = num.ndarray(n),

            # waveform measures,
            snr_1 = num.ndarray(n),
            snr_2 = num.ndarray(n),

            # statistical measures,
            pd = num.ndarray(n),
            td = num.ndarray(n),
            qs = num.ndarray(n),
            rmse = num.ndarray(n),
            total_d = num.ndarray(n),
            rsquared = num.ndarray(n),
            incidence = num.ndarray(n),
            cc = num.ndarray(n))

        q_by_event_involved = defaultdict(list)
        events_by_name = {}

        for i, s in enumerate(stats):
            matrices['event_name_1'][i] = s['event1'].name
            matrices['event_name_2'][i] = s['event2'].name
            matrices['depth_1'][i] = s['event1'].depth
            matrices['depth_2'][i] = s['event2'].depth
            matrices['magnitude_1'][i] = s['event1'].magnitude
            matrices['magnitude_2'][i] = s['event2'].magnitude
            matrices['origin_time_1'][i] = s['event1'].time
            matrices['origin_time_2'][i] = s['event2'].time
            matrices['northing_1'][i] = s['event1'].lat
            matrices['northing_2'][i] = s['event2'].lat
            matrices['azimuth_NKC_1'][i] = ortho.azimuth(s['event1'], station_nkc)
            matrices['azimuth_NKC_2'][i] = ortho.azimuth(s['event2'], station_nkc)
            matrices['slope_ratio_1'][i] = num.log(slopes_ratios.get(s['event1'].name, 5.))
            matrices['slope_ratio_2'][i] = num.log(slopes_ratios.get(s['event2'].name, 5.))
            matrices['snr_1'][i] = s['snr1']
            matrices['snr_2'][i] = s['snr2']

            matrices['lateral_distance'][i] = s['event1'].distance_to(s['event2'])
            matrices['azimuth_e1_e2'][i] = ortho.azimuth(s['event1'], s['event2'])
            matrices['azimuth_NKC_delta'][i] = abs((matrices['azimuth_NKC_1'][-1] - matrices['azimuth_NKC_2'][-1]))
            # add after next run!
            #fignames.append(s['figname'])
            matrices['pd'][i] = s['pd']
            matrices['td'][i] = s['td']
            matrices['qs'][i] = s['q']
            matrices['rmse'][i] = s['rmse']
            matrices['rsquared'][i] = s['rsquared']
            matrices['incidence'][i] = s['incidence']
            matrices['cc'][i] = s['cc']

            q_by_event_involved[s['event1'].name].append(s['q'])
            q_by_event_involved[s['event2'].name].append(s['q'])
            events_by_name[s['event1'].name] = s['event1'].pyrocko_event()
            events_by_name[s['event2'].name] = s['event2'].pyrocko_event()

        e = []
        std_by_event = []
        qs_by_event = []
        nqs = []

        for event, q in q_by_event_involved.items():
            e.append(event)
            qs_by_event.append(1./num.median(q))
            std_by_event.append(num.std(1./num.median(q)))
            nqs.append(len(q))

        print('loading markers from NKC')
        markers_nkc = marker.PhaseMarker.load_markers(
            '/home/marius/josef_dd/markers_with_polarities.pf')

        markers_nkc = [m for m in markers_nkc if m.one_nslc()[1] == 'NKC' and
                       m.get_phasename().upper() == want_phase]
        # key_replacements = {'*.NKCN.*.*': '*.NKC.*.*'})
        # emarkers = [marker.EventMarker(ie) for ie in e]
        reset_events(markers_nkc, events_by_name.values())
        marker_by_event_name = {}
        for m in markers_nkc:
            ev = m.get_event()
            if ev:
                marker_by_event_name[ev.name] = m
            else:
                print(ev)

        # ---------------------------------------------
        # polarity analysis
        polarity_matrices = {
            'polarity_1_1': [],
            'polarity_-1_-1': [],
            'polarity_diff': [],
            'polarity_None': [],
        }

        polarities_counter = {}
        def _count_polarities(p):
            if p:
                cnt = polarities_counter.get(str(p), 0)
                cnt += 1
                polarities_counter[p] = cnt

        n_skipped = 0
        for i, s in enumerate(stats):
            e1 = s['event1'].name
            e2 = s['event2'].name
            m1 = marker_by_event_name.get(e1, None)
            m2 = marker_by_event_name.get(e2, None)
            if None in [m1, m2]:
                n_skipped += 1
                continue

            p1 = m1.get_polarity()
            p2 = m2.get_polarity()

            _count_polarities(p1)
            _count_polarities(p2)
            if p1 == p2 and p1 == 1:
                matrix_key = 'polarity_1_1'
            elif p1 == p2 and p1 == -1:
                matrix_key = 'polarity_-1_-1'
            elif p1 == 1 and p2 == -1:
                matrix_key = 'polarity_1_-1'
            elif p1 ==-1 and p2 == 1:
                matrix_key = 'polarity_-1_1'
            else:
                print(e1, p1 , ' | ', e2, p2)
                matrix_key = 'polarity_None'

            pmtrx = polarity_matrices.get(matrix_key, [])
            pmtrx.append(s['q'])
            polarity_matrices[matrix_key] = pmtrx

        ii = num.argsort(qs_by_event)

        # ---------------------------------------------
        # differential P phase waveforms ==========================
        paired_waveforms = []
        for i, s in enumerate(stats):
            e1 = s['event1'].name
            e2 = s['event2'].name
            chunk = []
            for e in [e1, e2]:
                data_pile.chop()
                m = marker_by_event_name.get(e, None)
                if m is None:
                    print('No marker for event %s' % e)
                    sys.exit()

                trs = data_pile.chop(lambda tr: tr.station==m.one_nslc()[1] and
                                     tr.channel==config.channel)
                if not len(trs) == 1:
                    print('Length of traces != 1')
                    sys.exit()

                trs[-1].shift(-m.tmin)
                chunk.extend(trs)
            paired_waveforms.append((s['q']. s['cc'], subtract(*chunk)))

        # TODO: plot
        for i in ii:
            print('event: %s  Nmeasures: %s  median: %s  std: %s  slope_ratio: %s' % (
                e[i], nqs[i], qs_by_event[i], std_by_event[i],
                slopes_ratios.get(e[i], None)))

        polarities_counter['skipped'] = n_skipped
        for args in polarities_counter.items():
            print('%s polarities: %s' % args)

        keys = sorted(list(matrices.keys()))
        combinations = []
        n_keys = len(keys)

        if all_combinations:
            nrows = n_keys
            ncolumns = n_keys
            fig, axs = plt.subplots(nrows, ncolumns, figsize=(30, 30))

            for i, ki in enumerate(keys):
                for j, kj in enumerate(keys):
                    combinations.append((ki, kj, i, j))
                #keys.remove(ki)

            for ki, kj, irow, jcolumn in combinations:
                print(ki, kj)
                ax = axs[irow][jcolumn]
                for l in ['top', 'left', 'right', 'bottom']:
                    ax.spines['top'].set_visible(False)
                if irow == 0:
                    ax.text(1., 0., str(kj), transform=ax.transAxes, rotation=90.)
                elif jcolumn == len(keys)-1:
                    ax.text(0., 1., str(ki), transform=ax.transAxes)

                ax.axis('off')
                ax.scatter(matrices[ki], matrices[kj], s=point_size, color='black', alpha=0.2)

        else:

            def polyfit(ax, x, y):
                try:
                    isorted = num.argsort(y)
                    fit = num.polyfit(y[isorted], x[isorted], 20.)
                    p = num.poly1d(fit)
                    ax.plot(p(y[isorted]),y[isorted])
                except ValueError as e:
                    pass

            keys.remove('qs')
            n_keys = len(keys)
            nrows = int(num.sqrt(n_keys))
            ncolumns = n_keys % nrows
            #fig, axs = plt.subplots(nrows, ncolumns, figsize=(30, 30))
            fig, axs = plt.subplots(n_keys, 2, figsize=(9, 4*n_keys))
            for ikey, k in enumerate(keys):
                #ax = axs[int(nrows/(ikey+1))][ikey % ncolumns]
                ax = axs[ikey][0]
                x, y = 1./matrices['qs'], matrices[k]
                ax.scatter(x, y, s=point_size, color='black', alpha=0.2)

                ax.set_xlim(-200., 200)
                ax.set_ylabel(k)
                ax.axvline(0.0, color='black', alpha=0.15)
                polyfit(ax, x, y)
      
                ax = axs[ikey][1]
                x = 1./ x
                ax.scatter(x, y, s=point_size, color='black', alpha=0.2)
                polyfit(ax, x, y)
                ax.axvline(0.0, color='black', alpha=0.15)
                ax.set_xlim(-0.3, 0.3)

        fig.subplots_adjust(#hspace=0.02, wspace=0.02, top=0.95,
                            left=0.15, bottom=0.02, right=0.95)
        fig.savefig('covariances.png', dpi=260)

        n_keys = len(polarity_matrices.keys())
        fig_polarity, axs = plt.subplots(n_keys, 1, figsize=(7, 3*n_keys))
        for i, (key, pm) in enumerate(polarity_matrices.items()):
            ax = axs[i]
            ax.hist(pm, bins=25)
            print(num.median(pm))
            ax.axvline(num.median(pm), color='b')
            ax.axvline(num.mean(pm), color='r')
            ax.set_title(key)
        fig.tight_layout()
        fig_polarity.savefig('polarities.png', dpi=260)

        make_northing_combination_plot(matrices, fn_out='northings_vs_q.png')