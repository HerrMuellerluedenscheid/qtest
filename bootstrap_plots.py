#!/usr/bin/env python3
# encoding=utf8  
import matplotlib

matplotlib.use('Agg')
from pyrocko import orthodrome as ortho
from pyrocko.gui import marker
import numpy as num
import sys
from collections import defaultdict
from pyrocko.gf import *
from pyrocko.model.station import load_stations
from pyrocko.model.event import Event
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ImportError:
    import pickle


stations = load_stations('/data/meta/stations.pf')
station_by_nsl = {}
for s in stations:
    station_by_nsl[s.nsl()] = s
station_nkc = station_by_nsl[('', 'NKC','')]

point_size = 0.3
fn = sys.argv[1]

if len(sys.argv)>2:
    with open(sys.argv[2], 'rb') as f:
        slopes_ratios = pickle.load(f, encoding='latin1')
else:
    fn_slope_ratios = None
    slopes_ratios = {}


with open(fn, 'rb') as f:
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

        #fignames = [],

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

    matrices.update({
        'polarity_1': [],
        'polarity_-1': [],
        'polarity_None': []})

    q_by_event_involved = defaultdict(list)

    for i, s in enumerate(stats):
        matrices['event_name_1'][i] = s['event1'].name
        matrices['event_name_2'][i] = s['event2'].name
        matrices['depth_1'][i] = s['event1'].depth
        matrices['depth_2'][i] = s['event2'].depth
        matrices['lateral_distance'][i] = s['event1'].distance_to(s['event2'])
        matrices['magnitude_1'][i] = s['event1'].magnitude
        matrices['magnitude_2'][i] = s['event2'].magnitude
        matrices['origin_time_1'][i] = s['event1'].time
        matrices['origin_time_2'][i] = s['event2'].time
        matrices['northing_1'][i] = s['event1'].lat
        matrices['northing_2'][i] = s['event2'].lat
        matrices['azimuth_e1_e2'][i] = ortho.azimuth(s['event1'], s['event2'])
        matrices['azimuth_NKC_1'][i] = ortho.azimuth(s['event1'], station_nkc)
        matrices['azimuth_NKC_2'][i] = ortho.azimuth(s['event2'], station_nkc)
        matrices['azimuth_NKC_delta'][i] = abs((matrices['azimuth_NKC_1'][-1] - matrices['azimuth_NKC_2'][-1]))
        matrices['slope_ratio_1'][i] = num.log(slopes_ratios.get(s['event1'].name, 5.))
        matrices['slope_ratio_2'][i] = num.log(slopes_ratios.get(s['event2'].name, 5.))
        matrices['snr_1'][i] = s['snr1']
        matrices['snr_2'][i] = s['snr2']

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
    markers_nks = marker.PhaseMarker.load_markers('/home/marius/josef_dd/markers_with_polarities.pf')
    emarkers = [marker.EventMarker(ie) for ie in e]


    ii = num.argsort(qs_by_event)

    for i in ii:
        print('event: %s  Nmeasures: %s  median: %s  std: %s  slope_ratio: %s' % (
            e[i], nqs[i], qs_by_event[i], std_by_event[i],
            slopes_ratios.get(e[i], None)))

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
        keys.remove('qs')
        n_keys = len(keys)
        nrows = int(num.sqrt(n_keys))
        ncolumns = n_keys % nrows
        #fig, axs = plt.subplots(nrows, ncolumns, figsize=(30, 30))
        fig, axs = plt.subplots(n_keys, 2, figsize=(9, 4*n_keys))
        for ikey, k in enumerate(keys):
            #ax = axs[int(nrows/(ikey+1))][ikey % ncolumns]
            ax = axs[ikey][0]
            ax.scatter(1./matrices['qs'], matrices[k], s=point_size, color='black', alpha=0.2)
            ax.set_xlim(-200., 200)
            ax.set_ylabel(k)
            ax.axvline(0.0, color='black', alpha=0.15)
  
            ax = axs[ikey][1]
            ax.scatter(matrices['qs'], matrices[k], s=point_size, color='black', alpha=0.2)
            ax.axvline(0.0, color='black', alpha=0.15)
            ax.set_xlim(-0.3, 0.3)


    fig.subplots_adjust(#hspace=0.02, wspace=0.02, top=0.95,
                        left=0.15, bottom=0.02, right=0.95)
    fig.savefig('covariances.png', dpi=260)
    # plt.show()
