import logging
import numpy as num
import os
from scipy import stats
from pyrocko import model
from pyrocko import cake
from pyrocko import pile
from pyrocko import util
from pyrocko.gui_util import PhaseMarker, Marker
from pyrocko import trace
import matplotlib.pyplot as plt
from autogain.autogain import PickPie
from mtspec import mtspec

from qtest.util import e2s, s2t, reset_events
from qtest.distance_point2line import Coupler, Filtrate

logger = logging.getLogger()

#fn_markers = 'picks_part.pf'
#channel = 'HHZ'
#fn_traces = '/media/usb/webnet/wern/rest'
#fileformat = 'mseed'
#fn_stations = '/data/meta/stations_sachsennetz.pf'

fn_markers = '/home/marius/josef_dd/hypodd_markers_josef.pf'
channel = 'SHZ'
fn_traces = '/media/usb/webnet/gse2/2008Oct'
fileformat = 'gse2'
fn_stations = '/data/meta/stations.pf'

outdir_suffix = 'all_less_strict'
window_length = 0.2
fn_coupler = 'coupled.pickl'
minimum_magnitude = 1.
maximum_magnitude = 2.8
min_travel_distance = 200.
threshold_pass_factor = 4.
max_mag_diff = 0.2
want_phase = 'P'
time_bandwidth = 5.
ntapers = 3
fmin = 50.
#fmax = 65.
fmax = 75.
min_cc = 0.7
min_rsquared = 0.75
#rmse_max = 0.5
rmse_max = 10000.
position = 0.9
#fn_events = '/home/marius/josef_dd/events_from_sebastian.pf'
fn_events = '/home/marius/josef_dd/events_from_sebastian_check_M1.pf'
fn_earthmodel = '../../models/earthmodel_malek_alexandrakis.nd'
# define stations to use
do_plot = False

white_list = [
    #('', 'KRC', ''),
    #('', 'KAC', ''),
    #('', 'POC', ''),
    #('', 'NKC', ''),
    ('', 'LBC', ''),
    #('SX', 'WERN', ''),
]

phases = [cake.PhaseDef(want_phase), cake.PhaseDef(want_phase.lower())]

velocity_model = cake.load_model(fn_earthmodel)
events = model.load_events(fn_events)
events = filter(lambda x: maximum_magnitude>=x.magnitude>=minimum_magnitude, events)

events_by_time = {}
for e in events:
    events_by_time[e.time] = e

sources = [e2s(e) for e in events]

stations = model.load_stations(fn_stations)
stations = filter(lambda x: x.nsl() in white_list, stations)
targets = [s2t(s, channel=channel) for s in stations]
markers = Marker.load_markers(fn_markers)
markers = filter(lambda x: isinstance(x, PhaseMarker), markers)
data_pile = pile.make_pile(fn_traces, fileformat=fileformat)
#data_pile.snuffle()

reset_events(markers, events)

pie = PickPie(markers=markers, mod=velocity_model, event2source=e2s, station2target=s2t)
pie.process_markers(want_phase, stations=stations, channel=channel)

# CREATE COUPLES
if True:
    coupler = Coupler()
    coupler.process(
        sources,
        targets,
        velocity_model,
        phases,
        check_relevance_by=data_pile,
        ignore_segments=False,
        dump_to=fn_coupler)
else:
    coupler = Coupler(Filtrate.load_pickle(fn_coupler))

candidates = coupler.filter_pairs(
    threshold_pass_factor,
    min_travel_distance,
    coupler.filtrate,
    max_mag_diff=max_mag_diff
)

counter = {}
for t in targets:
    counter[t.codes] = [0, 0]


slope_positive = 0
slope_negative = 0

events = {}
slopes = []
qs = []

outdir_prepend = '%s%s' %(white_list[0][1], outdir_suffix)
if not os.path.isdir(outdir_prepend):
    os.mkdir(outdir_prepend)
else:
    print(' DIRECTORY EXISTS...............!')

for icand, candidate in enumerate(candidates):
    print(icand+1, len(candidates))
    s1, s2, t, td, pd, totald, i1, ray = candidate
    for s in [s1, s2]:
        events[s.time] = events_by_time[s.time]


    selector = lambda x: (x.station, x.channel)==(t.codes[1], t.codes[3])
    group_selector = lambda x: t.codes[1] in x.stations
    # s2 should be closer, hence it should be less affected by Q
    onset1 = pie.t(want_phase, (s1, t.codes[:3]))
    onset2 = pie.t(want_phase, (s2, t.codes[:3]))
    if not onset1 or not onset2:
        continue

    if onset1 < 10000:
        tmin1 = s1.time + onset1
    else:
        tmin1 = onset1

    if onset2 < 10000:
        tmin2 = s2.time + onset2
    else:
        tmin2 = onset2

    # grab trace segment of source:
    tmin_group1 = tmin1-window_length*(1-position)
    tmin_group2 = tmin2-window_length*(1-position)
    tmax_group1  = tmin1+window_length*position
    tmax_group2  = tmin2+window_length*position
    if not data_pile.is_relevant(tmin_group1, tmax_group1, group_selector) or \
            not data_pile.is_relevant(tmin_group2, tmax_group2, group_selector):
        print 'irrelevant'
        continue

    # check cross correlation across all stations:


    try:
        tr1 = list(data_pile.chopper(
            trace_selector=selector,
            tmin=tmin_group1, tmax=tmax_group1))[0][0]

        tr2 = list(data_pile.chopper(
            trace_selector=selector,
            tmin=tmin_group2, tmax=tmax_group2))[0][0]

    except IndexError as e:
        logger.info(e)
        continue

    cc = trace.correlate(tr1, tr2, normalization='normal').max()[1]
    if cc < min_cc:
        print 'low cc', cc
        continue
    print 'cc', cc

    y1 = tr1.ydata
    y2 = tr2.ydata

    nsamples_want = min(len(y1), len(y2))
    #nsamples_want = window

    y1 = y1[:nsamples_want]
    y2 = y2[:nsamples_want]

    a1, f = mtspec(data=y1,
                   delta=tr1.deltat,
                   number_of_tapers=ntapers,
           time_bandwidth=time_bandwidth, nfft=trace.nextpow2(nsamples_want)*2)
           #time_bandwidth=time_bandwidth , nfft=trace.nextpow2(nsamples_want))

    a2, f = mtspec(data=y2, delta=tr2.deltat, number_of_tapers=ntapers,
           time_bandwidth=time_bandwidth, nfft=trace.nextpow2(nsamples_want)*2)
    ratio = num.log(num.sqrt(a1)/num.sqrt(a2))

    indx = num.intersect1d(num.where(f>=fmin), num.where(f<=fmax))
    f_selected = f[indx]

    ratio_selected = ratio[indx]
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        f_selected, ratio[indx])
    # RMS Error:

    RMSE = num.sqrt(num.sum((ratio[indx]-(intercept+f_selected*slope))**2)/float(len(f_selected)))
    print 'XXX  RMSE XXXX', RMSE
    if RMSE > rmse_max:
        print 'XXX  RMSE too large XXXX', RMSE
        continue

    if r_value**2 < min_rsquared:
        print 'low rsquared: %s', r_value**2
        continue

    print f_selected
    print slope
    if num.isnan(slope):
        print "Warning: slope was Nan: skipped"
        continue

    if do_plot:
        fig, axs = plt.subplots(4,1)
        axs[0].plot(y1)
        axs[0].plot(y2)
        axs[0].set_ylabel('A [count]')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_title('P Wave')

        axs[1].plot(f, a1)
        axs[1].plot(f, a2)
        axs[1].set_yscale('log')
        axs[1].set_xlabel('f [Hz]')

        axs[2].plot(f_selected, ratio_selected)
        axs[2].plot([fmin, fmax], [intercept+fmin*slope, intercept + fmax*slope])

        axs[2].set_title('log')

        ratio = num.log(num.sqrt(a1)/num.sqrt(a2))
        ratio_selected = ratio[indx]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            f[indx], ratio[indx])
        print slope, 'b'
        print 'rvalue', r_value
        print 'pvalue', p_value

        axs[3].plot(f_selected, ratio_selected)
        axs[3].plot([fmin, fmax], [intercept+fmin*slope, intercept + fmax*slope])
        axs[3].set_title('log(sqrt/sqrt), R=%s, cc=%s' % (r_value, cc))
        print '>0, <0:',  slope_positive, slope_negative
        print counter

        fig.savefig('%s%s/example_wave_spectra_%s.png' % (tr1.station, outdir_suffix, int(tr1.tmin)))
        #plt.show()
    if slope>0:
        counter[t.codes][0] += 1
    else:
        counter[t.codes][1] += 1
    slopes.append(slope)
    qs.append(slope/ray.t[-1])
    print slope, 'a'
print slopes
fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(slopes) #(, bins=21)
ax = fig.add_subplot(122)
ax.hist(qs) #(, bins=21)


fig.savefig('slope_histogram.png')
num.savetxt('%s%s/slopes.txt' % (white_list[0][1], outdir_suffix), num.array(slopes).T)
num.savetxt('%s%s/qs.txt' % (white_list[0][1], outdir_suffix), num.array(qs).T)
#model.dump_events(events.values(), 'tobepicked.pf')
#plt.show()
