from scipy.stats import linregress
import numpy as num
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import mtspec
from pyrocko.gui.marker import load_markers, PhaseMarker, associate_phases_to_events
from pyrocko.pile import make_pile
from pyrocko.model.station import Station, dump_stations
from pyrocko import trace
import pyrocko.orthodrome as ortho


dir_data = '/data/vogtland2017/mseed'
fn_markers = '/data/vogtland2017/picked.pf'

markers = load_markers(fn_markers)

dpile = make_pile(dir_data)

tmin_shift = -0.2
tmax_shift = 0.5

# has to be negative:
tnoise_offset = 0.

twindow = tmax_shift - tmin_shift

want_channel = '001'
show_regression = False
# check:
time_bandwidth = 5
ntapers = 3
n_smooth = 40   # samples

smooth_taper = num.hanning(n_smooth)

stat = Station(lat=50.3591, lon=12.3855, elevation=714.0,
    network='XX', station='AACB', location='01')

# dump_stations([stat], '/data/vogtland2017/stations.pf')

def get_spectrum(tr, time_bandwidth, ntapers):
    ''' Return the spectrum on *ydata*, considering the *snr* stored in
    config
    '''
    ydata = tr.get_ydata()
    a, f = mtspec.mtspec(data=ydata, delta=tr.deltat, number_of_tapers=ntapers,
        time_bandwidth=time_bandwidth, nfft=trace.nextpow2(len(ydata))*2)

    return f, a

results = []

associate_phases_to_events(markers)

for m in markers:
    if not isinstance(m, PhaseMarker):
        continue

    trace_selector = lambda tr: m.match_nsl(tr.nslc_id[:3])

    chopped, used_files = dpile.chop(
        tmin=m.tmin+tmin_shift,
        tmax=m.tmax+tmax_shift,
        trace_selector=trace_selector)
    
    nchopped, nused_files = dpile.chop(
        tmin=m.tmin+tmin_shift-twindow+tnoise_offset,
        tmax=m.tmax+tmax_shift-twindow+tnoise_offset,
        trace_selector=trace_selector)

    event = m.get_event()
    magnitude = event.magnitude

    cutoffs = {}
    for i in range(len(chopped)):
        tr = chopped[i]
        trn = nchopped[i]

        assert (tr.nslc_id == trn.nslc_id)

        f, a = get_spectrum(tr, time_bandwidth, ntapers)
        fn, an = get_spectrum(trn, time_bandwidth, ntapers)

        a = num.convolve(a/smooth_taper.sum(), smooth_taper, mode='same')
        an = num.convolve(an/smooth_taper.sum(), smooth_taper, mode='same')
        snr = a / an

        i = num.where(num.logical_and(snr <= 2, f > 50.))[0]
        f_cutoff = f[min(i)]

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.plot(f, snr)

        # ax = fig.add_subplot(122)
        # ax.plot(f, a)
        # ax.plot(fn, an)
        # ax.axvline(f_cutoff)
        # ax.set_yscale('log')
        # ax.set_title(tr.channel)
        # plt.show()
        cutoffs[tr.channel] = f_cutoff
        distance = ortho.distance_accurate50m(stat, event)

    results.append((magnitude, cutoffs))


def filter_cutoffs(cutoff_dict):
    return cutoff_dict[want_channel]


fig = plt.figure()
ax = fig.add_subplot(111)

mags = num.zeros(len(results))
f_cutoffs = num.zeros(len(results))
for i, (mag, f_cutoff) in enumerate(results):
    mags[i] = mag
    f_cutoffs[i] = filter_cutoffs(f_cutoff)

if show_regression:
    r = linregress(mags, f_cutoffs)
    regress_mags = num.linspace(min(mags), max(mags), 2)
    ax.plot(regress_mags, r[1] + regress_mags*r[0])

ax.plot(mags, f_cutoffs, 'o', c='black')

ax.axhline(85, color='blue')
ax.text(
    max(mags), 85., "WEBNET cutoff frequency", color='blue',
    verticalalignment='bottom', horizontalalignment='right')
ax.set_ylabel('frequency [Hz]')
ax.set_xlabel('magnitude')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show() 