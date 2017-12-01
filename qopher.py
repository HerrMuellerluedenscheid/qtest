import matplotlib as mpl
mpl.use('Agg')

import logging
import numpy as num
import os
import sys
from scipy import stats
from pyrocko import model
from pyrocko import cake
from pyrocko import pile
from pyrocko import util
from pyrocko.gui.util import PhaseMarker, Marker
from pyrocko import trace
import matplotlib.pyplot as plt
from autogain.autogain import PickPie
from mtspec import mtspec

from qtest import config
from qtest.util import e2s, s2t, reset_events
from qtest.distance_point2line import Coupler, Filtrate, filename_hash

logger = logging.getLogger('qopher')

pjoin = os.path.join

class SNRException(Exception):
    pass


def get_spectrum(ydata, deltat, config):
    ''' Return the spectrum on *ydata*, considering the *snr* stored in
    config
    '''
    #isplit = int(len(ydata) / 2)
    a, f = mtspec(data=ydata, delta=deltat, number_of_tapers=config.ntapers,
        time_bandwidth=config.time_bandwidth, nfft=trace.nextpow2(len(ydata))*2)

    #  plt.figure()
    #  plt.plot(ydata[isplit:isplit*2])
    #  plt.plot(ydata[0:isplit])
    #  plt.show()
    return f, a


class Counter:
    def __init__(self, msg):
        self.msg = msg
        self.count = 0

    def __call__(self, info=''):
        self.count += 1
        logger.info("%s %s (%i)" % (self.msg, info, self.count))

    def __str__(self):
        return "%s: %i" % (self.msg, self.count)


fail_counter = {}
fail_counter['no_onset'] = Counter('no onset')
fail_counter['no_waveforms'] = Counter('no waveform data')
fail_counter['low_snr'] = Counter('Low SNR')
fail_counter['cc'] = Counter('Low CC')
fail_counter['rmse'] = Counter('RMSE')
fail_counter['IndexError'] = Counter('IndexError')
fail_counter['rsquared'] = Counter('Low rsquared')
fail_counter['slope'] = Counter('WARN: slope is NAN')


def run_qopher(config):
    if not os.path.isdir(config.outdir):
        os.mkdir(config.outdir)

    dir_png = pjoin(config.outdir, 'trace-plots')
    if not os.path.exists(dir_png):
        os.mkdir(dir_png)

    config.dump(filename=pjoin(config.outdir, 'config.yaml'))

    phases = [cake.PhaseDef(config.want_phase), cake.PhaseDef(config.want_phase.lower())]

    velocity_model = cake.load_model(config.earthmodel)
    events = model.load_events(config.events)
    events = filter(lambda x: config.mag_max>=x.magnitude>=config.mag_min, events)
    if config.tstart:
        events = filter(lambda x: x.time > util.stt(config.tstart), events)
    if config.tstop:
        events = filter(lambda x: x.time < util.stt(config.tstop), events)
    events_by_time = {}
    for e in events:
        events_by_time[e.time] = e

    sources = [e2s(e) for e in events]

    stations = model.load_stations(config.stations)
    if config.whitelist:
        #stations = filter(lambda x: '.'.join(x.nsl()) in config.whitelist, stations)
        stations = [s for s in stations if '.'.join(s.nsl()) in config.whitelist]
    if len(stations) == 0:
        raise Exception('No stations excepted by whitelist')

    targets = [s2t(s, channel=config.channel) for s in stations]
    markers = Marker.load_markers(config.markers)
    markers = filter(lambda x: isinstance(x, PhaseMarker), markers)
    data_pile = pile.make_pile(config.traces, fileformat=config.file_format)

    reset_events(markers, events)

    pie = PickPie(
        markers=markers,
        mod=velocity_model,
        event2source=e2s,
        station2target=s2t)

    pie.process_markers(
        config.want_phase, stations=stations, channel=config.channel)

    fn_cache = os.path.join(
        config.fn_couples, 'coupler_' + filename_hash(
            sources, targets, config.earthmodel, phases))
    try:
        coupler = Coupler(Filtrate.load_pickle(fn_cache))
    except IOError:
        if not os.path.isdir(config.fn_couples):
            os.mkdir(config.fn_couples)
        coupler = Coupler()
        coupler.process(
            sources,
            targets,
            velocity_model,
            phases,
            check_relevance_by=data_pile,
            ignore_segments=False,
            fn_cache=fn_cache)

    candidates = coupler.filter_pairs(
        config.traversing_ratio,
        config.traversing_distance_min,
        coupler.filtrate,
        max_mag_diff=config.mag_delta_max
    )

    counter = {}
    for t in targets:
        counter[t.codes] = [0, 0]

    slope_positive = 0
    slope_negative = 0

    slopes = []
    qs = []
    cc = None
    no_waveforms = []
    ncandidates = len(candidates)
    for icand, candidate in enumerate(candidates):
        print('... %1.1f' % ((icand/float(ncandidates)) * 100.))
        s1, s2, t, td, pd, totald, i1, travel_time_segment = candidate

        phase_keys = [(config.want_phase, (s, t.codes[:3])) for s in [s1, s2]]
        if any([pk in no_waveforms for pk in phase_keys]):
            continue

        selector = lambda x: (x.station, x.channel)==(t.codes[1], t.codes[3])
        group_selector = lambda x: t.codes[1] in x.stations

        trs = []
        for source in [s1, s2]:
            # s2 should be closer, hence it should be less affected by Q
            phase_key = (config.want_phase, (source, t.codes[:3]))
            tmin1 = pie.t(*phase_key)
            if not tmin1:
                fail_counter['no_onset']()
                break 

            if tmin1 < 10000.:
                tmin1 = source.time + tmin1

            # grab trace segment of source:
            tmin_group1 = tmin1 - config.window_length * (1.-config.position)
            tmax_group1 = tmin1 + config.window_length * config.position
            tmin_noise1 = tmin_group1 - config.window_length - \
                config.noise_window_shift
            tmax_noise1 = tmin_noise1 + config.window_length
            if not data_pile.is_relevant(tmin_group1, tmax_group1, group_selector):
                fail_counter['no_waveforms']()
                no_waveforms.append(phase_key)
                break 

            try:
                tr1 = list(data_pile.chopper(
                    trace_selector=selector,
                    tmin=tmin_group1, tmax=tmax_group1))[0][0]
                tr1_noise = list(data_pile.chopper(
                    trace_selector=selector,
                    tmin=tmin_noise1, tmax=tmax_noise1))[0][0]

                tr1.shift(-tr1.tmin)
                tr1_noise.shift(-tr1.tmin)
                print(tmin_group1, tmin_noise1)
                print(tmax_group1, tmax_noise1)


                trs.append((tr1, tr1_noise))
            except IndexError as e:
                fail_counter['IndexError']()
                break 

        if len(trs) != 2:
            continue

        (tr1, tr1_noise), (tr2, tr2_noise) = trs
        if config.cc_min:
            cc = trace.correlate(tr1, tr2, normalization='normal').max()[1]
            if cc < config.cc_min:
                fail_counter['cc'](cc)
                continue

        nsamples_want = min(tr1.data_len(), tr2.data_len())

        y1 = tr1.ydata[:nsamples_want]
        y2 = tr2.ydata[:nsamples_want]
        y1_noise = tr1_noise.ydata[:nsamples_want]
        y2_noise = tr2_noise.ydata[:nsamples_want]

        f1, a1 = get_spectrum(y1, tr1.deltat, config)
        f2, a2 = get_spectrum(y2, tr2.deltat, config)
        if config.snr:
            a1_noise, _ = get_spectrum(y1_noise, tr1.deltat, config)
            a2_noise, _ = get_spectrum(y2_noise, tr2.deltat, config)
            if min(a1/a1_noise) < config.snr or min(a2/a2_noise) < config.snr:
                fail_counter['low_snr'](e)
                continue

        ratio = num.log(num.sqrt(a1)/num.sqrt(a2))

        indx = num.intersect1d(
            num.where(f1>=config.fmin_lim),
            num.where(f1<=config.fmax_lim))

        f_selected = f1[indx]

        ratio_selected = ratio[indx]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            f_selected, ratio[indx])
        if config.rmse_max:
            # RMS Error:
            RMSE = num.sqrt(num.sum((ratio[indx]-(intercept+f_selected*slope))**2)/float(len(f_selected)))
            if RMSE > config.rmse_max:
                fail_counter['rmse']()
                continue

        if config.rsquared_min:
            if r_value**2 < config.rsquared_min:
                fail_counter['rsquared'](r_value**2)
                continue

        if num.isnan(slope):
            fail_counter['slope']()
            continue

        if config.plot:
            fig, axs = plt.subplots(3,1)
            axs[0].plot(tr1.get_xdata(), tr1.get_ydata())
            axs[0].plot(tr2.get_xdata(), tr2.get_ydata())
            axs[0].plot(tr1_noise.get_xdata(), tr1_noise.get_ydata(), color='grey')
            axs[0].plot(tr2_noise.get_xdata(), tr2_noise.get_ydata(), color='grey')
            axs[0].set_ylabel('A [count]')
            axs[0].set_xlabel('Time [s]')
            axs[0].set_title('P Wave')

            axs[1].plot(f1, a1)
            axs[1].plot(f2, a2)
            axs[1].set_yscale('log')
            axs[1].set_xlabel('f [Hz]')

            axs[2].plot(f_selected, ratio_selected)
            axs[2].plot([config.fmin_lim, config.fmax_lim],
                        [intercept+config.fmin_lim*slope, intercept + config.fmax_lim*slope])

            axs[2].set_title('log')

            #ratio = num.log(num.sqrt(a1)/num.sqrt(a2))
            #ratio_selected = ratio[indx]
            #slope, intercept, r_value, p_value, std_err = stats.linregress(
            #    f1[indx], ratio[indx])
            print("slope %s, rvalue %s, pvalue %s" % (slope, r_value, p_value))

            # axs[3].plot(f_selected, ratio_selected)
            # axs[3].plot([config.fmin_lim, config.fmax_lim], [intercept+config.fmin_lim*slope, intercept + config.fmax_lim*slope])
            # axs[3].set_title('log(sqrt/sqrt), R=%s, cc=%s' % (r_value, cc))
            # print('>0, <0: %s, %s' % (slope_positive, slope_negative))
            print(dir_png)
            fig.savefig(pjoin(dir_png, 'example_wave_spectra_%s.png' % icand))
            #fig.close()

        if slope>0:
            counter[t.codes][0] += 1
        else:
            counter[t.codes][1] += 1
        # qs.append(slope/ray.t[-1])
        qs.append(slope/travel_time_segment)
        
        print("slope %s" % slope)

    print(qs)
    status_str = ''
    for k, v in fail_counter.items():
        status_str += "%s\n" % v
    
    with open(pjoin(config.outdir, 'status.txt'), 'w') as f:
        f.write(status_str)
        f.write('median 1./q = %s\n' % num.median(qs))
        f.write('Nsamples = %s\n' % len(qs))
        # f.write('Npositive = %s\n' % len(qs))

    num.savetxt(pjoin(config.outdir, 'qs_inv.txt'), num.array(qs).T)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(qs, bins=41)

    fig.savefig(pjoin(config.outdir, 'slope_histogram.png'))


if __name__ == '__main__':
    from qtest import config
    c = config.QConfig.load(filename=sys.argv[1])
    run_qopher(c)
