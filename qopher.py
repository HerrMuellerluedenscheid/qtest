#!/usr/bin/env python3

import matplotlib as mpl
# mpl.use('Qt5Agg')
mpl.use('agg')

import logging
import numpy as num
import os
import sys
import pathlib
try:
    # python2
    import cPickle as pickle
except ImportError:
    # python3
    import pickle

from collections import defaultdict
from scipy import stats
from pyrocko import model
from pyrocko import cake
from pyrocko import pile
from pyrocko import util
from pyrocko.gui import snuffler
from pyrocko.gui.marker import PhaseMarker, Marker, associate_phases_to_events
from pyrocko import trace
import matplotlib.pyplot as plt
import mtspec
from autogain.autogain import PickPie

from qtest import config
from qtest.util import e2s, s2t, reset_events, get_spectrum
from qtest.distance_point2line import Coupler, Filtrate, filename_hash, fresnel_lambda

logger = logging.getLogger('qopher')

pjoin = os.path.join


n_smooth = 10   # samples
_smooth_taper = num.hanning(n_smooth)
_taper_sum = _smooth_taper.sum()

print("----Using %s points hanning smoothing for SNR in sepctral domain ----" % n_smooth)


def read_blacklist(fn):
    if fn is None:
        return []
    names = []
    with open(fn, 'r') as f:
        for line in f.readlines():
            names.append(line.split()[0])
    return names


def dump_qstats(qstats_dict, outdir):
    fn = pjoin(outdir, 'qstats.cpickle')
    with open(fn, 'wb') as f:
        pickle.dump(qstats_dict, f)


class Counter:
    def __init__(self, msg):
        self.msg = msg
        self.count = 0

    def __call__(self, info=''):
        self.count += 1
        logger.info("%s %s (%i)" % (self.msg, info, self.count))

    def __str__(self):
        return "%s: %i" % (self.msg, self.count)


__top_level_cache = {}


def run_qopher(config, snuffle=False):
   
    if not os.path.isdir(config.outdir):
        os.mkdir(config.outdir)

    fail_counter = {}
    fail_counter['no_onset'] = Counter('no onset')
    fail_counter['no_waveforms'] = Counter('no waveform data')
    fail_counter['no_data'] = Counter('WARN: ydata is None')
    fail_counter['low_snr'] = Counter('Low SNR')
    fail_counter['cc'] = Counter('Low CC')
    fail_counter['rmse'] = Counter('RMSE')
    fail_counter['IndexError'] = Counter('IndexError')
    fail_counter['rsquared'] = Counter('Low rsquared')
    fail_counter['slope'] = Counter('WARN: slope is NAN')
    fail_counter['bandwidth'] = Counter('WARN: Bandwidth to small')
    fail_counter['frequency_content_low'] = Counter('Frequency Band')

    data_pile = __top_level_cache.get((config.traces, config.file_format), False)
    if not data_pile:
        data_pile = config.get_pile()
        # data_pile = pile.make_pile(config.traces, fileformat=config.file_format)
        __top_level_cache[(config.traces, config.file_format)] = data_pile
    
    ptmax = data_pile.tmax
    ptmin = data_pile.tmin

    whitelist = config.whitelist or list(data_pile.stations.keys())

    phases = [
        # cake.PhaseDef(config.want_phase.upper()),
        cake.PhaseDef(config.want_phase.lower())]
    
    events_load = model.load_events(config.events)
    markers_load = PhaseMarker.load_markers(config.markers)
    reset_events(markers_load, events_load)
    stations = model.load_stations(config.stations)
    stations = [s for s in stations if '.'.join(s.nsl()) in whitelist]

    events = []
    blacklist_events = read_blacklist(config.blacklist_events_fn)
    for e in events_load:
        if not e:
            nskipped += 1
            continue
        if (config.tstart and e.time < util.stt(config.tstart)) or \
                (config.tstop and e.time > util.tts(config.tstop)) or \
                (config.mag_min and e.magnitude < config.mag_min):
            continue

        events.append(e)
    nskipped = 0
    logger.warn('nskipped because event was none: %s' % nskipped)
    sources = [e2s(e) for e in events]

    if len(stations) == 0:
        raise Exception('No stations excepted by whitelist')
    velocity_model = cake.load_model(config.earthmodel)
    velocity_fresnel = getattr(
        velocity_model.material(
            z=num.mean([s.depth for s in sources])),
        {'P': 'vp', 'S': 'vs'}[config.want_phase.upper()])

    for station in stations:
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

        # events = [e for e in events if ptmin<e.time<ptmax]
        # etimes = set([m.get_event().time for m in markers])

        targets = [s2t(station)]

        if snuffle:
            snuffler.snuffle(data_pile, events=events, stations=stations,
                           markers=markers)
            sys.exit(0)

        pie = PickPie(
            markers=markers,
            mod=velocity_model,
            event2source=e2s,
            station2target=s2t)

        # subtract origin times from markers
        pie.process_markers(
            config.want_phase.upper(), stations=[station], channel=config.channel)

        fn_cache = os.path.join(
            config.fn_couples, 'coupler_fix_' + filename_hash(
                sources, targets, config.earthmodel, phases))
        logger.info('phase cache filename: %s' % fn_cache)

        try:
            coupler = __top_level_cache.get(fn_cache, False)
            if not coupler:
                coupler = Coupler(Filtrate.load_pickle(fn_cache))
                __top_level_cache[fn_cache] = coupler
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
                ignore_segments=False,
                fn_cache=fn_cache)

        candidates = coupler.filter_pairs(
            config.traversing_ratio,
            config.traversing_distance_min,
            coupler.filtrate,
            max_mag_diff=config.mag_delta_max,
        )

        if len(candidates) == 0:
            logger.warn('len(filtered) == 0!')
            return

        pwrs = defaultdict(list)

        def check_lqt(trs, trs_rotated):
            '''Helper routine to verify power ratios.'''
            def pwr(_tr_):
                return num.sqrt(num.sum(tr.ydata**2))
            for tr in trs:
                pwrs[tr.channel].append(num.sqrt(num.sum(tr.ydata**2)))

        def plot_lqt():
            avg_t = num.mean(pwrs['T'])
            for channel_id, vals in pwrs.items():
                print(channel_id, num.mean(num.array(vals)/avg_t))

        counter = {}
        for t in targets:
            counter[t.codes] = [0, 0]

        qs = []
        cc = None
        no_waveforms = []
        ncandidates = len(candidates)
        figname = None

        needs_rotation = config.channel in 'LQT'

        all_stats = []
        for icand, candidate in enumerate(candidates):
            a2_noise = None
            a1_noise = None

            print('... %1.1f' % ((icand/float(ncandidates)) * 100.))
            s1, s2, trgt, td, pd, totald, incidence, travel_time_segment = candidate

            if s1.name in blacklist_events or s2.name in blacklist_events:
                print('---------------------------EVENT BLACKLISTED')
                continue

            fmax_lim = config.fmax_lim * config.fmax_factor

            if config.max_t_separation is not None:
                if abs(s1.time-s2.time) > config.max_t_separation:
                    print('---------------------------temporal speparation')
                    continue

            if config.lat_min is not None:
                if s1.lat < config.lat_min or s2.lat < config.lat_min:
                    print('---------------------------Latmin')
                    continue

            if config.use_fresnel:
                fmax_lim = min(
                    fmax_lim,
                    velocity_fresnel/fresnel_lambda(totald, td, pd))

            fmin_lim = config.fmin_lim
            if fmax_lim - fmin_lim < config.min_bandwidth:
                fail_counter['bandwidth']('%s - %s ' % (fmin_lim, fmax_lim))
                continue

            if config.save_stats:
                qstats = {'event1': s1, 'event2': s2, 'target': trgt, 'td': td, 'pd': pd,
                         'totald': totald, 'incidence': incidence}

            phase_keys = [(config.want_phase.upper(), (s, trgt.codes[:3])) for s in [s1, s2]]
            if any([pk in no_waveforms for pk in phase_keys]):
                fail_counter['no_waveforms'](' %s %s ' % (util.tts(s1.time),
                                                          util.tts(s2.time)))
                continue

            if needs_rotation:
                selector = lambda x: x.station==trgt.codes[1]
                in_channels = ['SHZ', 'SHN', 'SHE']
            else:
                selector = lambda x: (x.station, x.channel[-1])==(trgt.codes[1], trgt.codes[3][-1])

            group_selector = lambda x: trgt.codes[1] in x.stations

            trs = []
            for source in [s1, s2]:
                # s2 should be closer, hence it should be less affected by Q
                phase_key = (config.want_phase.upper(), (source, trgt.codes[:3]))
                tmin = pie.t(*phase_key)
                if not tmin:
                    fail_counter['no_onset']("%s %s" % (util.tts(source.time), str(trgt.codes[:3])))
                    break

                if s2.depth > s1.depth:
                    raise Exception('z2>z1')

                if config.depth_1_min and s1.depth<config.depth_1_min:
                    continue

                if tmin < 10000.:
                    tmin = source.time + tmin

                # grab trace segment of source:
                tmin = tmin - config.window_length * (1.-config.position)
                tmax = tmin + config.window_length
                tmax_noise = tmin - config.noise_window_shift
                tmin_noise = tmax_noise - config.window_length
                if not data_pile.is_relevant(tmin, tmax, group_selector):
                    fail_counter['no_waveforms']()
                    no_waveforms.append(phase_key)
                    break

                try:

                    tr1 = next(data_pile.chopper(trace_selector=selector,
                        tmin=tmin, tmax=tmax), None)
                    tr1_noise = next(data_pile.chopper(trace_selector=selector,
                        tmin=tmin_noise, tmax=tmax_noise), None)

                    if tr1 is None or tr1_noise is None:
                        fail_counter['no_waveforms']()
                        continue

                    if needs_rotation:
                        backazimuth = source.azibazi_to(trgt)[1]
                        trs_rot = trace.rotate_to_lqt(tr1, backazimuth, incidence, in_channels)
                        trs_rot_noise = trace.rotate_to_lqt(tr1_noise, backazimuth, incidence, in_channels)
                        tr1 = next((tr for tr in trs_rot if
                                    tr.channel==config.channel), False)
                        tr1_noise = next((tr for tr in trs_rot_noise if
                                          tr.channel==config.channel), False)

                        check_lqt(trs_rot, tr1)
                    else:
                        tr1 = tr1[0]
                        tr1_noise = tr1_noise[0]

                    if not tr1 or not tr1_noise:
                        print('XXXX XXXX')
                        continue

                    if tr1.ydata is not None:
                        tr1.ydata = tr1.ydata.astype(num.float)
                        tr1_noise.ydata = tr1_noise.ydata.astype(num.float)
                    else:
                        fail_counter['no_data']()
                        continue

                    if not tr1 or not tr1_noise:
                        fail_counter['no_waveforms']()
                        continue

                    tshift = -tr1.tmin
                    tr1.shift(tshift)
                    tr1_noise.shift(tshift)
                    trs.append((tr1, tr1_noise))
                except IndexError as e:
                    fail_counter['IndexError']()
                    break

            if len(trs) != 2:
                continue

            (tr1, tr1_noise), (tr2, tr2_noise) = trs
            if config.cc_min or config.save_stats:
                _tr1 = tr1.copy()
                _tr2 = tr2.copy()
                _tr1.highpass(4, fmin_lim)
                _tr2.highpass(4, fmin_lim)
                _tr1.lowpass(4, fmax_lim)
                _tr2.lowpass(4, fmax_lim)
                _tr1.ydata /= _tr1.ydata.max()
                _tr2.ydata /= _tr2.ydata.max()
                cc = trace.correlate(
                    _tr1, _tr2, mode=config.cc_mode,
                    normalization=config.cc_normalization).max()[1]
                if config.cc_min and cc < config.cc_min:
                    fail_counter['cc'](cc)
                    continue

            nsamples_want = min(tr1.data_len(), tr2.data_len())

            y1 = tr1.ydata[:nsamples_want]
            y2 = tr2.ydata[:nsamples_want]
            y1_noise = tr1_noise.ydata[:nsamples_want]
            y2_noise = tr2_noise.ydata[:nsamples_want]

            f1, a1 = get_spectrum(y1, tr1.deltat, config, adaptive=config.adaptive)
            f2, a2 = get_spectrum(y2, tr2.deltat, config, adaptive=config.adaptive)
            if config.snr is not None or config.save_stats:
                f1_noise, a1_noise = get_spectrum(y1_noise, tr1.deltat, config,
                    adaptive=config.adaptive)
                f2_noise, a2_noise = get_spectrum(y2_noise, tr2.deltat, config,
                    adaptive=config.adaptive)
                a1_smooth = num.convolve(a1/_taper_sum, _smooth_taper,
                                         mode='same')
                a2_smooth = num.convolve(a2/_taper_sum, _smooth_taper,
                                         mode='same')
                a1_noise = num.convolve(a1_noise/_taper_sum, _smooth_taper,
                                         mode='same')
                a2_noise = num.convolve(a2_noise/_taper_sum, _smooth_taper,
                                         mode='same')
                # a1_smooth = a1
                # a2_smooth = a2
                idx1 = num.where(num.logical_and(f1_noise>=fmin_lim, f1_noise<=fmax_lim))
                idx2 = num.where(num.logical_and(f2_noise>=fmin_lim, f2_noise<=fmax_lim))
                snr1 = min(a1_smooth[idx1]/a1_noise[idx1])
                snr2 = min(a2_smooth[idx2]/a2_noise[idx2])
                if  snr1 < config.snr or snr2 < config.snr:
                    fail_counter['low_snr']()
                    continue
            f1, a1 = get_spectrum(y1, tr1.deltat, config, normalize=False,
                                  adaptive=config.adaptive)
            f2, a2 = get_spectrum(y2, tr2.deltat, config, normalize=False,
                                  adaptive=config.adaptive)
            ratio = num.log(a1 / a2)

            if config.use_deconvolution:
                # ratio_trace = trace.deconvolve(tr1, tr2, waterlevel=0.05)
                # f1, ratio = get_spectrum(ratio_trace.get_ydata(), ratio_trace.deltat,
                #                     config, normalize=False, adaptive=config.adaptive)
                # ratio = num.log(ratio)
                # print(tr1.data_len(), tr2.data_len())
                y1 = tr1.get_ydata()
                y2 = tr2.get_ydata()
                n = min(tr1.data_len(), tr2.data_len())
                decon = mtspec.mt_deconvolve(y1[:n], y2[:n],
                                            tr1.deltat,
                                             time_bandwidth=config.time_bandwidth,
                                             # number_of_tapers=config.ntapers,
                                             number_of_tapers=5,
                                             weights='constant', demean=False)
                y_decon = decon['deconvolved']
                a, f = mtspec.mtspec(y_decon, tr1.deltat,
                              time_bandwidth=config.time_bandwidth)
                              #number_of_tapers=config.nta
                # If that works, a1 should have been num.sqrt(a1) and same for
                # a2...!
                ratio = num.log(a)
                # ratio = num.log(decon['spectral_ratio'])
                # print(decon.keys())

                f1 = decon['frequencies']
                # print(ratio)
                # ratio = num.log(ratio)
                # print(ratio)

            indx = num.intersect1d(
                num.where(f1>=fmin_lim),
                num.where(f1<=fmax_lim))

            if len(indx) == 0:
                fail_counter['frequency_content_low']()
                continue

            f_selected = f1[indx]
            ratio_selected = ratio[indx]

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                f_selected, ratio[indx])
            slope = slope * -1.
            if config.rmse_max or config.save_stats:
                # RMS Error:
                RMSE = num.sqrt(num.sum((ratio[indx]-(intercept+f_selected*slope))**2)/float(len(f_selected)))
                if config.rmse_max and RMSE > config.rmse_max:
                    fail_counter['rmse']()
                    continue

            if config.rsquared_min or config.save_stats:
                if config.rsquared_min and r_value**2 < config.rsquared_min:
                    fail_counter['rsquared'](r_value**2)
                    continue

            if num.isnan(slope):
                fail_counter['slope']()
                continue

            dir_png = pjoin(config.outdir, station.station, 'trace-plots')
            if not os.path.exists(dir_png):
                pathlib.Path(dir_png).mkdir(parents=True, exist_ok=True)

            config.dump(filename=pjoin(config.outdir, station.station, 'config.yaml'))

            if config.plot:
                figname = pjoin(dir_png, 'example_wave_spectra_%s.png' % icand)
                fig, axs = plt.subplots(3, 1)
                axs[0].plot(tr1.get_xdata(), tr1.get_ydata())
                axs[0].plot(tr2.get_xdata(), tr2.get_ydata())
                axs[0].plot(tr1_noise.get_xdata(), tr1_noise.get_ydata(), color='grey')
                axs[0].plot(tr2_noise.get_xdata(), tr2_noise.get_ydata(), color='grey')
                axs[0].set_ylabel('A [count]')
                axs[0].set_xlabel('Time [s]')
                stext = '%s %s' % (s1.name, s2.name)
                axs[0].set_title('Channel: %s, %s' % (tr1.channel, stext))

                if cc is not None:
                    axs[0].text(
                        0., 0.9, 'q = %1.2f | cc = %1.2f ' %
                        (slope/travel_time_segment, cc),
                        transform=axs[0].transAxes
                    )

                axs[1].plot(f1, a1)
                axs[1].plot(f2, a2)
                if a1_noise is not None:
                    axs[1].plot(f1_noise, a1_noise)
                    axs[1].plot(f2_noise, a2_noise)
                axs[1].set_yscale('log')
                axs[1].set_xlabel('f [Hz]')

                axs[2].plot(f_selected, ratio_selected)
                axs[2].plot([fmin_lim, fmax_lim],
                            [intercept+fmin_lim*slope, intercept + fmax_lim*slope])

                axs[2].set_title('log')

                print("slope %s, rvalue %s, pvalue %s" % (slope, r_value, p_value))

                fig.savefig(figname)

            if config.save_stats:
                qstats['rsquared'] = r_value ** 2
                qstats['rmse'] = RMSE
                qstats['snr1'] = snr1
                qstats['snr2'] = snr2
                qstats['cc'] = cc
                qstats['q'] = slope/travel_time_segment
                qstats['figname'] = figname
                all_stats.append(qstats)

            if slope>0:
                counter[trgt.codes][0] += 1
            else:
                counter[trgt.codes][1] += 1
            # qs.append(slope/ray.t[-1])

            Q = num.pi * travel_time_segment / slope
            qs.append(1./Q)
            # qs.append(slope/travel_time_segment)
            tr1.drop_data()
            tr2.drop_data()
            tr1_noise.drop_data()
            tr2_noise.drop_data()
            del(trs)

            print("slope %s" % slope)
        outdir = pjoin(config.outdir, station.station)
        if config.save_stats:
            dump_qstats(all_stats, outdir)

        print(qs)
        status_str = ''
        for k, v in fail_counter.items():
            status_str += "%s\n" % v
   
        if not os.path.isdir(pjoin(config.outdir, station.station)):
            os.mkdir(pjoin(config.outdir, station.station))

        with open(pjoin(config.outdir, station.station, 'status.txt'), 'w') as f:
            f.write(status_str)
            f.write('median 1./q = %s\n' % num.median(qs))
            f.write('Nsamples = %s\n' % len(qs))
            # f.write('Npositive = %s\n' % len(qs))

        plot_lqt() 
        num.savetxt(pjoin(outdir, 'qs_inv.txt'), num.array(qs).T)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(qs, bins=121)

        fig.savefig(pjoin(outdir, 'slope_histogram.png'))
        print('results saved at: %s ' % outdir)


if __name__ == '__main__':
    from qtest import config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--snuffle', action='store_true')
    args = parser.parse_args()
    c = config.QConfig.load(filename=args.config)
    run_qopher(c, args.snuffle)
