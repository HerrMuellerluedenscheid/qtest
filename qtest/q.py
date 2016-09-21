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
from pyrocko.gf import Target, LocalEngine
from pyrocko import orthodrome
from pyrocko.gui_util import PhaseMarker
from pyrocko import util
from pyrocko import cake, model
from pyrocko import pile
from pyrocko import moment_tensor
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import signal, interpolate
from qtest.micro_engine import DataTracer, Tracer, Builder
from qtest.micro_engine import Noise, Chopper, DDContainer
from qtest.micro_engine import associate_responses, UniformTTPerturbation
from autogain.autogain import PhasePie, PickPie
from distance_point2line import Coupler, Animator, Filtrate, fresnel_lambda
from util import Magnitude2Window, Magnitude2fmin, fmin_by_magnitude
from util import e2extendeds, e2s, s2t


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
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


def plot_locations(items, use_effective_latlon=False):
    f = plt.figure(figsize=(3,3))
    ax = f.add_subplot(111)
    lats = []
    lons = []
    for item in items:
        ax.plot(item.lon, item.lat, 'og')
        if use_effective_latlon:
            lat, lon = item.effective_latlon
        else:
            lat = item.lat
            lon = item.lon
        lats.append(lat)
        lons.append(lon)

    ax.plot(num.mean(lons), num.mean(lats), 'xg')
    ax.set_title('locations')
    ax.set_xlabel('lon', size=7)
    ax.set_ylabel('lat', size=7)

    y_range = num.max(lats)-num.min(lats)
    x_range = num.max(lons)-num.min(lons)
    ax.set_ylim([num.min(lats)-0.05*y_range, num.max(lats)+0.05*y_range])
    ax.set_xlim([num.min(lons)-0.05*x_range, num.max(lons)+0.05*x_range])
    f.savefig("locations_plot.png", dpi=160., bbox_inches='tight', pad_inches=0.01)


def intersect_all(*args):
    for i, l in enumerate(args):
        if i==0:
            inters = num.intersect1d(l, args[1])
        else:
            try:
                inters = num.intersect1d(inters, args[i+1])
            except IndexError:
                return inters


def extract(fxfy, upper_lim=0., lower_lim=99999.):
    indx = num.where(num.logical_and(fxfy[0]>=upper_lim, fxfy[0]<=lower_lim))
    indx = num.array(indx)
    return fxfy[:, indx].reshape(2, len(indx.T))


def exp_fit(x, y, m):
    return num.exp(y - m*x)

def iter_chopper(tr, tinc=None, tpad=0.):
    iwin = 0
    tmin = tr.tmin + tpad
    tmax = tr.tmax - tpad
    if tinc is None:
        tinc = tmax - tmin
    while True:
        wmin, wmax = tmin+iwin*tinc, min(tmin+(iwin+1)* tinc, tmax)
        eps = tinc*1e-6
        if wmin >= tmax-eps: break
        ytr = tr.chop(tmin=wmin-tpad, tmax=wmax+tpad, include_last=False, inplace=False, want_incomplete=False)
        iwin += 1
        yield ytr

_taperer = trace.CosFader(xfrac=0.15)


cached_coefficients = {}
def _get_cached_filter_coefs(order, corners, btype):
    '''from pyrocko trace module'''
    ck = (order, tuple(corners), btype)
    if ck not in cached_coefficients:
        if len(corners) == 0:
            cached_coefficients[ck] = signal.butter(order, corners[0], btype=btype)
        else:
            cached_coefficients[ck] = signal.butter(order, corners, btype=btype)

    return cached_coefficients[ck]


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


class Spectra(DDContainer):
    def __init__(self, *args, **kwargs):
        DDContainer.__init__(self, args, kwargs)
        self.fit_function = exp_fit
        self.spectra = []

    def set_fit_function(self, func):
        self.fit_function = func

    def plot_all(self, ax=None, colors='rb', alpha=1., legend=True):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        count = 0
        i_fail = 0
        i = 0
        for tracer, fxfy in self.spectra:
            if fxfy is None:
                ax.text(0.01, 0.90+i_fail, 'no data', transform=ax.transAxes, verticalalignment='top')
                i_fail -= 0.1
                continue
            fx, fy = num.vsplit(fxfy, 2)
            if colors != 'rb':
                color = colors[tracer]
            else:
                color = colors[count]
            ax.plot(fx.T[1:], fy.T[1:], label=tracer.label(), color=color, alpha=alpha)
            #ax.plot(fx.T, fy.T, '+', label=tracer.label(), color=color, alpha=alpha)
            ax.axvspan(tracer.fmin, tracer.fmax, facecolor='0.5', alpha=0.1)
            count += 1

        ax.autoscale()
        #ax.set_xlim((0., 85.))
        #ax.set_title("$\sqrt{PSD}$")
        ax.set_ylabel("A [counts]")
        ax.set_xlabel("f[Hz]")
        ax.set_yscale("log")

    def iter_linefit(self, func=None):
        if func==None:
            func=self.fit_function
        for s,ta,fxfy in self.iterdd():
            yield s, ta, curve_fit(func, *fxfy)

    def tracer(self):
        (tr1, _) = self.spectra[0]
        (tr2, _) = self.spectra[1]
        return tr1, tr2

    def amps(self):
        # list of amplitude spectra arrays
        a = []
        for s in self.spectra:
            a.append(s[1][1])
        return a

    def freqs(self):
        # list of frequency arrays
        f = []
        for s in self.spectra:
            f.append(s[1][0])
        return f

    def combined_freqs(self):
        # return array with frequencies used e.g. for interpolation
        return num.sort(num.unique(num.array(self.freqs())))

    def get_interpolated_spectra(self, fmin=None, fmax=None):
        if not fmin:
            fmin = -99999.
        if not fmax:
            fmax = 999999.
        both_f = self.combined_freqs()
        indx = num.where(num.logical_and(both_f>=fmin, both_f<=fmax))
        f_use = both_f[indx]
        a_s = num.empty((2, len(f_use)))
        i = 0
        for tr, fxfy in self.spectra:
            fx, fy = fxfy
            f = interpolate.interp1d(fx, fy)
            a_s[i][:] = f(f_use)
            i += 1

        return f_use, a_s


class SyntheticCouple():
    def __init__(self, master_slave, method='mtspec', use_common=False):
        self.method = method
        self.master_slave = master_slave
        self.spectra = Spectra()
        self.noisy_spectra = Spectra()
        self.fit_function = None
        self.colors = None
        self.good = True
        self.use_common = use_common
        self.repeat = 0
        self.noise_level = 0
        self.invert_data = None
        self._cc = None
        self.filters = {}
        self.normalize_waveforms = False
        self.normalization_factor = 1.

    def drop_data(self):
        self.master_slave[0].drop_data()
        self.master_slave[1].drop_data()

    def cc_coef(self):
        if self._cc is None:
            (tr1, _) = self.spectra.spectra[0]
            (tr2, _) = self.spectra.spectra[1]
            if tr1.processed and tr2.processed and tr1!="NoData" and tr2!="NoData":
                try:
                    ctr = trace.correlate(
                        tr1.processed,
                        tr2.processed,
                        mode='same',
                        use_fft=True,
                        normalization='normal')
                    t, self._cc = ctr.max()
                except AttributeError:
                    self._cc = 0.
            else:
                self._cc = 0.

        return self._cc

    def process(self, **pp_kwargs):
        ready = []
        want_length = -1
        for i, tracer in enumerate(self.master_slave):
            tr = tracer.process(**pp_kwargs)
            if tr is False or isinstance(tr, str):
                self.spectra.spectra.append((tracer, tr))
                self.good = tr
                continue

            if want_length<(tr.tmax-tr.tmin):
                want_length = tr.tmax-tr.tmin

            ready.append((tr, tracer))

        for tr, tracer in ready:
            if self.normalize_waveforms:
                ynew = tr.get_ydata()
                self.normalization_factor = num.max(num.abs(ynew))
                ynew /= self.normalization_factor
                tr.set_ydata(ynew)

            length = tr.tmax - tr.tmin
            diff = want_length - length
            tr.extend(tmin=tr.tmin-diff, tmax=tr.tmax, fillmethod='repeat')

            #f, a = self.get_spectrum(tr, tracer, length)
            f, a = tracer.processed_spectrum(self.method, filters=self.filters)

            #f_n, a_n = tracer.noise_spectrum(self.method, filters=self.filters)
            #power_noise = util.power(f_n, a_n)

            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

    #def get_spectrum(self, tr, tracer, wantlength, spectralize_kwargs=None):
    #    if not spectralize_kwargs:
    #        spectralize_kwargs = {}
    #    spectralize_kwargs.update({'tinc': tracer.tinc,
    #                               'chopper': tracer.chopper,
    #                               'filters': self.filters})
    #    return spectralize(tr, self.method, **spectralize_kwargs)

    def plot(self, colors, **kwargs):
        colors = 'rb'
        fn = kwargs.pop('savefig', False)
        #fig = plt.figure(figsize=(4, 6.5))
        #ax = fig.add_subplot(3, 1, 3)
        fig = plt.figure(figsize=(7.5, 3.5))
        ax = fig.add_subplot(1, 2, 2)
        colors = 'rb'
        #self.spectra.plot_all(ax, colors=colors, legend=False)
        self.spectra.plot_all(ax, legend=False)
        if self.invert_data:
            Q = self.invert_data[4]
            ax.text(0.01, 0.01, "1/Q=%1.6f\ncc=%1.2f" % (Q, self._cc),
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes)
        ax.text(0.0, 1.01, "b)", verticalalignment='bottom',
                horizontalalignment='right',
                transform=ax.transAxes)
        yshift=0
        info_str = ''
        for i, tracer in enumerate(self.master_slave):
            #ax = fig.add_subplot(3, 1, 1+i)
            ax = fig.add_subplot(1, 2, 1)
            if isinstance(tracer, DataTracer):
                tracer.setup_data()
                tr = tracer.processed
            else:
                tr = tracer.setup_data()

            otime = tracer.source.time
            plot_traces(tr=tr, t_shift=-otime, ax=ax, label=tracer.label(),
                        color=colors[i])
            #plot_traces(tr=tr, t_shift=-otime, ax=ax, label=tracer.label(), color=colors[tracer])
            #ax.text(0.01, 0.01, 'Ml=%1.1f, station: %s'
            #        % (tracer.source.magnitude, tracer.target.codes[1]),
            #        size=8,
            #        verticalalignment='bottom', horizontalalignment='left',
            #        transform=ax.transAxes )
            #info_str += "\notime: %s,\n M: %1.1f, s: %s" % (util.time_to_str(otime),
            #                                                tracer.source.magnitude,
            #                                                ".".join(tracer.target.codes))'
        #ax.set_xlabel('A [m/s]')
        #ax = fig.add_subplot(4, 1, 4)
        #info_str += '\ncc=%s' % self.cc_coef()
        ax.set_ylabel("normalized amplitude [m/s]")
        ax.set_xlabel("time after origin [s]")
        ax.text(0.0, 1.01, "a)", verticalalignment='bottom',
                horizontalalignment='right',
                transform=ax.transAxes)
        #ax.axis('off')
        #plt.tight_layout()
        #fig.subplots_adjust(hspace=0.21, wspace=0.21)
        if fn:
            fig.tight_layout()
            fig.savefig(fn, dpi=200)
        Qs = []
        return Qs

    def delta_onset(self):
        ''' Get the onset difference between two (!) used phases'''
        diff = 0
        assert len(self.master_slave)==2
        for tr in self.master_slave:
            if not diff:
                diff = tr.onset()
            else:
                diff -= (tr.onset())
        return diff

    def set_fit_function(self, func):
        self.spectra.set_fit_function(func)
        self.fit_function = func

    def get_slopes(self):
        return self.spectra.get_slopes()

    def get_spectra(self):
        return self.spectra

    def _q_1(self, fxfy, A, dists, arrivals):
        # following eqn:
        # http://www.ga.gov.au/corporate_data/81414/Jou1995_v15_n4_p511.pdf (5)
        t1, t2 = arrivals
        return num.pi*num.abs(num.array(fxfy[0][0]))*(t2-t1)/(num.log(A[0])+num.log(dists[0])-num.log(A[1])-num.log(dists[1]))

    def get_average_slopes(self, upper_lim=0., lower_lim=9999):
        x_master, y_master, x_slave, y_slave = self.get_average_spectra()
        fits = []
        for fxfy in [num.array([x_master, y_master]), num.array([x_slave, y_slave])]:
            slope_section = extract(fxfy, upper_lim=upper_lim, lower_lim=lower_lim)
            popt, pcov = curve_fit( self.fit_function, *slope_section)
            fits.append([popt, pcov])

        return fits

    def get_average_spectra(self):
        """Returns average source spectra for master and slave event"""
        fx_master = []
        fy_master = []
        fx_slave = []
        fy_slave = []
        for s, ta, fxfy in self.spectra.iterdd():
            if s.is_slave:
                fx_slave.append(list(fxfy[0]))
                fy_slave.append(fxfy[1])

            if s.is_master:
                fx_master.append(list(fxfy[0]))
                fy_master.append(fxfy[1])

        fy_master= num.asarray(fy_master)
        fy_slave = num.asarray(fy_slave)
        if len(fx_master)>1:
            intersecting_master, average_spectrum_master = average_spectra(fx_master, fy_master)
        else:
            intersecting_master, average_spectrum_master = fx_master[0], fy_master[0]
        if len(fx_slave)>1:
            intersecting_slave, average_spectrum_slave = average_spectra(fx_slave, fy_slave)
        else:
            intersecting_slave, average_spectrum_slave = fx_slave[0], fy_slave[0]
        return intersecting_master, average_spectrum_master, intersecting_slave , average_spectrum_slave

    def get_target_distances_and_depths(self):
        ms = self.master_slave
        d1 = ms[0].target.distance_to(ms[0].source)
        d2 = ms[1].target.distance_to(ms[1].source)
        return d1, d2, ms[0].source.depth, ms[1].source.depth

    def frequency_range_work(self):
        return max(self.master_slave[0].fmin, self.master_slave[1].fmin),\
            min(self.master_slave[0].fmax, self.master_slave[1].fmax)


    def __str__(self):
        s = 'master: %s\n slave: %s\n good: %s\n' % (
                                                    self.master_slave[0],
                                                    self.master_slave[1],
                                                    str(self.good))
        return s


def limited_frequencies_ind(fmin, fmax, f):
    return num.where(num.logical_and(f>=fmin, f<=fmax))

def noisy_spectral_ratios(pairs):
    '''Ratio of two overlapping spectra'''
    ratios = []
    indxs = []
    spectra_couples = defaultdict(list)
    for tr, fxfy in pairs.noisy_spectra.spectra:
        spectra_couples[tr].append(fxfy)

    one, two = spectra_couples.values()
    for i in xrange(len(one)):
        fmin = max(pairs.fmin, min(min(one[i][0]), min(two[i][0])))
        fmax = min(pairs.fmax, max(max(one[i][0]), min(two[i][0])))
        ind0 = limited_frequencies_ind(fmin, fmax, one[i][0])
        ind1 = limited_frequencies_ind(fmin, fmax, two[i][0])
        fy_ratio = one[i][1][ind0]/two[i][1][ind1]
        ratios.append(fy_ratio)
        indxs.append(one[i][0][ind0])
    return indxs, ratios


#def spectral_ratio(couple):
#    '''Ratio of two overlapping spectra.
#
#    i is the mean intercept.
#    s is the slope ratio of each individual linregression.'''
#    assert len(couple.spectra.spectra)==2, '%s spectra found: %s' % (len(couple.spectra.spectra), couple.spectra.spectra)
#    fx = []
#    fy = []
#    s = None
#    if couple.use_common:
#        cfmin = max([couple.spectra.spectra[1][0].fmin, couple.spectra.spectra[0][0].fmin])
#        cfmax = min([couple.spectra.spectra[1][0].fmax, couple.spectra.spectra[0][0].fmax])
#    else:
#        cfmin = None
#        cfmax = None
#    dis = None
#    for tr, fxfy in couple.spectra.spectra:
#        if dis is None:
#            dis = tr.source.distance_to(tr.target)
#        else:
#            assert tr.source.distance_to(tr.target) < dis
#
#        fs, a = fxfy
#        if cfmin and cfmax:
#            fmin = cfmin
#            fmax = cfmax
#        else:
#            fmin = max(tr.fmin, fs.min())
#            fmax = min(tr.fmax, fs.max())
#        ind = limited_frequencies_ind(fmin, fmax, fs)
#        slope, interc, r, p, std = linregress(fs[ind], num.log(a[ind]))
#        if not s:
#            i = interc
#            s = slope
#            #s = num.exp(slope)
#        else:
#            i += interc
#            s -= slope
#            #s -= num.exp(slope)
#        fx.append(fs)
#    print 'TODO: rethink this method'
#    return i/2., -s, num.sort(num.hstack(fx)), r, p, std

def spectral_ratio(couple):
    '''Ratio of two overlapping spectra.

    The alternative method.

    i is the mean intercept.
    s is the slope ratio of each individual linregression.'''
    assert len(couple.spectra.spectra)==2, '%s spectra found: %s' % (len(couple.spectra.spectra), couple.spectra.spectra)

    if couple.use_common:
        cfmin, cfmax = couple.frequency_range_work()
    else:
        raise Exception("deprecated")
        cfmin = None
        cfmax = None

    f_use, a_s = couple.spectra.get_interpolated_spectra(cfmin, cfmax)

    slope, interc, r, p, std = linregress(f_use, num.log(a_s[1]/a_s[0]))

    return interc, slope, f_use, a_s, r, p, std



class QInverter:
    def __init__(self, couples, cc_min=0.8, onthefly=False, snr_min=0.):
        if len(couples)==0:
            raise Exception('Empty list of test couples')
        self.couples = couples
        self.cc_min = cc_min
        self.onthefly = onthefly
        self.snr_min = snr_min
        self.snr_min = snr_min

    def invert(self):
        self.allqs = []
        self.ratios = []
        widgets = ['regression analysis', progressbar.Percentage(), progressbar.Bar()]
        pb = progressbar.ProgressBar(maxval=len(self.couples), widgets=widgets).start()
        less_than_cc_thres = 0
        for i_c, couple in enumerate(self.couples):
            pb.update(i_c+1)
            fail_message = None
            if self.onthefly:
                tracer_master, tracer_slave = couple.master_slave
                tr1 = tracer_master.setup_data(normalize=couple.normalize_waveforms)
                tr2 = tracer_slave.setup_data(normalize=couple.normalize_waveforms)
                if not tr1 or not tr2:
                    logger.debug('tr1 or tr2 are None')
                    couple.drop_data()
                    continue
                else:
                    couple.process()

            snr_tr1 = tracer_master.snr()
            snr_tr2 = tracer_slave.snr()
            if snr_tr1 < self.snr_min:
                fail_message = 'snr %s < %s. skip' % (snr_tr1, self.snr_min)
            if snr_tr2 < self.snr_min:
                fail_message = 'snr %s < %s. skip' % (snr_tr2, self.snr_min)
            if fail_message:
                logger.debug(fail_message)
                couple.drop_data()
                continue

            cc_coef = couple.cc_coef()
            if cc_coef < self.cc_min:
                less_than_cc_thres += 1
                logger.debug('lower than cc threshold. skip')
                continue

            interc, slope, fx, a_s, r, p, std = spectral_ratio(couple)
            dt = couple.delta_onset()
            Q = num.pi*dt/slope
            if num.isnan(Q) or Q==0. or dt>4.:
                logger.debug('Q is nan or dt>4')
                if self.onthefly:
                    couple.drop_data()
                continue
            couple.invert_data = (dt, fx, slope, interc, 1./Q, r, p, std)
            self.allqs.append(1./Q)
            if self.onthefly:
                couple.drop_data()
        pb.finish()

    def plot(self, ax=None, q_threshold=None, relative_to=None, want_q=False):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1,1,1)
        median = num.median(self.allqs)
        mean = num.mean(self.allqs)
        if q_threshold is not None:
            filtered = filter(lambda x: x>median-q_threshold and x<median+q_threshold, self.allqs)
        else:
            filtered = self.allqs

        fsize = 10
        ax.hist(filtered, bins=99, color="blue")
        ax.set_xlabel(r'$Q^{-1}$', size=fsize)
        ax.set_ylabel('counts', size=fsize)
        txt ='median: %1.3f\n$\sigma$: %1.3f' % (median, num.std(self.allqs))
        ax.text(0.01, 0.99, txt, size=fsize, transform=ax.transAxes, verticalalignment='top')
        ax.axvline(0, color='black', alpha=0.3)
        ax.axvline(median, color='black')
        #ax.set_xlim((-1., 1.))
        if q_threshold is not None:
            ax.set_xlim([q_threshold, q_threshold])
        elif relative_to=='median':
            ax.set_xlim([median-q_threshold, median+q_threshold])
        elif relative_to=='mean':
            ax.set_xlim([mean-q_threshold, mean+q_threshold])

        if want_q:
            ax.axvline(want_q, color='r')
        return

    def analyze_selected_couples(self, couples, indx, indxinvert):
        fig = plt.gcf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #ax3 = fig.add_subplot(223)
        #ax4 = fig.add_subplot(224)

        for i in indxinvert:
            c = couples[i]
            freqs = c.spectra.freqs()
            amps = c.spectra.amps()
            fmin, fmax = c.frequency_range_work()
            freqsi, ampsi = c.spectra.get_interpolated_spectra(fmin, fmax)

            for i in xrange(1):
                ax1.plot(freqs[i], amps[i], alpha=0.6, linewidth=0.1, color='red')
            ax2.plot(freqsi, num.log(ampsi[0]/ampsi[1]), alpha=0.6, linewidth=0.1, color='red')

        for i in indx:
            c = couples[i]
            freqs = c.spectra.freqs()
            amps = c.spectra.amps()
            fmin, fmax = c.frequency_range_work()
            freqsi, ampsi = c.spectra.get_interpolated_spectra(fmin, fmax)

            for i in xrange(1):
                ax1.plot(freqs[i], amps[i], alpha=0.6, linewidth=0.1, color='blue')
            ax2.plot(freqsi, num.log(ampsi[0]/ampsi[1]), alpha=0.6, linewidth=0.1, color='blue')

        #for ax in [ax1, ax2, ax3, ax4]:'
        ax1.set_xlim((25, 110))
        ax1.set_xscale('log')
        ax1.set_yscale('log')


    def analyze(self, couples=None, fnout_prefix="q_fit_analysis"):
        couples_with_data = filter(lambda x: x.invert_data is not None,
                                   self.couples)
        couples_with_data = filter(lambda x:
                                   x.invert_data[4] < 0.5 and \
                                   not num.isnan(x.invert_data[4]) and\
                                   num.abs(x.invert_data[4])<5E3, 
                                   couples_with_data)
        n = len(couples_with_data)
        Qs = num.empty(n)
        mags = num.empty(n)
        magdiffs = num.empty(n)
        cc = num.empty(n)
        d_traveled = num.empty(n)
        d_passing = num.empty(n)
        rs = num.empty(n)
        ps = num.empty(n)
        stds = num.empty(n)
        dts = num.empty(n)
        fwidth = num.empty(n)
        by_target = {}
        target_combis = []


        for ic, c in enumerate(couples_with_data):
            dt, fx, slope, interc, Q, r, p, std = c.invert_data
            #if abs(Q)>0.5 or num.isnan(Q) or num.abs(Q)>1E30:
            #    continue

            fwidth[ic] = num.max(fx)-num.min(fx)
            rs[ic] = r**2
            ps[ic] = num.log(p)
            dts[ic] = dt
            stds[ic] = std
            Qs[ic] = Q
            cc[ic] = c.cc_coef()
            meanmag = (c.master_slave[0].source.magnitude+c.master_slave[1].source.magnitude)/2.
            mags[ic] = meanmag
            d_traveled[ic] = c.ray[3]
            d_passing[ic] = c.ray[4]
            magdiff = abs(c.master_slave[0].source.magnitude-c.master_slave[1].source.magnitude)
            magdiffs[ic] = magdiff
            tr1, tr2 = c.master_slave
            key = '%s-%s' % (tr1.target.codes[1], tr2.target.codes[1])
            target_combis.append(key)
            try:
                by_target[key].append(Q)
            except KeyError:
                by_target[key] = [Q]

        results = {'Q': Qs,
                   'mean mag': mags,
                   'magdiff': magdiffs,
                   'cc': cc,
                   'd_trav': d_traveled,
                   'd_pass': d_passing,
                   'r2-value': rs,
                   'log(p-value)': ps,
                   'std': stds,
                   'dts': dts,
                   'fwidth': fwidth,
                   }
        selector_min = None
        selector_max = None
        #selector = "std"
        #selector_max = 0.005
        #selector = "r2-value"
        #selector_min = 0.75
        #selector = "Q"
        #selector_max = 0.
        #selector_min = 0.90
        selectors = [
            #("std", None, 0.025),
            ("r2-value", 0.2, None),
            #("cc", 0.8, None),
            #("magdiff", None,  0.3),
        ]
        indxall = num.arange(len(couples_with_data)) 
        indx = indxall
        #indxinvert = num.arange(len(couples_with_data))
        for selector, selector_min, selector_max in selectors:
            if selector is not None:
                if selector_max is not None:
                    indx_tmp = num.where(results[selector]<=selector_max)
                    indx = num.intersect1d(indx, indx_tmp[0])

                if selector_min is not None:
                    indx_tmp = num.where(results[selector]>=selector_min)
                    indx = num.intersect1d(indx, indx_tmp[0])
        indxinvert = num.setdiff1d(indxall, indx)
        #if abs(len(indxinvert)-len(indxall))<2:
        #    logger.warn('bad results')
        #    return 
        print 'len indx', len(indx)
        print 'len indxinvert', len(indxinvert)

        markersize = 1.
        alpha = 0.8
        indx_style = {'marker': 'o', 'markerfacecolor': 'blue', 
                             'alpha': alpha, 'markersize': markersize,
                             'linestyle': 'None'}

        invert_indx_style = {'marker': 'o', 'markerfacecolor': 'red', 
                             'alpha': alpha-0.1, 'markersize': markersize,
                             'linestyle': 'None'}

        fig = plt.figure(figsize=(16, 14))
        keys = results.keys()
        combinations = []
        for ik in xrange(len(keys)):
            k1 = keys.pop()
            for k2 in keys:
                combinations.append((k1, k2))

        nrows = num.ceil(num.sqrt(len(combinations)+1))
        ncols = int(num.ceil(float(len(combinations)+1)/nrows))

        for icomb, combination in enumerate(combinations):
            wanty, wantx = combination
            ax = fig.add_subplot(nrows, ncols, icomb+1)
            ax.plot(results[wantx][indxinvert], results[wanty][indxinvert], **invert_indx_style)
            ax.plot(results[wantx][indx], results[wanty][indx], **indx_style)
            ax.set_xlabel(wantx, fontsize=8)
            ax.set_ylabel(wanty, fontsize=8)
        ax = fig.add_subplot(nrows, ncols, icomb+2)
        ax.hist(results["Q"][indxinvert], bins=50,
            color=invert_indx_style["markerfacecolor"], alpha=alpha)
        ax.hist(results["Q"][indx], bins=50,
                color=indx_style["markerfacecolor"], alpha=alpha)
        median = num.median(results["Q"][indx])
        txt ='median: %1.4f\n$\sigma$: %1.5f' % (
            median, num.std(results["Q"][indx]))
        ax.text(0.01, 0.99, txt, size=6, transform=ax.transAxes,
                verticalalignment='top')
        #ax.axvline(median, color='black')

        plt.tight_layout()
        fig.savefig(fnout_prefix + "_qvs.png", dpi=400)

        fig = plt.figure()
        self.analyze_selected_couples(couples_with_data, indx, indxinvert)
        plt.tight_layout()
        fig.savefig(fnout_prefix + "_spectra.png", dpi=200)

        fig = plt.figure(figsize=(10, 10))
        nrow = 3
        ncolumns = int(len(by_target)/nrow)+1
        i = 1
        q_threshold = 2000
        target_combis_indx = []
        target_combis_indxinvert = []
        want_hists_indx = {}
        want_hists_indxinvert = {}
        for i, target_combi in enumerate(target_combis):
            if target_combi not in want_hists_indx:
                want_hists_indx[target_combi] = []
            if target_combi not in want_hists_indxinvert:
                want_hists_indxinvert[target_combi] = []

            if i in indx:
                want_hists_indx[target_combi].append(results["Q"][i])
            elif i in indxinvert:
                want_hists_indxinvert[target_combi].append(results["Q"][i])
            else:
                raise Exception("Index in None of them")
        all_combis = list(set(target_combis)) 
        print 'all combis:', all_combis
        n_want = len(all_combis)
        nrows = num.max((int(num.ceil(num.sqrt(n_want+1))), 2))
        ncols = num.max((int(num.ceil(float(n_want)/nrows)), 2))
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        axs = dict(zip(all_combis, flatten_list(axs)))
        for k, v in want_hists_indxinvert.items():
            axs[k].hist(v, color=invert_indx_style["markerfacecolor"], alpha=alpha)
            axs[k].set_title(k)

        for k, v in want_hists_indx.items():
            axs[k].hist(v, color=indx_style["markerfacecolor"], alpha=alpha)
            axs[k].set_title(k)

        fig.savefig(fnout_prefix + "_bytarget.png")

        fig, axs = plt.subplots(nrows, ncols, sharex=True)
        axs = dict(zip(all_combis, flatten_list(axs)))
        for i, c in enumerate(couples_with_data):
            tracer1, tracer2 = c.master_slave
            key = '%s-%s' % (tracer1.target.codes[1], tracer2.target.codes[1])
            tr1 = tracer1.setup_data()
            tr2 = tracer2.setup_data()

            tr1.shift(-tr1.tmin)
            tr2.shift(-tr2.tmin)
            if i in indx:
                color = indx_style["markerfacecolor"]
                label = 'good'
            else:
                color = invert_indx_style["markerfacecolor"]
                label = ''
            axs[key].plot(tr1.get_xdata(), tr1.get_ydata(), color=color,
                          alpha=0.5, linewidth=0.4)
            axs[key].plot(tr2.get_xdata(), tr2.get_ydata(), color=color,
                          alpha=0.5, linewidth=0.4)

        for k, ax in axs.items():
            ax.set_title(k)

        fig.savefig(fnout_prefix + "_traces.png", dpi=240)

        for k, v in results.iteritems():
            with open(fnout_prefix+"data_%s.txt" % k, 'w') as f:
                for item in v:
                    f.write("%1.6f \n" % item)

def model_plot(mod, ax=None, parameter='qp', cmap='copper', xlims=None):
    cmap = mpl.cm.get_cmap(cmap)
    ax = ax_if_needed(ax)
    x, z = num.meshgrid(xlims, mod.profile('z'))
    p = num.repeat(mod.profile(parameter), len(xlims)).reshape(x.shape)
    contour = ax.contourf(x, z, p, cmap=cmap, alpha=0.5)


def location_plots(tracers, colors=None, background_model=None, parameter='qp'):
    fig = plt.figure(figsize=(5,4))
    minx, maxx = num.zeros(len(tracers)), num.zeros(len(tracers))
    miny, maxy = num.zeros(len(tracers)), num.zeros(len(tracers))
    ax = fig.add_subplot(111)
    widgets = ['plotting model and rays: ', progressbar.Percentage(), progressbar.Bar()]
    pb = progressbar.ProgressBar(maxval=len(tracers)-1, widgets=widgets).start()
    for itr, tr in enumerate(tracers):
        pb.update(itr)
        arrival = tr.arrival()
        z, x, t = arrival.zxt_path_subdivided()
        x = x[0].ravel()*cake.d2m
        ax.plot( x, z[0].ravel(), color=colors[tr], alpha=0.3)
        minx[itr] = num.min(x)
        maxx[itr] = num.max(x)
        miny[itr] = num.min(z)
        maxy[itr] = num.max(z)
    pb.finish()
    minx = min(minx.min(), -100)-100
    maxx = max(maxx.max(), 100)+100
    xlims=(minx, maxx)
    ylims=(min(miny), max(maxy))
    model_plot(background_model, ax=ax, parameter=parameter, xlims=xlims)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Depth [m]')
    ax.invert_yaxis()



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


def dbtest(noise_level=0.00000000005):
    print '-------------------db test-------------------------------'
    use_real_shit = False
    use_extended_sources = False
    use_responses = True
    load_coupler = True
    fn_coupler = 'dummy_coupling.p'
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
    #method = 'filter'
    method = 'mtspec'
    #method = 'sine_psd'
    if method == 'filter':
        delta_f = 3.
        fcs = num.arange(30, 85, delta_f)
        fwidth = 3
        filters = [(f, fwidth) for f in fcs]
    else:
        filters = None
    min_magnitude = 2.
    max_magnitude = 6.
    fminrange = 20.
    #use_common = False
    use_common = True
    fmax_lim = 80.
    #zmax = 10700
    fmin = 35.
    fmin = Magnitude2fmin.setup(lim=fmin)
    fmax = 85.
    #window_by_magnitude = Magnitude2Window.setup(0.8, 1.)
    window_by_magnitude = Magnitude2Window.setup(0.08, 5.)
    quantity = 'velocity'
    store_id = 'qplayground_total_2'
    #store_id = 'qplayground_total_2_q25'
    #store_id = 'qplayground_total_2_q400'
    #store_id = 'qplayground_total_1_hr'
    #store_id = 'qplayground_total_4_hr'
    #store_id = 'qplayground_total_4_hr_full'
    #store_id = 'ahfullgreen_2'
    #store_id = 'ahfullgreen_4'

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

    engine = LocalEngine(store_superdirs=['/data/stores', '/media/usb/stores'])
    print engine
    #engine = LocalEngine(store_superdirs=['/media/usb/stores'])
    store = engine.get_store(store_id)
    config = engine.get_store_config(store_id)
    mod = config.earthmodel_1d

    gf_padding = 50
    zmin = config.source_depth_min + gf_padding
    zmax = config.source_depth_max - gf_padding
    dist_min = config.distance_min
    dist_max = config.distance_max
    channel = 'SHZ'
    tt_mu = 0.0
    tt_sigma = 0.01
    save_figs = True
    nucleation_radius = 0.1

    # distances used if not real sources:
    if test_scenario:
        distances = num.linspace(config.distance_min+gf_padding,
                                 config.distance_max-gf_padding, 12)
        source_depths = num.linspace(zmin, zmax, 12)
    else:
        distances = num.arange(config.distance_min+gf_padding, config.distance_max-gf_padding, 200)
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
    want_phase = 's'
    fn_noise = '/media/usb/webnet/mseed/noise.mseed'
    fn_records = '/media/usb/webnet/mseed'
    #if use_real_shit:
    if False:
        noise = Noise(files=fn_noise, scale=noise_level)
        noise_pile = pile.make_pile(fn_records)
    else:
        #noise = RandomNoiseConstantLevel(noise_level)
        noise = None
        noise_pile = None

    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    all_depths = [e.depth for e in events]
    some_depths = [d/1000. for d in all_depths if d>8500]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('$q_p$ model')
    #ax.set_xticks([100, 200, 300])
    ax.invert_yaxis()
    ax.set_ylim(0, 12.)
    plot_model(mod, ax=ax, parameters=['qp'])
    #ax.axes.get_yaxis().set_visible(False)
    ax.axhspan(min(some_depths), max(some_depths), alpha=0.1)

    #ax = fig.add_subplot(1, 3, 1, sharey=ax)
    #plot_model(mod, ax=ax, parameters=['vp'])
    ##cake_plot.sketch_model(mod, ax)
    #ax.axhspan(min(some_depths), max(some_depths), alpha=0.1)
    #ax.set_ylim(0, 12.)
    ##ax.set_ylim(0, 12000.)
    #ax.axes.get_yaxis().set_visible(True)

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
    if not want_station=='all':
        print 'warning: only using station: %s' %'.'.join(want_station)
        stations = filter(lambda x: want_station == x.nsl(), stations)


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
                #'elevation': 0., 'codes': ('cz', 'kvc', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('cz', 'lbc', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('cz', 'kaz', '', channel), 'store_id': store_id}
                'elevation': 0., 'codes': ('cz', 'vac', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('cz', 'nkc', '', channel), 'store_id': store_id}
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
            fig, ax = Animator.get_3d_ax()
            Animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
            Animator.plot_sources(sources=targets, reference=coupler.hookup, ax=ax)

        elif use_real_shit is True:
            targets = [s2t(s, channel, store_id=store_id) for s in stations]
            events = filter(lambda x: x.depth>zmin and x.depth<zmax, events)
            events = filter(lambda x: x.magnitude>=min_magnitude, events)
            events = filter(lambda x: x.magnitude<=max_magnitude, events)
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
            #for s in sources:
            #    #strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
            #    #                                                         dipmin, dipmax,
            #    #                                                         rakemin, rakemax)
            #    #s.strike = strike
            #    #s.dip = dip
            #    #s.rake = rake
        if use_responses:
            associate_responses(
                glob.glob('responses/resp*'),
                targets,
                time=util.str_to_time('2012-01-01 00:00:00.'))
        #associate_responses(glob.glob('responses/*pz'),
        #                    targets,
        #                    time=util.str_to_time('2012-01-01 00:00:00.'),
        #                    type='polezero')

        #plot_response(response=targets[0].filter.response)
        logger.info('number of sources: %s' % len(sources))
        logger.info('number of targets: %s' % len(targets))
        coupler.process(sources, targets, mod, [want_phase, want_phase.lower()],
                        ignore_segments=True, dump_to=fn_coupler, check_relevance_by=noise_pile)
    #fig, ax = animator.get_3d_ax()
    #animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
    pairs_by_rays = coupler.filter_pairs(4, 1000, data=coupler.filtrate,
                                         max_mag_diff=0.2)
    animator = Animator(pairs_by_rays)
    #plt.show()
    widgets = ['plotting segments: ', progressbar.Percentage(), progressbar.Bar()]
    paired_sources = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, incidence_angle = p
        paired_sources.extend([s1, s2])
    used_mags = [s.magnitude for s in paired_sources]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(used_mags)
    paired_source_dict = paired_sources_dict(paired_sources)
    animator.plot_sources(sources=paired_source_dict, reference=coupler.hookup, ax=None, alpha=1)
    #pb = progressbar.progressbar(maxval=len(pairs_by_rays)-1, widgets=widgets).start()
    #for i_r, r in enumerate(pairs_by_rays):
    #    e1, e2, t, td, pd, segments = r
    #    animator.plot_ray(segments, ax=ax)
    #    pb.update(i_r)
    #print 'done'
    #pb.finish()
    #plt.show()
    testcouples = []
    pairs = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1 = p
        fmin2 = None
        pair = []
        for sx in [s1, s2]:
            fmin1 = fmin_by_magnitude(sx.magnitude)
            if want_phase.upper()=="S":
                # accounts for fc changes: Abstract
                # http://www.geologie.ens.fr/~madariag/Programs/Mada76.pdf
                fmin1 /= 1.5

            #fmax = min(fmax_lim, vp/fresnel_lambda(totald, td, pd))
            #print 'test me, change channel code id to lqt'
            #t.dip = -90. + i1
            #t.azimuth = t.azibazi_to(sx)[1]
            tracer1 = Tracer(sx, t, p_chopper, channel=channel, fmin=fmin1,
                             fmax=fmax, want=quantity, 
                             perturbation=perturbation.perturb(0))
            tracer1.engine = engine
            dist1, depth1 = tracer1.get_geometry()
            if dist1< dist_min or dist1>dist_max:
                break
            if fmax-fmin1<fminrange:
                break
            pair.append(tracer1)
            tracers.extend(pair)
            pairs.append(pair)

        if len(pair)==2:
            testcouple = SyntheticCouple(master_slave=pair, method=method, use_common=use_common)
            testcouple.normalize_waveforms = normalize_waveforms
            testcouple.ray = p
            testcouple.filters = filters
            #testcouple.process(noise=noise)
            #if len(testcouple.spectra.spectra)!=2:
            #   logger.warn('not 2 spectra in test couple!!!! why?')
            #   continue
            testcouples.append(testcouple)
    if len(tracers)==0:
        raise Exception('no tracers survived the assessment')

    #builder = Builder()
    #tracers = builder.build(tracers, engine=engine, snuffle=False)
    colors = UniqueColor(tracers=tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('location_model_db1.png', dpi=200)
    #plt.show()
    #widgets = ['processing couples: ', progressbar.Percentage(), progressbar.Bar()]
    #pb = progressbar.ProgressBar(len(pairs)-1, widgets=widgets).start()
    #for i_p, pair in enumerate(pairs):
    #    pb.update(i_p)
    #    testcouple = SyntheticCouple(master_slave=pair, method=method, use_common=use_common)
    #    testcouple.ray = r
    #    testcouple.filters = filters
    #    testcouple.process(noise=noise)
    #    if len(testcouple.spectra.spectra)!=2:
    #        logger.warn('not 2 spectra in test couple!!!! why?')
    #        continue
    #    testcouples.append(testcouple)
    #pb.finish()
    #testcouples = filter(lambda x: x.good==True, testcouples)
    #outfn = 'testimage'
    #plt.gcf().savefig('output/%s.png' % outfn)
    inverter = QInverter(couples=testcouples, onthefly=True, cc_min=0.8)
    inverter.invert()
    for i, testcouple in enumerate(num.random.choice(testcouples, 30)):
        fn = 'synthetic_tests/%s/example_%s_%s.png' % (want_phase, store_id, str(i).zfill(2))
        print fn
        testcouple.plot(infos=infos, colors=colors, noisy_q=False, savefig=fn)
    inverter.plot()
    #inverter.plot(q_threshold=800, relative_to='median', want_q=want_q)
    fig = plt.gcf()
    plt.tight_layout()
    fig.savefig('synthetic_tests/%s/hist_db%s.png' %(want_phase, store_id), dpi=200)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    inverter.analyze()
    plt.show()


def reset_events(markers, events):
    for e in events:
        marks = filter(lambda x: x.get_event_time()==e.time, markers)
        map(lambda x: x.set_event(e), marks)


def paired_sources_dict(paired_sources):
    paired_source_dict = {}
    for s in paired_sources:
        if not s in paired_source_dict.keys():
            paired_source_dict[s] = 1
        else:
            paired_source_dict[s] += 1
    return paired_source_dict

