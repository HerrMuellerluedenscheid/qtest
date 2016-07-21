import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rc('ytick', labelsize=10)
mpl.rc('xtick', labelsize=10)

import copy
import progressbar
from pyrocko.gf import meta, DCSource, RectangularSource, Target, LocalEngine
from pyrocko.gf import SourceWithMagnitude, OutOfBounds#, CircularSource
from pyrocko.guts import String, Float, Int
from pyrocko import orthodrome
from pyrocko.gui_util import PhaseMarker
from pyrocko import util
from pyrocko import cake, model, cake_plot
from pyrocko import pile
from pyrocko import moment_tensor
from pyrocko.fomosto import qseis
from pyrocko.trace import nextpow2
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as num
import os
import glob
from micro_engine import DataTracer, Tracer, Builder#, Brunes
from micro_engine import Noise, RandomNoise, RandomNoiseConstantLevel, Chopper, DDContainer
from micro_engine import associate_responses, TTPerturbation, UniformTTPerturbation
from autogain.autogain import PhasePie, PickPie
from brune import Brune
import logging
from distance_point2line import Coupler, Animator, Filtrate, fresnel_lambda
from util import Magnitude2Window, Magnitude2fmin, fmin_by_magnitude, M02tr
from rupture_size import radius as source_radius
try:
    from pyrocko.gf import BoxcarSTF, TriangularSTF, HalfSinusoidSTF
except ImportError as e:
    print 'CHANGE BRANCHES'
    raise e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pjoin = os.path.join
km = 1000.

methods_avail = ['fft', 'psd']
try:
    import pymutt
    methods_avail.append('pymutt')
except Exception as e:
    logger.exception(e)
try:
    from mtspec import mtspec, sine_psd
    methods_avail.append('mtspec')
    methods_avail.append('sine_psd')
except Exception as e:
    logger.exception(e)


def check_method(method):
    if method not in methods_avail:
        logger.exception("Method %s not available" % method)
        raise Exception("Method %s not available" % method)
    else:
        logger.debug('method used %s' % method)
        return True


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


def plot_traces(tr, t_shift=0, ax=None, label='', color='r'):
    ax = ax_if_needed(ax)
    ydata = tr.get_ydata()
    if tr:
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


def get_stf(magnitude=0, stress=0.1, vr=2750., type=None):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    duration = M02tr(Mo, stress, vr)
    print 'CHECK THE DURATION: %s' % duration
    if type==None:
        stf = None
    elif type=='boxcar':
        stf = BoxcarSTF(duration=duration)
    elif type=='triangular':
        stf = TriangularSTF(duration=duration)
    elif type=='halfsin':
        stf = HalfSinusoidSTF(duration=duration)
    elif type=='brunes':
        stf = None
    else:
        raise Exception('unknown STF type: %s' % type)
    return stf


class RectangularBrunesSource(RectangularSource):
    brunes = Brune.T(optional=True, default=None)
    def __init__(self, **kwargs):
        RectangularSource.__init__(self, **kwargs)
        #self.stf = get_stf(self.magnitude)
        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

#class CircularBrunesSource(CircularSource):
#    brunes = Brune.T(optional=True, default=None)
#    def __init__(self, **kwargs):
#        CircularSource.__init__(self, **kwargs)
#        #self.stf = get_stf(self.magnitude)
#        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


class DCSourceWid(DCSource):
    id = String.T(optional=True, default=None)
    brunes = Brune.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)
        #self.stf = get_stf(self.magnitude)
        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


class LineSource(DCSource):
    discretized_source_class = meta.DiscretizedMTSource
    i_sources = Int.T()
    velocity = Float.T()
    brunes = Brune.T(optional=True, default=None)

    def base_key(self):
        return DCSource.base_key(self)+(self.i_sources, self.velocity)

    def discretize_basesource(self, store):

        c = store.config
        dz = c.source_depth_delta

        #i_sources sollte eher i_sources/2. sein!
        north_shifts = num.arange(-dz*self.i_sources, dz*self.i_sources, dz)
        print 'line_source_points ', north_shifts

        times = num.ones(len(north_shifts)) * (north_shifts[-1]-north_shifts[0])/self.velocity
        n = times.size

        mot = moment_tensor.MomentTensor(strike=self.strike, dip=self.dip, rake=self.rake,
                                                            scalar_moment=1.0/n)
        amplitudes = num.ones(n)/n
        m6s = num.repeat(mot.m6()[num.newaxis, :], n, axis=0)
        m6s[:, :] *= amplitudes[:, num.newaxis]
        ds = meta.DiscretizedMTSource(
            lat=self.lat,
            lon=self.lon,
            times=times,
            north_shifts=self.north_shift + north_shifts,
            east_shifts=self.east_shift,
            depths=self.depth,
            m6s=m6s)
        return ds


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


def spectralize(tr, method='mtspec', chopper=None, tinc=None):
    if method=='fft':
        # fuehrt zum amplitudenspektrum
        f, a = tr.spectrum(tfade=chopper.get_tfade(tr.tmax-tr.tmin))

    elif method=='psd':
        a_list = []
        #f_list = []
        tpad = tinc/2.
        #trs = []
        for itr in iter_chopper(tr, tinc=tinc, tpad=tpad):
            if tinc is not None:
                nwant = int(tinc * 2 / itr.deltat)
                if nwant != itr.data_len():
                    if itr.data_len() == nwant + 1:
                        itr.set_ydata( itr.get_ydata()[:-1] )
                    else:
                        continue

            itr.ydata = itr.ydata.astype(num.float)
            itr.ydata -= itr.ydata.mean()
            #if self.tinc is not None:
            win = num.hanning(itr.data_len())
            #else:
            #    win = num.ones(tr.data_len())

            itr.ydata *= win
            f, a = itr.spectrum(pad_to_pow2=True)
            a = num.abs(a)**2
            a *= itr.deltat * 2. / num.sum(win**2)
            a[0] /= 2.
            a[a.size/2] /= 2.
            a_list.append(a)
            #f_list.append(f)
            #trs.append(itr)
        a = num.vstack(a_list)
        a = median = num.median(a, axis=0)

    elif method=='pymutt':
        # berechnet die power spectral density
        r = pymutt.mtft(tr.ydata, dt=tr.deltat)
        f = num.arange(r['nspec'])*r['df']
        a = r['power']
        a = num.sqrt(a)

    elif method=='sine_psd':
        tr = tr.copy()
        ydata = tr.get_ydata()
        tr.set_ydata(ydata/num.max(num.abs(ydata)))
        a, f = sine_psd(data=tr.ydata, delta=tr.deltat)
        a = num.sqrt(a)

    elif method=='mtspec':
        tr = tr.copy()
        #_tr.set_network('C')
        #tr.taper(_taperer, inplace=True, chop=False)

        #pad_length = (_tr.tmax - _tr.tmin) * 0.5
        #_tr.extend(_tr.tmin-pad_length, _tr.tmax+pad_length)
        #trace.snuffle([_tr, tr])
        #import pdb
        #pdb.set_trace()
        ydata = tr.get_ydata()
        ydata = num.asarray(ydata, dtype=num.float)
        tr.set_ydata(ydata/num.max(num.abs(ydata)))
        a, f = mtspec(data=tr.ydata,
                      delta=tr.deltat,
                      number_of_tapers=10,
                      #number_of_tapers=12,
                      #number_of_tapers=2,
                      #time_bandwidth=7.5,
                      time_bandwidth=4.,
                      nfft=nextpow2(len(tr.get_ydata())),
                      quadratic=True,
                      statistics=False)
        a = num.sqrt(a)

    return f, a


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

    def plot_all(self, ax=None, colors=None, alpha=1., legend=True):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        count = 0
        i_fail = 0
        for tracer, fxfy in self.spectra:
            if fxfy is None:
                ax.text(0.01, 0.90+i_fail, 'no data', transform=ax.transAxes, verticalalignment='top')
                i_fail -= 0.1
                continue
            fx, fy = num.vsplit(fxfy, 2)
            if colors:
                color = colors[tracer]
            else:
                color = 'black'
            ax.plot(fx.T, fy.T, label=tracer.label(), color=color, alpha=alpha)
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


class TracerColor():
    def __init__(self, attr, color_map=mpl.cm.coolwarm, tracers=None):
        self.tracers = tracers
        self.attr = attr
        self.color_map = color_map
        self.min_max = None

        self.set_range()

    def __getitem__(self, tracer):
        v = getattr_dot(tracer, self.attr)
        return self.color_map(self.proj(v))

    def proj(self, v):
        minv, maxv = self.min_max
        return (v-minv)/(maxv-minv)

    def set_range(self):
        vals = [getattr_dot(trs, self.attr) for trs in self.tracers]
        self.min_max = (min(vals), max(vals))


class UniqueColor():
    def __init__(self, color_map=mpl.cm.coolwarm, tracers=None):
        self.tracers = tracers
        self.color_map = color_map
        self.mapping = dict(zip(self.tracers, num.linspace(0, 1, len(self.tracers))))

    def __getitem__(self, tracer):
        return self.color_map(self.mapping[tracer])


class SyntheticCouple():
    def __init__(self, master_slave, method='mtspec', use_common=False):
        check_method(method)
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

    def cc_coef(self):
        if self._cc is None:
            (tr1, _) = self.spectra.spectra[0]
            (tr2, _) = self.spectra.spectra[1]
            ctr = trace.correlate(tr1.processed, tr2.processed, mode='same',
                                  normalization='normal')
            t, self._cc = ctr.max()
        return self._cc

    def process(self, **pp_kwargs):
        ready = []
        length = -1
        for i, tracer in enumerate(self.master_slave):
            tr = tracer.process(**pp_kwargs)
            if tr is False or isinstance(tr, str):
                self.spectra.spectra.append((tracer, tr))
                self.good = tr
                continue

            if length<(tr.tmax-tr.tmin):
                length = tr.tmax-tr.tmin

            ready.append((tr, tracer))

        for tr, tracer in ready:
            f, a = self.get_spectrum(tr, tracer, length)
            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

    def get_spectrum(self, tr, tracer, wantlength):
        length = tr.tmax - tr.tmin
        diff = wantlength - length
        tr.extend(tmin=tr.tmin-diff, tmax=tr.tmax, fillmethod='repeat')
        return spectralize(tr, self.method, tracer.chopper, tracer.tinc)

    def plot(self, colors, **kwargs):
        fn = kwargs.pop('savefig', False)
        fig = plt.figure(figsize=(4, 6.5))
        ax = fig.add_subplot(3, 1, 3)
        #self.spectra.plot_all(ax, colors=colors, legend=False)
        self.spectra.plot_all(ax, legend=False)
        if self.invert_data:
            Q = self.invert_data[-1]
            ax.text(0.01, 0.01, "1/Q=%1.6f" % Q,
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    transform=ax.transAxes)
        yshift=0
        info_str = ''
        for i, tracer in enumerate(self.master_slave):
            ax = fig.add_subplot(3, 1, 1+i)
            ax.set_xlabel('time [s]')
            tr = tracer.processed
            otime = tracer.source.time
            plot_traces(tr=tr, t_shift=-otime, ax=ax, label=tracer.label(), color='black')
            #plot_traces(tr=tr, t_shift=-otime, ax=ax, label=tracer.label(), color=colors[tracer])
            ax.text(0.01, 0.01, 'Ml=%1.1f, station: %s'
                    %( tracer.source.magnitude, tracer.target.codes[1]),
                    size=8,
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes )
            #info_str += "\notime: %s,\n M: %1.1f, s: %s" % (util.time_to_str(otime),
            #                                                tracer.source.magnitude,
            #                                                ".".join(tracer.target.codes))

        #ax = fig.add_subplot(4, 1, 4)
        #info_str += '\ncc=%s' % self.cc_coef()
        #ax.text(0.01, 0.01, info_str, verticalalignment='bottom', horizontalalignment='left',
        #        transform=ax.transAxes)
        #ax.axis('off')
        plt.tight_layout()
        #fig.subplots_adjust(hspace=0.21, wspace=0.21)
        if fn:
            #fig.tight_layout()
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


def spectral_ratio(couple):
    '''Ratio of two overlapping spectra.

    i is the mean intercept.
    s is the slope ratio of each individual linregression.'''
    assert len(couple.spectra.spectra)==2, '%s spectra found: %s' % (len(couple.spectra.spectra), couple.spectra.spectra)
    fx = []
    fy = []
    s = None
    if couple.use_common:
        cfmin = max([couple.spectra.spectra[1][0].fmin, couple.spectra.spectra[0][0].fmin])
        cfmax = min([couple.spectra.spectra[1][0].fmax, couple.spectra.spectra[0][0].fmax])
    else:
        cfmin = None
        cfmax = None

    for tr, fxfy in couple.spectra.spectra:
        fs, a = fxfy
        if cfmin and cfmax:
            fmin = cfmin
            fmax = cfmax
        else:
            fmin = max(tr.fmin, fs.min())
            fmax = min(tr.fmax, fs.max())
        ind = limited_frequencies_ind(fmin, fmax, fs)
        slope, interc, r, p, std = linregress(fs[ind], num.log(a[ind]))
        if not s:
            i = interc
            s = num.exp(slope)
        else:
            i += interc
            s -= num.exp(slope)
        fx.append(fs)
    return i/2., -s, num.sort(num.hstack(fx))


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
#
#    a_s = []
#    f_s = []
#    for tr, fxfy in couple.spectra.spectra:
#        fs, a = fxfy
#        if cfmin is not None and cfmax is not None:
#            fmin = cfmin
#            fmax = cfmax
#        else:
#            fmin = max(tr.fmin, fs.min())
#            fmax = min(tr.fmax, fs.max())
#        ind = limited_frequencies_ind(fmin, fmax, fs)
#        a_s.append(a[ind])
#        f_s.append(fs[ind])
#        fx.append(fs[ind])
#
#    aratio = a_s[1]/a_s[0]
#    slope, interc, r, p, std = linregress(fs[ind], num.log(aratio))
#    return interc, slope, num.sort(num.hstack(fx))



class QInverter:
    def __init__(self, couples, cc_min=0.8):
        if len(couples)==0:
            raise Exception('Empty list of test couples')
        self.couples = couples
        self.cc_min = cc_min

    def invert(self):
        self.allqs = []
        self.ratios = []
        widgets = ['regression analysis', progressbar.Percentage(), progressbar.Bar()]
        pb = progressbar.ProgressBar(maxval=len(self.couples), widgets=widgets).start()
        for i_c, couple in enumerate(self.couples):
            cc_coef = couple.cc_coef()
            if cc_coef< self.cc_min:
                logger.info('lower than cc threshold. skip')
                continue
            else:
                logger.info('higher than cc threshold. ')
            pb.update(i_c+1)
            interc, slope, fx = spectral_ratio(couple)
            dt = couple.delta_onset()
            Q = num.pi*dt/slope
            if num.isnan(Q):
                logger.warn('Q is nan')
                continue
            couple.invert_data = (dt, fx, slope, interc, 1./Q)
            self.allqs.append(1./Q)
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
        ax.hist(filtered, bins=99)
        ax.set_xlabel('Q', size=fsize)
        ax.set_ylabel('counts', size=fsize)
        txt ='median: %1.1f\n$\sigma$: %1.1f' % (median, num.std(self.allqs))
        ax.text(0.01, 0.99, txt, size=fsize, transform=ax.transAxes, verticalalignment='top')
        ax.axvline(0, color='black', alpha=0.3)
        ax.axvline(median, color='blue')
        if q_threshold is not None:
            ax.set_xlim([q_threshold, q_threshold])
        elif relative_to=='median':
            ax.set_xlim([median-q_threshold, median+q_threshold])
        elif relative_to=='mean':
            ax.set_xlim([mean-q_threshold, mean+q_threshold])

        if want_q:
            ax.axvline(want_q, color='r')
        return

    def analyze(self, couples=None):
        # by meanmag
        Qs = []
        mags = []
        magdiffs = []
        by_target = {}
        cc = []
        for c in self.couples:
            if c.invert_data is None:
                continue
            Q = c.invert_data[-1]
            Qs.append(Q)
            cc.append(c.cc_coef())
            meanmag = (c.master_slave[0].source.magnitude+c.master_slave[1].source.magnitude)/2.
            mags.append(meanmag)

            magdiff = abs(c.master_slave[0].source.magnitude-c.master_slave[1].source.magnitude)
            magdiffs.append(magdiff)
            try:
                by_target['%s-%s'%(c.master_slave[0].target.codes[1], c.master_slave[1].target.codes[1]) ].append(Q)
            except KeyError:
                by_target['%s-%s'%(c.master_slave[0].target.codes[1], c.master_slave[1].target.codes[1]) ] = [Q]
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('cc coef vs Q')
        ax.plot(cc, Qs, 'bo')

        ax = fig.add_subplot(2, 2, 2)
        ax.plot(mags, Qs, 'bo')
        ax.set_title('mean magnitude vs Q')

        ax = fig.add_subplot(2, 2, 3)
        ax.plot(magdiffs, Qs, 'bo')
        ax.set_title('magnitude difference vs Q')

        nrow = 3
        ncolumns = int(len(by_target)/nrow)+1
        fig = plt.figure()
        i = 0
        q_threshold = 2000
        for k, v in by_target.items():
            if len(v)<3:
                continue
            ax = fig.add_subplot(nrow, ncolumns, i)
            ax.hist(filter(lambda x: x<=q_threshold, v), bins=20)
            ax.set_title(k)
            i += 1


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



def process_couple(args):
    testcouple, n_repeat, noise = args
    testcouple.process(repeat=n_repeat, noise=noise)
    return testcouple


def invert_test_2D_parallel(noise_level=0.001):
    print '-------------------invert_test_2D -------------------------------'
    #builder = Builder(cache_dir='cache-parallel')
    builder = Builder(cache_dir='muell-cache')
    #x_targets = num.array([1000., 10000., 20000., 30000., 40000., 50000.])
    #d1s = num.linspace(1000., 80000., 15)
    d1s = [12000.]
    ##### ready
    #z1 = 11.*km
    #z2 = 13.*km
    ###########
    z1 = 11.*km
    z2 = 13.*km
    ############

    #if True:
    #    d2s = d1s*z2/z1
    parallel = True

    #x_targets = num.array([d1, d2])
    #y_targets = num.array([0.]*len(x_targets))

    lat = 50.2059
    lon = 12.5152
    sampling_rate = 250
    #sampling_rate = 100.
    time_window = 15.
    #time_window = 24
    #n_repeat = 1000
    n_repeat = 100
    sources = []
    method = 'mtspec'
    #method = 'pymutt'
    #method = 'psd'
    strike = 170.
    dip = 70.
    rake = -30.
    fmin = 50.
    fmax = 100.
    source_mech = qseis.QSeisSourceMechMT(mnn=1E10, mee=1E10, mdd=1E10)
    #source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)
    #source_mech.m_dc *= 1E10
    earthmodel = 'models/constantall.nd'
    #earthmodel = 'models/qplayground_30000m_simple_gradient_fh_wl.nd'
    #earthmodel = 'models/inv_test2_simple.nd'
    #earthmodel = 'models/inv_test6.nd'
    mod = cake.load_model(earthmodel)
    component = 'r'
    target_kwargs = {
        'elevation': 0., 'codes': ('', 'KVC', '', 'Z'), 'store_id': None}

    #targets = xy2targets(x_targets, y_targets, lat, lon, 'Z',  **target_kwargs)
    #logger.info('Using %s targets' % (len(targets)))

    tracers = []
    source_depths = [z1, z2]
    p_chopper = Chopper('first(p|P)', fixed_length=0.4, phase_position=0.5,
                        phaser=PhasePie(mod=mod))
    magnitude = 2.

    sources = [DCSourceWid(lat=float(lat),
                        lon=float(lon),
                        depth=sd,
                        strike=strike,
                        dip=dip,
                        rake=rake,
                        magnitude=magnitude,
                        stf=get_stf(magnitude, type='triangular'),
                        ) for sd in source_depths]
    pairs = []

    for i in xrange(len(d1s)):
        print("%s/%s" %(i, len(d1s)))
        if parallel:
            d1 = d1s[i]
            d2 = d1*z2/z1
        else:
            d1 = d1s[i]
            d2 = d1

        targets = xy2targets([d1, d2], [0., 0.], lat, lon, 'Z',  **target_kwargs)
        pair = []
        for itarget, target in enumerate(targets):
            config = qseis.QSeisConfigFull.example()
            config.id='C0%s' % (i)
            #config.time_region = (meta.Timing(8-time_window/2.), meta.Timing(8+time_window/2.))
            config.time_region = (meta.Timing(8-time_window/2.), meta.Timing(8+time_window/2.))
            config.time_window = time_window
            config.nsamples = (sampling_rate*int(config.time_window))+1
            config.earthmodel_1d = mod
            config.source_mech = source_mech
            config.sw_sampling = 1
            #config.gradient_resolution_vp = 120.
            config.wavenumber_sampling = 6.
            config.validate()
            slow = p_chopper.arrival(sources[itarget], target).p/(cake.r2d*cake.d2m/cake.km)
            config.slowness_window = [slow*0.6, slow*0.9, slow*1.1, slow*1.4]
#
            tracer = Tracer(sources[itarget], target, p_chopper, config=config, fmin=fmin, fmax=fmax,
                            component=component)
            tracers.append(tracer)
            pair.append(tracer)
        pairs.append(pair)


    #qs = qseis.QSeisConfig()
    #qs.qseis_version = '2006a'
    #qs.sw_flat_earth_transform = 1
    #slow = p_chopper.arrival(sources[itarget], target).p/(cake.r2d*cake.d2m/cake.km)
    #qs.slowness_window = [slow*0.5, slow*0.9, slow*1.1, slow*1.5]
    #qs.sw_algorithm = 1
    ##neu 
    #qs.aliasing_suppression_factor = 0.4
    ##neu
    #qs.filter_surface_effects = 1

    colors = UniqueColor(tracers=tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('locations_model_parallel%s.png'%parallel)
    #plt.show()
    tracers = builder.build(tracers, snuffle=False)
    #noise = RandomNoise(noise_level)
    testcouples = []
    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair, method=method)
        testcouple.process()#repeat=n_repeat, noise=noise)
        testcouples.append(testcouple)

    #pool = multiprocessing.Pool()
    #args = [(tc, n_repeat, noise) for tc in testcouples]
    #for arg in args:
    #    process_couple(arg)
    print 'Fix parallel version'
    #pool.map(process_couple, args)
    dist_vs_Q = []
    for i_tc, testcouple in enumerate(testcouples):

        infos = '''
        Strike: %s\nDip: %s\n Rake: %s\n Sampling rate [Hz]: %s\n 
        noise_level: %s\nmethod: %s
        ''' % (strike, dip, rake, sampling_rate, noise_level, method)

        # return the list of noisy Qs. This should be cleaned up later.....!
        Qs = testcouple.plot(infos=infos, colors=colors, noisy_Q=False, fmin=fmin, fmax=fmax)
        fig = plt.gcf()
        d1, d2, z1, z2 = testcouple.get_target_distances_and_depths()
        mean_dist = num.mean((d1, d2))/1000. 
        fig.savefig('1D_mdistkm%i.png' % (mean_dist), dpi=240)
        dist_vs_Q.append((mean_dist, num.median(Qs)))
    fig = plt.figure(figsize=(6, 4.6))
    ax = fig.add_subplot(111)
    for val in dist_vs_Q:
        ax.plot(val[0], num.abs(val[1]), 'bo')
    ax.set_title('Distance vs Q (parallel: %s)' % parallel)
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('abs(Q)')
    fig.savefig('distance_vs_q_parallel%s.png' % parallel, dpi=240)
    plt.show()
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    inverter.plot()
    plt.show()



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



def dbtest(noise_level=0.0000000000005):
    print '-------------------db test-------------------------------'
    use_real_shit = True
    use_extended_sources = False
    use_responses = True                            # 2 test
    load_coupler = True
    test_scenario = True
    #want_station = ('cz', 'nkc', '')
    #want_station = ('cz', 'kac', '')
    want_station = 'all'
    lat = 50.2059
    lon = 12.5152
    sources = []
    method = 'mtspec'
    #method = 'sine_psd'
    min_magnitude = 2.
    max_magnitude = 6.
    fminrange = 20.
    #use_common = False
    use_common = True
    fmax_lim = 80.
    #zmax = 10700
    fmin = 31.
    fmin = Magnitude2fmin.setup(lim=fmin)
    fmax = 90.
    window_by_magnitude = Magnitude2Window.setup(0.2, 0.02)
    quantity = 'displacement'
    #store_id = 'qplayground_total_2'
    #store_id = 'qplayground_total_2_q25'
    #store_id = 'qplayground_total_2_q400'
    #store_id = 'qplayground_total_1_hr'
    #store_id = 'qplayground_total_4_hr'
    #store_id = 'qplayground_total_4_hr_full'
    store_id = 'ahfullgreen_3'

    # setting the dc components:

    strikemin = 160
    strikemax = 180
    dipmin = -60
    dipmax = -80
    rakemin = 20
    rakemax = 40

    engine = LocalEngine(store_superdirs=['/data/stores', '/media/usb/stores'])
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
    tt_mu = 0.
    tt_sigma = 0.0001
    save_figs = True
    nucleation_radius = 0.1

    # distances used if not real sources:
    if test_scenario:
        distances = num.linspace(config.distance_min+gf_padding, config.distance_max-gf_padding, 12)
        source_depths = num.linspace(zmin, zmax, 12)
    else:
        distances = num.arange(config.distance_min+gf_padding, config.distance_max-gf_padding, 200)
        source_depths = num.arange(zmin, zmax, 200)

    perturbation = UniformTTPerturbation(mu=tt_mu, sigma=tt_sigma)
    perturbation.plot()
    p_chopper = Chopper('first(p)', phase_position=0.3,
                        by_magnitude=window_by_magnitude,
                        phaser=PhasePie(mod=mod))
    stf_type = 'brunes'
    #stf_type = 'halfsin'
    #stf_type =  None
    tracers = []
    want_phase = 'p'
    fn_coupler = 'dummy_coupling.yaml'
    #fn_coupler = None
    fn_noise = '/media/usb/webnet/mseed/noise.mseed'
    fn_records = '/media/usb/webnet/mseed'
    #if use_real_shit:
    if False:
        noise = Noise(files=fn_noise, scale=noise_level)
        noise_pile = pile.make_pile(fn_records)
    else:
        noise = RandomNoiseConstantLevel(noise_level)
        noise_pile = None

    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    all_depths = [e.depth for e in events]
    some_depths = [d/1000. for d in all_depths if d>8500]
    fig = plt.figure(figsize=(6,8))
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
        filtrate = Filtrate.load(filename=fn_coupler)
        sources = filtrate.sources
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
                    mag = float(1.+num.random.random()*0.2)
                    strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
                                                                             dipmin, dipmax,
                                                                             rakemin, rakemax)
                    mt = moment_tensor.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)
                    e = model.Event(lat=lat, lon=lon, depth=float(sd), moment_tensor=mt)
                    if use_extended_sources is True:
                        sources.append(e2extendeds(e, north_shift=float(d),
                        #print 'use line source!'
                        #sources.append(e2linesource(e, north_shift=float(d),
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
    pairs_by_rays = coupler.filter_pairs(4, 1200, data=coupler.filtrate, max_mag_diff=0.1)
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
    pairs = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1 = p
        fmin2 = None
        pair = []
        for sx in [s1, s2]:
            fmin1 = fmin_by_magnitude(sx.magnitude)
            #fmax = min(fmax_lim, vp/fresnel_lambda(totald, td, pd))
            #print 'test me, change channel code id to lqt'
            #t.dip = -90. + i1
            #t.azimuth = t.azibazi_to(sx)[1]
            tracer1 = Tracer(sx, t, p_chopper, channel=channel, fmin=fmin1,
                             fmax=fmax, want=quantity,
                             perturbation=perturbation.perturb(0))

            dist1, depth1 = tracer1.get_geometry()
            if dist1< dist_min or dist1>dist_max:
                break
            if fmax-fmin1<fminrange:
                break
            pair.append(tracer1)
            tracers.extend(pair)
            pairs.append(pair)
    if len(tracers)==0:
        raise exception('no tracers survived the assessment')

    builder = Builder()
    tracers = builder.build(tracers, engine=engine, snuffle=False)
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
        testcouple = SyntheticCouple(master_slave=pair, method=method, use_common=use_common)
        testcouple.process(noise=noise)
        if len(testcouple.spectra.spectra)!=2:
            logger.warn('not 2 spectra in test couple!!!! why?')
            continue
        testcouples.append(testcouple)
    pb.finish()
    testcouples = filter(lambda x: x.good==True, testcouples)
    #outfn = 'testimage'
    #plt.gcf().savefig('output/%s.png' % outfn)
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    for i, testcouple in enumerate(num.random.choice(testcouples, 10)):
        fn = 'synthetic_tests/%s/example_%s_%s.png' % (want_phase, store_id, str(i).zfill(2))
        testcouple.plot(infos=infos, colors=colors, noisy_q=False, savefig=fn)
    inverter.plot(q_threshold=800, relative_to='median', want_q=want_q)
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

def s2t(s, channel='Z', store_id=None):
    return Target(lat=s.lat, lon=s.lon, depth=s.depth, elevation=s.elevation,
                  codes=(s.network, s.station, s.location, channel), store_id=store_id)


def e2extendeds(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
        s, d, r = mt.both_strike_dip_rake()[0]
    else:
        mt = False
        mag = e.magnitude
        s, d, r = None, None, None

    a = source_radius([mag])
    if nucleation_radius is not None:
        nucleation_x, nucleation_y = (num.random.random(2)-0.5)*2.*nucleation_radius
        nucleation_x = float(nucleation_x)
        nucleation_y = float(nucleation_y)
    else:
        nucleation_x, nucleation_y = None, None
    velocity = 3500.
    if stf_type=='brunes':
        # mu nachschauen!
        # beta aus Modell
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None

    stf = get_stf(mag, type=stf_type)
    return RectangularBrunesSource(
       lat=float(e.lat), lon=float(e.lon), depth=float(e.depth), north_shift=float(north_shift),
       east_shift=float(east_shift), time=float(e.time), width=float(a[0]), length=float(a[0]),
       strike=float(s), dip=float(d), rake=float(r), magnitude=float(mag), brunes=brunes,
        velocity=float(velocity),
       nucleation_x=float(nucleation_x), nucleation_y=float(nucleation_y), stf=stf)
    #return RectangularSource(
    #   lat=e.lat, lon=e.lon, depth=e.depth, north_shift=north_shift,
    #   east_shift=east_shift, time=e.time, width=float(a[0]), length=float(a[0]),
    #   strike=mt.strike1, dip=mt.dip1, rake=mt.rake1, magnitude=mag,
    #   nucleation_x=nucleation_x, nucleation_y=nucleation_y, stf=stf)


#def e2extendeds(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
#    if e.moment_tensor:
#        mt = e.moment_tensor
#        mag = mt.magnitude
#    else:
#        mt = False
#        mag = e.magnitude
#    a = source_radius([mag])
#    #d = num.sqrt(a[0])
#    print 'magnitude: ', mag
#    print 'source radius: ', a
#    if nucleation_radius is not None:
#        nucleation_x, nucleation_y = (num.random.random(2)-0.5)*2.*nucleation_radius
#        nucleation_x = float(nucleation_x)
#        nucleation_y = float(nucleation_y)
#    else:
#        nucleation_x, nucleation_y = None, None
#    #nucleation_x = 0.95
#    #nucleation_y = 0.
#    stf = get_stf(mag, type=stf_type)
#    print nucleation_x, nucleation_y
#    print mt.strike1, mt.strike2
#    print mt.dip1, mt.dip2
#    print mt.rake1, mt.rake2
#    print '.'*80
#    return CircularBrunesSource(
#       lat=e.lat, lon=e.lon, depth=e.depth, north_shift=north_shift,
#       east_shift=east_shift, time=e.time, radius=float(a[0]),
#       strike=mt.strike1, dip=mt.dip1, rake=mt.rake1, magnitude=mag,
#       nucleation_x=nucleation_x, nucleation_y=nucleation_y, stf=stf)


def e2s(e, north_shift=0., east_shift=0., stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
    else:
        mt = False
        mag = e.magnitude

    stf = get_stf(mag, type=stf_type)
    if stf_type=='brunes':
        # mu nachschauen!
        # beta aus Modell
        velocity = 3500.
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None
    s = DCSourceWid.from_pyrocko_event(e)
    s.brunes = brunes
    s.north_shift = north_shift
    s.east_shift = east_shift
    s.magnitude = mag
    s.stf = stf
    return s

def paired_sources_dict(paired_sources):
    paired_source_dict = {}
    for s in paired_sources:
        if not s in paired_source_dict.keys():
            paired_source_dict[s] = 1
        else:
            paired_source_dict[s] += 1
    return paired_source_dict

def apply_webnet():
    print '-------------------apply  -------------------------------'


    # aus dem GJI 2015 paper ueber vogtland daempfung von Gaebler:
    # Shearer 1999: Qp/Qs = 2.25 (intrinsic attenuation)
    load_coupler = False
    builder = Builder()
    #method = 'sine_psd'
    method = 'mtspec'
    use_common = True
    fmax = 110
    fminrange = 30

    vp = 6000.
    fmin_by_magnitude = Magnitude2fmin.setup(lim=30)
    min_magnitude = 0.3
    max_magnitude = 4.
    #min_magnitude = 0.
    mod = cake.load_model('models/earthmodel_malek_alexandrakis.nd')
    #markers = PhaseMarker.load_markers('/media/usb/webnet/meta/phase_markers2008_extracted.pf')
    #events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    markers = PhaseMarker.load_markers('/data/meta/webnet_reloc/hypo_dd_markers.pf')
    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    print len(events)
    #events = num.random.choice(events, num_use_events)
    events = filter(lambda x: x.magnitude>= min_magnitude, events)
    events = filter(lambda x: x.magnitude<= max_magnitude, events)
    print '%s events'% len(events)
    reset_events(markers, events)
    pie = PickPie(markers=markers, mod=mod, event2source=e2s, station2target=s2t)
    stations = model.load_stations('/data/meta/stations.pf')
    want_phase = 'P'
    #window_length = {'S': 0.4, 'P': 0.4}
    window_by_magnitude = Magnitude2Window.setup(0.08, 2.8)
    #window_by_magnitude = Magnitude2Window.setup(0.1, 2.8)
    phase_position = {'S': 0.2, 'P': 0.3}
    #window_length = {'S': 0.4, 'P': 0.4}
    #phase_position = {'S': 0.2, 'P': 0.2}

    # in order to rotate into lqt system
    rotate_channels = {'in_channels': ('SHZ', 'SHN', 'SHE'),
                       'out_channels': ('L', 'Q', 'T')}

    channels = {'P': 'SHZ', 'S': 'SHE' }
    channel = channels[want_phase.upper()]
    pie.process_markers(phase_selection=want_phase, stations=stations, channel=channel)
    p_chopper = Chopper(
        startphasestr=want_phase, by_magnitude=window_by_magnitude,
        phase_position=phase_position[want_phase.upper()], phaser=pie)
    tracers = []

    if load_coupler:
        logger.warn('LOAD COUPLER')
    #fn_coupler = 'dummy_webnet_pairing_%s.yaml' % want_phase
    fn_coupler = 'webnet_pairing_noinci%s.yaml' % want_phase

    fn_mseed = '/media/usb/webnet/mseed'
    ignore = ['*.STC.*.SHZ']

    data_pile = pile.make_pile(fn_mseed)
    if data_pile.tmax==None or data_pile.tmin == None:
        raise Exception('failed reading mseed')
    # webnet Z targets:
    targets = [s2t(s, channel) for s in stations]
    if load_coupler:
        filtrate = Filtrate.load(filename=fn_coupler)
        sources = filtrate.sources
        coupler = Coupler(filtrate)
    else:
        coupler = Coupler()
        sources = [e2s(e) for e in events]
        coupler.process(
            sources, targets, mod, [want_phase, want_phase.lower()],
            ignore_segments=True, dump_to=fn_coupler)

    fig, ax = Animator.get_3d_ax()
    #print coupler.filtrate
    pairs_by_rays = coupler.filter_pairs(4., 1000., data=coupler.filtrate,
                                         ignore=ignore, max_mag_diff=0.5)
    paired_sources = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1 = p
        paired_sources.extend([s1, s2])

    paired_source_dict = paired_sources_dict(paired_sources)
    Animator.plot_sources(sources=paired_source_dict, reference=coupler.hookup, ax=ax, alpha=1)
    #pb = progressbar.ProgressBar(maxval=len(pairs_by_rays)-1, widgets=widgets).start()
    #for i_r, r in enumerate(pairs_by_rays):
    #    e1, e2, t, td, pd, segments = r
    #    Animator.plot_ray(segments, ax=ax)
    #    pb.update(i_r)
    #pb.finish()
    goods = 0
    bads = 0
    testcouples = []
    print 'No Fresnel check!'
    for r in pairs_by_rays:
        s1, s2, t, td, pd, totald, i1 = r
        #fmax = vp/fresnel_lambda(totald, td, pd)
        fmin1 = fmin_by_magnitude(s1.magnitude)
        tracer1 = DataTracer(data_pile=data_pile, source=s1, target=t,
                             chopper=p_chopper, channel=channel, fmin=fmin1,
                             fmax=fmax, incidence_angle=i1)
                             #rotate_channels=rotate_channels)
        tracer1.setup_data()

        fmin2 = fmin_by_magnitude(s2.magnitude)
        tracer2 = DataTracer(data_pile=data_pile, source=s2, target=t,
                             chopper=p_chopper, channel=channel, fmin=fmin2,
                             fmax=fmax, incidence_angle=i1)
                             #rotate_channels=rotate_channels)
        tracer2.setup_data()
        if fmax-fmin1<fminrange or fmax-fmin2<fminrange:
            continue
        else:
            pair = [tracer1, tracer2]
            testcouple = SyntheticCouple(master_slave=pair,
                                         method=method)
            testcouple.process()
            if testcouple.good:
                testcouples.append(testcouple)
                goods += 1
            else:
                bads += 1
            tracers.extend(pair)
            #pairs.append(pair)
    print 'good/bad' , goods, bads
    colors = UniqueColor(tracers=tracers)
    tracers = builder.build(tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('location_model_db1.png', dpi=200)
    #plt.show()
    #pb = progressbar.ProgressBar(maxval=len(pairs), widgets=pb_widgets('processing couples')).start()
    #goods = 0
    #bads = 0
    #for i_p, pair in enumerate(pairs):
    #    pb.update(i_p)
    #    if len(pair)!=2:
    #        import pdb
    #        pdb.set_trace()
    #    #testcouple = SyntheticCouple(master_slave=pair, method=method)
    #    #testcouple.process()
    #    #if testcouple.good:
    #    #    testcouples.append(testcouple)
    #    #    goods += 1
    #    #else:
    #    #    bads += 1
    #print 'goods', goods
    #print 'bads', bads
    #pb.finish()
    #plt.show()
    #testcouples = filter(lambda x: x.delta_onset()>0.06, testcouples)
    inverter = QInverter(couples=testcouples, cc_min=0.8)
    inverter.invert()
    good_results = filter(lambda x: x.invert_data is not None, testcouples)
    for i, tc in enumerate(num.random.choice(good_results, 10)):
        fn = 'application/%s/example_%s.png' % (want_phase, str(i).zfill(2))
        tc.plot(infos=infos, colors=colors, savefig=fn)
    inverter.plot()#q_threshold=600, relative_to='median')
    fig = plt.gcf()
    fig.savefig('application/%s/hist_application.png' % want_phase, dpi=600)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #couples = inverter.couples
    inverter.analyze()



def e2linesource(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
        s, d, r = mt.both_strike_dip_rake()[0]
    else:
        mt = False
        mag = e.magnitude
        s, d, r = None, None, None

    #a = source_radius([mag])
    #d = num.sqrt(a[0])
    velocity = 3500.
    if stf_type=='brunes':
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None

    stf = get_stf(mag, type=stf_type)
    return LineSource(
       lat=float(e.lat), lon=float(e.lon), depth=float(e.depth), north_shift=float(north_shift),
       east_shift=float(east_shift), time=float(e.time), i_sources=10,
       strike=float(s), dip=float(d), rake=float(r), magnitude=float(mag), brunes=brunes,
        velocity=velocity, stf=stf)



if __name__=='__main__':
    #invert_test_1()
    #qpqs()

    # DIESER:
    #invert_test_2()
    #invert_test_2D(noise_level=0.0000001)
    #invert_test_2D_parallel(noise_level=0.1)
    #dbtest()
    apply_webnet()
    plt.show()
    #noise_test()
    #qp_model_test()
    #constant_qp_test()
    #sdr_test()
    #vp_model_test()

__all__ = '''
DCSourceWid
'''.split()
