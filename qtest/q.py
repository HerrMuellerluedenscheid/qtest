import numpy as num
import os
import progressbar
import logging

import matplotlib as mpl # noqa
mpl.use('Agg')
mpl.rc('ytick', labelsize=10)
mpl.rc('xtick', labelsize=10)

import matplotlib.pyplot as plt # noqa
plt.style.use('ggplot')
from pyrocko.gf import Target
from pyrocko import orthodrome
from pyrocko import cake
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy import signal, interpolate, optimize
from qtest.micro_engine import DataTracer, DDContainer
from qtest.invert import ModelWithValues, DiscretizedModelNNInterpolation
from autogain.autogain import PhasePie, PickPie
from distance_point2line import Coupler, Animator
from util import Magnitude2Window, Magnitude2fmin, fmin_by_magnitude


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
        tr.set_ydata(tr.ydata/num.max(num.abs(tr.ydata)))
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
            fx, fy, f_noise = num.vsplit(fxfy, 3)
            if colors != 'rb':
                color = colors[tracer]
            else:
                color = colors[count]
            ax.plot(fx.T[1:], fy.T[1:], label=tracer.label(), color=color, alpha=alpha)
            ax.plot(fx.T[1:], f_noise.T[1:], label=tracer.label(), color=color, alpha=alpha)
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
        #for tr, fxfy in self.spectra:
        #    fx, fy = fxfy
        #    f = interpolate.interp1d(fx, fy)
        #    a_s[i][:] = f(f_use)

        return f_use, a_s


class SyntheticCouple():
    def __init__(self, master_slave, method='mtspec', use_common=False, ray_segment=None):
        self.method = method
        self.master_slave = master_slave
        self.spectra = Spectra()
        self.noisy_spectra = Spectra()
        self.fit_function = None
        self.colors = None
        self.good = True
        self.snr_min = 0
        self.use_common = use_common
        self.ray_segment = ray_segment
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
        for i, tracer in enumerate(self.master_slave):
            tr = tracer.process(**pp_kwargs)
            if tr is False or isinstance(tr, str):
                self.spectra.spectra.append((tracer, tr))
                self.good = tr
                continue

            ready.append((tr, tracer))

        for tr, tracer in ready:
            #if self.normalize_waveforms:
            #    ynew = tr.get_ydata()
            #    self.normalization_factor = num.max(num.abs(ynew))
            #    ynew /= self.normalization_factor
            #    tr.set_ydata(ynew)

            #f, a = self.get_spectrum(tr, tracer, length)
            f, a_noise, a_signal = tracer.processed_spectrum(self.method, filters=self.filters)

            #f_n, a_n = tracer.noise_spectrum(self.method, filters=self.filters)
            #power_noise = util.power(f_n, a_n)
            print f.shape, a_noise.shape, a_signal.shape
            fxfy = num.vstack((f, a_noise, a_signal))
            self.spectra.spectra.append((tracer, fxfy))

    def plot(self, colors=None, **kwargs):
        fn = kwargs.pop('savefig', False)
        #fig = plt.figure(figsize=(4, 6.5))
        #ax = fig.add_subplot(3, 1, 3)
        fig = plt.figure(figsize=(7.5, 3.5))
        ax = fig.add_subplot(1, 2, 2)
        if not colors:
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


def spectral_ratio(couple):
    '''Ratio of two overlapping spectra.

    The alternative method.

    i is the mean intercept.
    s is the slope ratio of each individual linregression.'''
    assert len(couple.spectra.spectra)==2, '%s spectra found: %s' % (len(couple.spectra.spectra), couple.spectra.spectra)

    cfmin, cfmax = couple.frequency_range_work()


    a_s = []
    f_s = []
    a_noise = []
    for i, (tr, fxfy) in enumerate(couple.spectra.spectra):
        f, a_n, a = fxfy
        indx = num.where(num.logical_and(f>=cfmin, f<=cfmax))
        a_noise.append(a_n[indx])
        a_s.append(a[indx])
        f_s.append(f[indx])

    if any(a_s[0]/a_noise[0] < couple.snr_min) or\
            any(a_s[1]/a_noise[1] < couple.snr_min) or\
            not all(f_s[0] == f_s[1]):
        return None, None, f_s[0], a_s, None, None, None

    f_use = f_s[0]

    #f_use, a_s = couple.spectra.get_interpolated_spectra(cfmin, cfmax)

    slope, interc, r, p, std = linregress(f_use, num.log(a_s[1]/a_s[0]))

    return interc, slope, f_use, a_s, r, p, std

def prepare_data_in_couple(couple):
    tracer_master, tracer_slave = couple.master_slave
    tr1 = tracer_master.setup_data(normalize=couple.normalize_waveforms)
    tr2 = tracer_slave.setup_data(normalize=couple.normalize_waveforms)
    if not tr1 or not tr2:
        logger.debug('tr1 or tr2 are None')
        couple.drop_data()
        return False

    return couple.master_slave

class QInverter3D:
    ''' Uses only the significant bit of the ray segment.'''
    def __init__(self, couples, discretized_grid):
        if len(couples)==0:
            raise Exception('Empty list of test couples')
        self.discretized_grid = discretized_grid
        self.dws_grid = DiscretizedModelNNInterpolation.from_model(self.discretized_grid)

        # Weighting factors dependent on the penetration of each voxel:
        self.dws = []

        # Matrix G. This matrix is equivalent to the times a voxel is
        # penetrated by a ray, or the time a ray spends inside a certain voxel.
        self.penetration_duration = []

        # Matrix d, which is equivalent to the measured t* of that ray segment.
        self.slopes = num.empty(len(couples))

        logger.info('Evaluating model penetration_duration...')
        for icouple, couple in enumerate(couples):
            trs = prepare_data_in_couple(couple)
            if not trs:
                continue
            w = self.discretized_grid.cast_ray(couple.ray_segment, return_quantity='times')

            #self.dws.append(num.ravel(self.dws_grid.cast_ray(couple.ray_segment)))
            self.dws.append(num.ravel(w))

            # Get the 'derivative weighted sum'. This is a formalism to describe the
            # density of rays withing each voxel and will be used as weighting factors
            # in the inversion.
            w = self.discretized_grid.cast_ray(couple.ray_segment)
            self.penetration_duration.append(w)

            interc, slope, fx, a_s, r, p, std = spectral_ratio(couple)
            self.slopes[icouple] = slope
        logger.info('Evaluating model penetration_duration finished')

    def invert(self, qstart=350.):
        ''' run the inversion'''

        def searchold():
            n_G = len(self.weights)
            tstars = num.zeros(n_G)
            errors = num.zeros(n_G)
            for ig in range(n_G):
                # t* = sum(t * q)
                tstar_theo = num.sum(self.weights[ig] * test_model)
                errors[ig] = tstar_theo - slopes[ig]

            # L1 norm
            return num.sum(num.abs(errors))

        nx, ny, nz = self.discretized_grid._shape()
        self.G = num.zeros((len(self.penetration_duration), (nx*ny*nz)))
        for iw, w in enumerate(self.penetration_duration):
            self.G[iw] = num.ravel(w)

        def search(test_model, weights):
            ''':param test_model: flattened model instance'''
            #import pdb
            #pdb.set_trace()
            e = num.sum(num.abs((num.sum(self.G * test_model * weights, axis=1) - self.slopes)))
            print 'error=', e
            print 'test_model average', num.average(test_model)
            print 'test_model min, max', num.min(test_model), num.max(test_model)
            return e

        initial_guess = num.ones((nx, ny, nz)) * 1./qstart/num.pi


        args = ( num.sum(self.dws, axis=0)/num.sum(self.dws),
                )

        if False:
            def print_fun(x, f, accepted):
                print("at minima %.4f accepted %d" % (f, int(accepted)))
            minimizer_kwargs = {"method":"BFGS",
                                "args": args}
            result = optimize.basinhopping(
                search,
                num.ravel(initial_guess),
                callback=print_fun,
                #T=1000000.,
                #stepsize=0.1,
                #interval=10,
                disp=True,
                minimizer_kwargs=minimizer_kwargs)

        if False:
            result = optimize.minimize(search,
                              num.ravel(initial_guess),
                              args=(penetration_duration, ))

        if True:
            result = optimize.root(
                search, num.ravel(initial_guess), args=args, method='lm')

        if False:
            import time
            for q in num.linspace(10,1000, 30):
                print '------------'
                print q*num.pi
                initial_guess = num.ones((nx* ny* nz)) * 1./q
                t1 = time.time()
                print search(initial_guess, penetration_duration=penetration_duration)
                t2 = time.time()
                print searchOLD(initial_guess)
                t3 = time.time()
                print t2-t1, t3-t2


        logger.info('Optimizer finished with state: %s' % result.success)
        logger.info(result.message)

        result_model = ModelWithValues.from_model(self.discretized_grid)
        result_model.values = num.reshape(result.x, (nx, ny, nz))

        return result, result_model


class QInverter:
    def __init__(self, couples, cc_min=0.8, onthefly=False, snr_min=0.):
        if len(couples)==0:
            raise Exception('Empty list of test couples')
        self.couples = couples
        self.cc_min = cc_min
        self.onthefly = onthefly
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
                trs = prepare_data_in_couple(couple)
                if not trs:
                    continue
                else:
                    tracer_master, tracer_slave = trs

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

    def dump_results(self, fn):
        num.savetxt(fn, num.array(self.allqs))

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

