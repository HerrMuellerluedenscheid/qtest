import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rc('ytick', labelsize=8)
mpl.rc('xtick', labelsize=8)

import copy
import progressbar
from pyrocko.gf import meta, DCSource, RectangularSource, Target, LocalEngine, SourceWithMagnitude
from pyrocko.guts import String
from pyrocko import orthodrome
from pyrocko.gui_util import PhaseMarker
from pyrocko import util
from pyrocko import cake, model
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
from distance_point2line import Coupler, Animator, Filtrate
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
    from mtspec import mtspec
    methods_avail.append('mtspec')
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
    if tr:
        ax.plot(tr.get_xdata()+t_shift, tr.get_ydata(), label=label, color=color)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('displ [m]')


def plot_model(mod, ax=None, label='', color=None, parameters=['qp']):
    ax = ax_if_needed(ax)
    z = mod.profile('z')
    colors = 'rgbcy'
    label_mapping = {'qp': 'Q$_p$',
                     'qs': 'Q$_s$',
                     'vp': 'v$_p$',
                     'vs': 'v$_s$'}
    for ip, parameter in enumerate(parameters):
        profile = mod.profile(parameter)
        if ip>=1:
            ax = ax.twiny()
        ax.plot(profile, -z/1000, label=label, c=colors[ip])
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


def M02tr(Mo, stress, vr):
    #stress=stress drop in MPa
    #vr=rupture velocity in m/s
    #Mo = seismic moment calculated with Hanks and Kanamori,1979    
    #Mo=(10.**((3./2.)*(Mw+10.73))/1E+7 #Mo in Nm
    #Calculate rupture length or source radio (m) with Madariaga(1976), stress drop on a circular fault
    Lr = ((7.*Mo)/(16.*stress*1E+6))**(1./3.) #stress Mpa to Nm2
    #Calculate ruputure time in seconds with the rupture velocity
    tr = Lr/vr
    return tr

def get_stf(magnitude=0, stress=0.1, vr=2750., type=None):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    duration = M02tr(Mo, stress, vr)
    if type=='boxcar':
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


class DCSourceWid(DCSource):
    id = String.T(optional=True, default=None)
    brunes = Brune.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)
        #self.stf = get_stf(self.magnitude)
        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

class HaskellSourceWid(RectangularSource):
    id = String.T(optional=True, default=None)
    attitude = String.T(optional=True, default=None)

    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)

    @property
    def is_master(self):
        return self.attitude=='master'

    @property
    def is_slave(self):
        return self.attitude=='slave'


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

def spectralize(tr, method, chopper=None, tinc=None):
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

    elif method=='mtspec':
        a, f = mtspec(data=tr.ydata,
                      delta=tr.deltat,
                      number_of_tapers=5,
                      time_bandwidth=4.,
                      nfft=nextpow2(len(tr.get_ydata())),
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
            ax.plot(fx.T, fy.T, label=tracer.label(), color=color, alpha=alpha)
            ax.axvspan(tracer.fmin, tracer.fmax, facecolor='0.5', alpha=0.1)
            count += 1

        ax.autoscale()
        ax.set_title("$\sqrt{PSD}$")
        ax.set_ylabel("A")
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

    def process(self, **pp_kwargs):
        for tracer in self.master_slave:
            #self.noise_level = pp_kwargs.pop('noise_level', self.noise_level)
            #self.repeat = pp_kwargs.pop('repeat', self.repeat)
            tr = tracer.process(**pp_kwargs)
            if tr is None:
                self.spectra.spectra.append((tracer, tr))
                self.good = False
                continue
            f, a = self.get_spectrum(tr, tracer)
            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

            #for i in xrange(self.repeat):
            #    #tr = tracer.process(**pp_kwargs).copy()
            #    tr = tracer.process(**pp_kwargs)
            #    tr_noise = add_noise(tr, level=self.noise_level)
            #    f, a = self.get_spectrum(tr_noise, tracer)
            #    fxfy = num.vstack((f,a))
            #    self.noisy_spectra.spectra.append((tracer, fxfy))

    def get_spectrum(self, tr, tracer):
        return spectralize(tr, self.method, tracer.chopper, tracer.tinc)

    def plot(self, colors, **kwargs):
        fig = plt.figure(figsize=(3, 6))
        ax = fig.add_subplot(4, 1, 2)
        self.spectra.plot_all(ax, colors=colors, legend=False)
        if self.invert_data:
            Q = self.invert_data[-1]
            ax.text(0.01, 0.01, "Q=%1.1f" % Q, verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes)
        '''
        # Q model right panel:
        ax = fig.add_subplot(3, 2, 2)
        for tracer in self.master_slave:
            plot_model(mod=tracer.config.earthmodel_1d,
                       label=tracer.config.id,
                       color=colors[tracer],
                       ax=ax,
                       parameters=kwargs.get('parameters', ['qp']))
        #ax.set_xlim((ax.get_xlim()[0]*0.9,
        #             ax.get_xlim()[1]*1.1))
        for tr in self.master_slave:
            ax.axhline(-tr.source.depth, ls='--', label='z %s' % tr.config.id,
                       color=colors[tr])
        '''
        ax = fig.add_subplot(4, 1, 1)
        yshift=0
        for tracer in self.master_slave:
            tr = tracer.processed
            otime = tracer.source.time
            plot_traces(tr=tr, t_shift=-otime, ax=ax, label=tracer.label(), color=colors[tracer])
            info_str = "otime: %s, mag: %1.1f, codes: %s, phase: %s" % (util.time_to_str(otime),
                                                                     tracer.source.magnitude,
                                                                     ".".join(tracer.target.codes),
                                                                        tracer.chopper.startphasestr)

            ax.text(0.01, 0.01+yshift, info_str, verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes)
            yshift = 0.07

        #if self.noise_level!=0.:
        #    trs = tracer.process(normalize=kwargs.get('normalize', False)).copy()
        #    trs = add_noise(trs, level=self.noise_level)
        #    plot_traces(tr=trs, ax=ax, label=tracer.config.id, color=colors[tracer])

        Qs = []
        ax = fig.add_subplot(4, 1, 3)
        if kwargs.get('noisy_Q', False):
            fxs, fy_ratios = noisy_spectral_ratios(self)

            xs = []
            ys = []
            ys_ratio = []
            for i in xrange(len(fxs)):
                slope, interc, r_value, p_value, stderr = linregress(fxs[i], num.log(fy_ratios[i]))
                dt = self.delta_onset()
                Q = -1.*num.pi*dt/slope
                if num.abs(Q)>10000:
                    logger.warn('Q pretty large... skipping %s' % Q)
                    continue

                Qs.append(Q)
                xs.append(fxs[i])
                ys.append(interc+fxs[i]*slope)
                ys_ratio.append(num.log(fy_ratios[i]))
            if len(Qs)<2:
                ax.text(0.5, 0.5, 'Too few Qs', transform=ax.transAxes)
                return

            std_q = num.std(Qs)
            maxQ = max(Qs)
            minQ = min(Qs)

            c_m = mpl.cm.coolwarm
            norm = mpl.colors.Normalize(vmin=minQ, vmax=maxQ)
            s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            for i in xrange(len(xs)):
                c = s_m.to_rgba(Qs[i])
                ax.plot(xs[i], ys_ratio[i], color=c, alpha=0.2)
                ax.plot(xs[i], ys[i], color=c, alpha=0.2)
            ax.set_xlabel('f[Hz]')
            ax.set_ylabel('log(A1/A2)')
            cb = plt.colorbar(s_m)
            cb.set_label('Q')
            ax = fig.add_subplot(4, 1, 4)
            v_range = 200
            #ax.hist(num.array(Qs)[num.where(num.abs(Qs)<=v_range)], bins=25)
            try:
                ax.hist(num.array(Qs), bins=25)
            except AttributeError as e:
                logger.warn(e)
            #ax.set_xlim([-120., 120.])
            med = num.median(Qs)
            ax.text(0.01, 0.99, 'median: %1.1f\n $\sigma$: %1.1f' %(med, std_q),
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax.transAxes)
            ax.set_ylabel('Count')
            ax.set_xlabel('Q')
        #ax = fig.add_subplot(3, 2, 6)
        #infos(ax, kwargs.pop('infos'))
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


def spectral_ratio_old(couple):
    '''Ratio of two overlapping spectra'''
    assert len(couple.spectra.spectra)==2
    fx = []
    fy = []
    cfmin = max([couple.spectra.spectra[1][0].fmin, couple.spectra.spectra[0][0].fmin])
    cfmax = min([couple.spectra.spectra[1][0].fmax, couple.spectra.spectra[0][0].fmax])

    for tr, fxfy in couple.spectra.spectra:
        fs, a = fxfy
        fx.append(fxfy[0])
        fy.append(fxfy[1])

    ind0 = limited_frequencies_ind(cfmin, cfmax, fx[0])
    ind1 = limited_frequencies_ind(cfmin, cfmax, fx[1])
    assert all(fx[0][ind0]==fx[1][ind1])
    fy_ratio = fy[0][ind0]/fy[1][ind1]
    #return fx[0][ind0], fy_ratio
    slope, interc, r, p, std = linregress(fx[0][ind0], fy_ratio)
    return interc, slope, fx[0][ind0]


def spectral_ratio(couple):
    '''Ratio of two overlapping spectra.

    i is the mean intercept.
    s is the slope ratio of each individual linregression.'''
    assert len(couple.spectra.spectra)==2
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
        #print r, p, std
        if not s:
            i = interc
            s = num.exp(num.abs(slope))
        else:
            i += interc
            s -= num.exp(num.abs(slope))
        fx.append(fs)
        #fy.append(fxfy[1])
    return i/2., s, num.sort(num.hstack(fx))
    #ind0 = limited_frequencies_ind(fmin, fmax, fx[0])
    #ind1 = limited_frequencies_ind(fmin, fmax, fx[1])
    #assert all(fx[0][ind0]==fx[1][ind0])
    #fy_ratio = fy[0][ind0]/fy[1][ind1]
    #return fx[0][ind0], fy_ratio





class QInverter:
    def __init__(self, couples):
        self.couples = couples

    def invert(self):
        self.allqs = []
        self.ratios = []
        #self.p_values = []
        #self.stderrs = []
        widgets = ['regression analysis', progressbar.Percentage(), progressbar.Bar()]
        pb = progressbar.ProgressBar(maxval=len(self.couples), widgets=widgets).start()
        for i_c, couple in enumerate(self.couples):
            pb.update(i_c+1)
            interc, slope, fx = spectral_ratio(couple)
            #fx, fy_ratio = spectral_ratio(couple)
            #slope, interc, r_value, p_value, stderr = linregress(fx, num.log(fy_ratio))
            #slope = spectral_ratio(couple)
            dt = couple.delta_onset()
            Q = num.abs(num.pi*dt/slope)
            if num.isnan(Q):
                logger.warn('Q is nan')
                continue
            #couple.invert_data = (dt, fx, slope, interc, num.log(fy_ratio), Q)
            couple.invert_data = (dt, fx, slope, interc, Q)
            self.allqs.append(Q)
            #self.p_values.append(p_value)
            #self.stderrs.append(stderr)
        pb.finish()

    def plot(self, ax=None, q_threshold=None):
        fig = plt.figure(figsize=(4, 4))
        #ax = fig.add_subplot(2,1,1)
        ax = fig.add_subplot(1,1,1)
        median = num.median(self.allqs)
        if q_threshold is not None:
            filtered = filter(lambda x: x>median-q_threshold and x<median+q_threshold, self.allqs)
        else:
            filtered = self.allqs
        ax.hist(filtered, bins=100)
        ax.set_xlabel('Q')
        ax.set_ylabel('counts')
        txt ='median: %1.1f\n$\sigma$: %1.1f' % (median, num.std(self.allqs))
        return 
        #ax = fig.add_subplot(2,1,2)
        #ax.text(0.01, 0.99, txt, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
        #c_m = mpl.cm.coolwarm
        #norm = mpl.colors.Normalize(vmin=0, vmax=len(self.couples))
        #s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        #s_m.set_array([])
        #for i_couple, couple in enumerate(self.couples):
        #    #c = s_m.to_rgba(i_couple)
        #    if couple.invert_data == None:
        #        continue
        #    #dt, fx, slope, interc, log_fy_ratio, Q = couple.invert_data

        #    # Note: fx is the combined fx
        #    dt, fx, slope, interc, Q = couple.invert_data
        #    ax.plot(fx, slope*fx, alpha=0.05, color='black')
        #    #ax.plot(fx, log_fy_ratio, alpha=0.1, color='black')


def model_plot(mod, ax=None, parameter='qp', cmap='copper', xlims=None):
    cmap = mpl.cm.get_cmap(cmap)
    ax = ax_if_needed(ax)
    x, z = num.meshgrid(xlims, mod.profile('z'))
    p = num.repeat(mod.profile(parameter), len(xlims)).reshape(x.shape)
    contour = ax.contourf(x, z, p, cmap=cmap, alpha=0.5)
    #plt.colorbar(contour)


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

def qp_model_test():
    print '------------------------------qp_model_test---------------------------'
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    source_depth = 12000.
    sampling_rate = 500
    time_window = 14
    noise_level = 0.02
    n_repeat = 100
    sources = []
    targets = []

    strike = 170.
    dip = 70.
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)

    target_kwargs = {'elevation': 0.,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []
    s0 = HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=source_depth,
                          strike=strike,
                          dip=dip,
                          rake=rake,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='00',
                          attitude='master')

    for i_fn, mod_fn in enumerate(['models/earthmodel1.nd', 'models/earthmodel2.nd', 'models/earthmodel3.nd']):
        config = qseis.QSeisConfigFull.example()

        mod = cake.load_model(mod_fn)

        config.id='C0%s' % (i_fn)
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
        config.time_window = time_window
        config.nsamples = (sampling_rate*config.time_window)+1
        config.earthmodel_1d = mod
        config.source_mech = source_mech
        configs.append(config)
        p_chopper = Chopper('first(p|P)', fixed_length=0.1, phase_position=0.5,
                            phaser=PhasePie(mod=mod))
        tracers.append(Tracer(s0, targets[0], p_chopper, config=config))

    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    tracers = builder.build(tracers)

    testcouple = SyntheticCouple(master_slave=tracers)

    testcouple.process(method='pymutt',
                       noise_level=noise_level,
                       repeat=n_repeat)

    infos = '''    Qp Model Test
    Strike: %s
    Dip: %s
    Rake: %s
    Sampling rate [Hz]: %s
    dist_x: %s
    dist_y: %s
    source_depth: %s
    noise_level: %s
    ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, source_depth, noise_level)
    colors = UniqueColor(tracers=tracers)
    testcouple.plot(infos=infos, colors=colors)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)
    plt.show()
    #fx_ma, fy_ma, fx_sl, fy_sl = testcouple.get_average_spectra()

    #slope_ratio = m_slopes/s_slopes

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(fx_ma, fy_ma, 'x', label="master")
    #ax.plot(fx_sl, fy_sl, 'o', label="slave")
    #ax.set_yscale('log')
    #ax.set_xscale('log')

def vp_model_test():
    '''
    Eigentlich muesste der Chopper angepasst werden, damit phase in der Mitte
    bleibt. Aber mit normal noise sollte es egal sein, ob noise vor oder hinter
    dem Onset ist.'''
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    source_depth = 15000.
    sampling_rate = 500
    time_window = 15
    noise_level = 0.02
    n_repeat = 0
    sources = []
    targets = []

    strike = 170.
    dip = 70
    rake = -30.
    mod = cake.load_model('models/earthmodel1.nd')

    v_perturb_perc = num.arange(-2, 2.2, 0.2)
    models = []
    for i_pert, pert in enumerate(v_perturb_perc):
        new_mod = cake.LayeredModel()
        for layer in mod.layers():
            if isinstance(layer, cake.HomogeneousLayer):
                layer = copy.deepcopy(layer)
                layer.m.vp *= (100.+pert)/100.
            else:
                raise Exception('Not a HomogeneousLayer')
            new_mod.append(layer)
        models.append(new_mod)

    target_kwargs = {'elevation': 0.,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    strike = 170.
    dips = num.arange(0, 90, 10)
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    tracers = []
    s0 = HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=source_depth,
                          strike=strike,
                          dip=dip,
                          rake=rake,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='std',
                          attitude='master')

    for i_model, mod in enumerate(models):
        config = qseis.QSeisConfigFull.example()
        config.id='C%s' % i_model
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
        config.time_window = time_window
        config.nsamples = (sampling_rate*config.time_window)+1
        config.earthmodel_1d = mod
        config.source_mech = source_mech
        p_chopper = Chopper('first(p|P)', fixed_length=0.1, phase_position=0.5,
                            phaser=PhasePie(mod=config.earthmodel_1d))
        tracer0 = Tracer(s0, targets[0],
                         chopper=p_chopper,
                         config=config,
                         normalize=True)
        tracers.append(tracer0)

    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    tracers = builder.build(tracers)

    testcouple = SyntheticCouple(master_slave=tracers)
    testcouple.process(method='pymutt',
                       noise_level=noise_level,
                       repeat=n_repeat,
                       normalize=True)

    infos = ''' VP Model Test
    Strike: %s
    Dip: %s
    Rake: %s
    Sampling rate [Hz]: %s
    dist_x: %s
    dist_y: %s
    source_depth: %s
    noise_level: %s
    ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, source_depth, noise_level)
    colors = UniqueColor(tracers=tracers)
    testcouple.plot(infos=infos, colors=colors, parameter='vp', no_legend=True)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)
    plt.show()

def invert_test_1():
    print '-------------------invert_test_1-------------------------------'
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    source_depth = 12000.
    sampling_rate = 500
    time_window = 14
    noise_level = 0.02
    n_repeat = 100
    sources = []
    targets = []

    strike = 170.
    dip = 70.
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)

    target_kwargs = {'elevation': 0.,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []

    s0 = HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=source_depth,
                          strike=strike,
                          dip=dip,
                          rake=rake,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='00',
                          attitude='master')

    # eine Quelle 3000m unter der anderen
    dzs = num.arange(-2000, 2000, 1000)
    pairs = []
    for dz in dzs:
        s1 = HaskellSourceWid(lat=float(lat),
                              lon=float(lon),
                              depth=source_depth+dz,
                              strike=strike,
                              dip=dip,
                              rake=rake,
                              magnitude=.5,
                              length=0.,
                              width=0.,
                              risetime=0.02,
                              id='00',
                              attitude='master')
        pair = []
        for i_s, src in enumerate([s0, s1]):
            config = qseis.QSeisConfigFull.example()
            mod = cake.load_model('models/earthmodel1.nd')
            config.id='C0%s' % (i_s)
            config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
            config.time_window = time_window
            config.nsamples = (sampling_rate*config.time_window)+1
            config.earthmodel_1d = mod
            config.source_mech = source_mech
            configs.append(config)
            p_chopper = Chopper('first(p|P)',
                                fixed_length=0.1,
                                phase_position=0.5,
                                phaser=PhasePie(mod=mod))
            tracer = Tracer(src, targets[0], p_chopper, config=config)
            tracers.append(tracer)
            pair.append(tracer)
        pairs.append(pair)


    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    tracers = builder.build(tracers)

    testcouples = []
    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair)
        #testcouple.process(method='pymutt',
        testcouple.process(method='mtspec',
                           noise_level=noise_level,
                           repeat=n_repeat)
        testcouples.append(testcouple)
        infos = '''Qp Model Test
        Strike: %s
        Dip: %s
        Rake: %s
        Sampling rate [Hz]: %s
        dist_x: %s
        dist_y: %s
        source_depth: %s
        noise_level: %s
        ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, source_depth, noise_level)
        colors = UniqueColor(tracers=tracers)
        testcouple.plot(infos=infos, colors=colors)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)

    inverter = QInverter(couples=testcouples)
    inverter.invert(fmin=50, fmax=200)
    inverter.plot()

def invert_test_2(noise_level=0.01):
    print '-------------------invert_test_2-------------------------------'
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    fmin = 50
    fmax = 180
    sampling_rate = 500
    time_window = 25
    n_repeat = 1000
    sources = []
    targets = []
    method = 'mtspec'
    strike = 170.
    dip = 70.
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)

    target_kwargs = {'elevation': 0.,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []
    #source_depth_pairs = [(8*km, 10*km), (10*km, 12*km), (12*km, 14*km), (8*km, 14*km)]
    source_depth_pairs = [(10*km, 12*km)]
    pairs = []
    for z_pair in source_depth_pairs:
        s0 = HaskellSourceWid(lat=float(lat),
                              lon=float(lon),
                              depth=z_pair[0],
                              strike=strike,
                              dip=dip,
                              rake=rake,
                              magnitude=.5,
                              length=0.,
                              width=0.,
                              risetime=0.02,
                              id='00',
                              attitude='master')

        s1 = HaskellSourceWid(lat=float(lat),
                              lon=float(lon),
                              depth=z_pair[1],
                              strike=strike,
                              dip=dip,
                              rake=rake,
                              magnitude=.5,
                              length=0.,
                              width=0.,
                              risetime=0.02,
                              id='00',
                              attitude='master')
        pair = []
        for i_s, src in enumerate([s0, s1]):
            config = qseis.QSeisConfigFull.example()
            mod = cake.load_model('models/inv_test2.nd')
            config.id='C0%s' % (i_s)
            config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
            config.time_window = time_window
            config.nsamples = (sampling_rate*config.time_window)+1
            config.earthmodel_1d = mod
            config.source_mech = source_mech
            configs.append(config)
            p_chopper = Chopper('first(p|P)',
                                fixed_length=0.1,
                                phase_position=0.5,
                                phaser=PhasePie(mod=mod))
            tracer = Tracer(src, targets[0], p_chopper, config=config)
            tracers.append(tracer)
            pair.append(tracer)
        pairs.append(pair)


    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    tracers = builder.build(tracers)
    colors = UniqueColor(tracers=tracers)
    location_plots(tracers, colors=colors, background_model=mod)
    plt.show()

    noise = RandomNoise(noise_level)
    testcouples = []
    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair, fmin=fmin, fmax=fmax)
        #testcouple.process(method='pymutt',
        testcouple.process(method=method,
                           #noise_level=0.1,
                           repeat=n_repeat, 
                           noise=noise)
        testcouples.append(testcouple)
        infos = '''
        Strike: %s
        Dip: %s
        Rake: %s
        Sampling rate [Hz]: %s
        dist_x: %s
        dist_y: %s
        noise_level: %s
        method: %s
        ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, noise_level, method)
        testcouple.plot(infos=infos, colors=colors, noisy_Q=True)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)

    inverter = QInverter(couples=testcouples)
    inverter.invert()
    inverter.plot()
    plt.show()

def invert_test_2D(noise_level=0.001):
    print '-------------------invert_test_2D -------------------------------'
    builder = Builder(cache_dir='test-cache')
    #builder = Builder(cache_dir='muell-cache')
    #x_targets = num.array([1000., 10000., 20000., 30000., 40000., 50000.])
    z2 = 13*km
    z1 = 11*km
    d1 = 28000.
    d2 = d1*z2/z1
    print d2
    x_targets = num.array([d2, d1])
    y_targets = num.array([0.]*len(x_targets))

    lat = 50.2059
    lon = 12.5152
    sampling_rate = 400.
    #sampling_rate = 20.
    time_window = 20.
    n_repeat = 100
    sources = []
    method = 'mtspec'
    #method = 'pymutt'
    strike = 170.
    dip = 70.
    rake = -30.
    fmin = 50.
    fmax = 180.
    source_mech = qseis.QSeisSourceMechMT(mnn=1E6, mee=1E6, mdd=1E6)
    #source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)
    #source_mech.m_dc *= 1E10
    #earthmodel = 'models/inv_test2_simple.nd'
    #earthmodel = 'models/inv_test2_simple.nd'
    earthmodel = 'models/constantall.nd'
    mod = cake.load_model(earthmodel)
    component = 'r'
    target_kwargs = {
        'elevation': 0., 'codes': ('', 'KVC', '', 'Z'), 'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'Z',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []
    #source_depth_pairs = [(8*km, 10*km), (10*km, 12*km), (12*km, 14*km), (8*km, 14*km)]
    #source_depth_pairs = [(9*km, 11*km)]
    source_depth_pairs = [(11*km, 13*km)]
    pairs = []
    for z_pair in source_depth_pairs:
        s0 = DCSourceWid(lat=float(lat),
                              lon=float(lon),
                              depth=z_pair[0],
                              strike=strike,
                              dip=dip,
                              rake=rake,
                              magnitude=.5,
                              id='00')

        s1 = DCSourceWid(lat=float(lat),
                              lon=float(lon),
                              depth=z_pair[1],
                              strike=strike,
                              dip=dip,
                              rake=rake,
                              magnitude=.5,
                              id='00')
        for target in targets:
            pair = []
            for i_s, src in enumerate([s0, s1]):
                p_chopper = Chopper(
                    'first(p|P)', fixed_length=0.2, phase_position=0.5,
                                    phaser=PhasePie(mod=mod))

                config = qseis.QSeisConfigFull.example()
                config.id='C0%s' % (i_s)
                config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
                config.time_window = time_window
                config.nsamples = (sampling_rate*config.time_window)+1
                config.earthmodel_1d = mod
                config.source_mech = source_mech

                slow = p_chopper.arrival(
                src, target).p/(cake.r2d*cake.d2m/cake.km)
                config.slowness_window = [slow*0.5, slow*0.9, slow*1.1, slow*1.5]

                #config.time_reduction_velocity = \
                #    src.distance_to(target)/p_chopper.onset(target, src)/1000.
                #print dir(config)
                ##config.time_reduction_velocity *= 0.2
                #print config.time_reduction_velocity
                configs.append(config)
                tracer = Tracer(src, target, p_chopper, config=config,
                                component=component)
                tracers.append(tracer)
                pair.append(tracer)
            pairs.append(pair)


    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    tracers = builder.build(tracers, snuffle=False)
    colors = UniqueColor(tracers=tracers)

    #location_plots(tracers, colors=colors, background_model=mod)

    noise = RandomNoise(noise_level)
    testcouples = []

    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair, method=method)
        testcouple.process()
        testcouples.append(testcouple)
        infos = '''
        Strike: %s\nDip: %s\n Rake: %s\n Sampling rate [Hz]: %s\n dist_x: %s\n dist_y: %s
        noise_level: %s\nmethod: %s
        ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, noise_level, method)
        testcouple.plot(infos=infos, colors=colors, fmin=fmin, fmax=fmax)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)

    inverter = QInverter(couples=testcouples)
    inverter.invert()
    inverter.plot()
    plt.show()


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

    #sources = [HaskellSourceWid(lat=float(lat),
    #                      lon=float(lon),
    #                      depth=sd,
    #                      strike=strike,
    #                      dip=dip,
    #                      rake=rake,
    #                      magnitude=.5,
    #                      length=0.,
    #                      width=0.,
    #                      risetime=0.02,
    #                      id='0%i'%(sd/1000.),
    #                      attitude='master') for sd in source_depths]
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


def fmin_by_magnitude(magnitude, stress=0.1, vr=2750):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    duration = M02tr(Mo, stress, vr)
    return 1./duration

class Magnitude2fmin():
    def __init__(self, stress, vr, lim):
        self.stress = stress
        self.vr = vr
        self.lim = lim

    def __call__(self, magnitude):
        return max(fmin_by_magnitude(magnitude, self.stress, self.vr), self.lim)

    @classmethod
    def setup(cls, stress=0.1, vr=2750, lim=0.):
        return cls(stress, vr, lim)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('fmin')

class Magnitude2Window():
    def __init__(self, t_static, t_factor):
        self.t_static = t_static
        self.t_factor = t_factor

    def __call__(self, magnitude):
        return self.t_static+self.t_factor/fmin_by_magnitude(magnitude)

    @classmethod
    def setup(cls, t_static=0.1, t_factor=5.):
        return cls(t_static, t_factor)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('time')


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


def dbtest(noise_level=0.00005):
    print '-------------------db test-------------------------------'
    use_real_shit = True
    load_coupler = False
    #want_station = ('CZ', 'NKC', '')
    #want_station = ('CZ', 'KAC', '')
    want_station = 'all'
    lat = 50.2059
    lon = 12.5152
    sources = []
    method = 'mtspec'
    min_magnitude = 1.4
    max_magnitude = 5.4
    fminrange = 20.
    use_common = False
    fmax = 85.
    fmin = 30.
    window_by_magnitude = Magnitude2Window.setup(0.6, 5.)
    #window_by_magnitude = Magnitude2Window.setup(0.2, 5.)
    fmin_by_magnitude = Magnitude2fmin.setup(lim=fmin)
    #store_id = 'qplayground_invtest7'
    #store_id = 'qplayground_30000m_2'
    #store_id = 'qplayground_30000m_waveform_sampling5'
    #store_id = 'qplayground_30000m_simple3'
    #store_id = 'qplayground_30000m_continuous2'
    #store_id = 'qplayground_10000m_continuous2'
    #store_id = 'qplayground_10000m_continuous2_q25'
    #store_id = 'qplayground_10000m_continuous2_q800'
    #store_id = 'qplayground_10000m_continuous2_noflatearth'
    store_id = 'qplayground_total_1'
    strikemin = 160
    strikemax = 180
    dipmin = -60
    dipmax = -80
    rakemin = 20
    rakemax = 40
    engine = LocalEngine(store_superdirs=['/data/stores'])
    store = engine.get_store(store_id)
    config = engine.get_store_config(store_id)
    mod = config.earthmodel_1d
    zmin = config.source_depth_min
    zmax = config.source_depth_max
    dist_min = config.distance_min
    dist_max = config.distance_max
    plot_model(mod, parameters=['vp', 'qp'])
    fig = plt.gcf()
    fig.savefig('hist_db%s_model.png' %store_id, dpi=200)
    channel = 'SHZ'
    tt_mu = 0.
    tt_sigma = 0.01
    #tt_sigma = 0.2
    #perturbation = TTPerturbation(mu=tt_mu, sigma=tt_sigma)
    perturbation = UniformTTPerturbation(mu=tt_mu, sigma=tt_sigma)
    perturbation.plot()
    p_chopper = Chopper('first(p)', phase_position=0.4,
                        by_magnitude=window_by_magnitude,
                        phaser=PhasePie(mod=mod))
    stf_type = 'brunes'
    tracers = []
    want_phase = 'p'
    fn_coupler = 'synthetic_pairing.yaml'
    fn_noise = '/media/usb/webnet/mseed/noise.mseed'
    fn_records = '/media/usb/webnet/mseed'
    #noise = RandomNoiseConstantLevel(noise_level)
    noise = Noise(files=fn_noise, scale=noise_level)
    if stf_type=='brunes':
        # mu nachschauen!
        # beta aus Modell
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=3400.)
    else:
        brunes = False

    if noise:
        noise_pile = pile.make_pile(fn_records)
    else:
        noise_pile = None

    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    #lat = float(num.mean([e.lat for e in events]))
    #lon = float(num.mean([e.lon for e in events]))
    stations = model.load_stations('/data/meta/stations.cz.pf')
    if not want_station=='all':
        print 'Warning: only using station: %s' %'.'.join(want_station)
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
                #'elevation': 0., 'codes': ('CZ', 'KVC', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('CZ', 'LBC', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('CZ', 'KAZ', '', channel), 'store_id': store_id}
                'elevation': 0., 'codes': ('CZ', 'VAC', '', channel), 'store_id': store_id}
                #'elevation': 0., 'codes': ('CZ', 'NKC', '', channel), 'store_id': store_id}
            targets = [Target(lat=lat, lon=lon, **target_kwargs)]
            source_depths = num.arange(zmin, zmax, 200)
            distances = num.arange(1800, 2000., 200)
            #distances = num.arange(0., 2000., 200)
            sources = []
            for d in distances:
                d = num.sqrt(d**2/2.)
                for sd in source_depths:
                    mag = float(1.6+num.random.random()*2)
                    #mag = float(1.)
                    #strike, dip, rake = moment_tensor.random_strike_dip_rake()
                    strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
                                                                             dipmin, dipmax,
                                                                             rakemin, rakemax)
                    sources.append(DCSourceWid(
                        lat=float(lat),
                        lon=float(lon),
                        depth=float(sd),
                        magnitude=float(mag),
                        strike=float(strike),
                        dip=float(dip),
                        rake=float(rake),
                        north_shift=float(d),
                        east_shift=float(d),
                        stf=get_stf(type=stf_type),
                        brunes=brunes))
        elif use_real_shit is True:

            targets = [s2t(s, channel, store_id=store_id) for s in stations]
            events = filter(lambda x: x.depth>zmin and x.depth<zmax, events)
            events = filter(lambda x: x.magnitude>=min_magnitude, events)
            events = filter(lambda x: x.magnitude<=max_magnitude, events)
            sources = [e2s(e) for e in events]
            for s in sources:
                strike, dip, rake = moment_tensor.random_strike_dip_rake(strikemin, strikemax,
                                                                         dipmin, dipmax,
                                                                         rakemin, rakemax)
                s.strike = strike
                s.dip = dip
                s.rake = rake
                s.stf = get_stf(type=stf_type)
                s.brunes = brunes

        associate_responses(
            glob.glob('responses/RESP*'),
            targets,
            time=util.str_to_time('2012-01-01 00:00:00.'))
        #associate_responses(glob.glob('responses/*pz'),
        #                    targets,
        #                    time=util.str_to_time('2012-01-01 00:00:00.'),
        #                    type='polezero')

        #plot_response(response=targets[0].filter.response)
        logger.info('number of sources: %s' % len(sources))
        logger.info('number of targets: %s' % len(targets))
        #for t in targets:
        #    t.validate()
        #for s in sources:
        #    s.validate()
        coupler.process(sources, targets, mod, [want_phase, want_phase.lower()],
                        ignore_segments=True, dump_to=fn_coupler, check_relevance_by=noise_pile)
    fig, ax = Animator.get_3d_ax()
    #Animator.plot_sources(sources=targets, reference=coupler.hookup, ax=ax)
    Animator.plot_sources(sources=sources, reference=coupler.hookup, ax=ax)
    pairs_by_rays = coupler.filter_pairs(4, 1000, data=coupler.filtrate, max_mag_diff=0.5)
    animator = Animator(pairs_by_rays)
    widgets = ['plotting segments: ', progressbar.Percentage(), progressbar.Bar()]
    paired_sources = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd = p
        paired_sources.extend([s1, s2])
    used_mags = [s.magnitude for s in paired_sources]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(used_mags)
    paired_source_dict = paired_sources_dict(paired_sources)
    Animator.plot_sources(sources=paired_source_dict, reference=coupler.hookup, ax=None, alpha=1)
    #pb = progressbar.ProgressBar(maxval=len(pairs_by_rays)-1, widgets=widgets).start()
    #for i_r, r in enumerate(pairs_by_rays):
    #    e1, e2, t, td, pd, segments = r
    #    Animator.plot_ray(segments, ax=ax)
    #    pb.update(i_r)
    #print 'done'
    #pb.finish()
    #plt.show()
    pairs = []
    for r in pairs_by_rays:
        s1, s2, t  = r[0:3]
        fmin1 = fmin_by_magnitude(s1.magnitude)
        tracer1 = Tracer(s1, t, p_chopper, channel=channel, fmin=fmin1,
                         fmax=fmax, want='velocity', perturbation=perturbation.perturb(0))
        dist1, depth1 = tracer1.get_geometry()
        if dist1< dist_min or dist1>dist_max:
            continue
        fmin2 = fmin_by_magnitude(s2.magnitude)
        tracer2 = Tracer(s2, t, p_chopper, channel=channel, fmin=fmin2,
                         fmax=fmax, want='velocity', perturbation=perturbation.perturb(0))
        dist2, depth2 = tracer1.get_geometry()
        if fmax-fmin1<fminrange or fmax-fmin2<fminrange:
            continue
        if dist2< dist_min or dist2>dist_max:
            continue
        else:
            pair = [tracer1, tracer2]
            tracers.extend(pair)
            pairs.append(pair)

    builder = Builder()
    tracers = builder.build(tracers, engine=engine, snuffle=False)
    colors = UniqueColor(tracers=tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('location_model_db1.png', dpi=200)
    #plt.show()
    testcouples = []
    widgets = ['processing couples: ', progressbar.Percentage(), progressbar.Bar()]
    pb = progressbar.ProgressBar(maxval=len(pairs)-1, widgets=widgets).start()
    for i_p, pair in enumerate(pairs):
        pb.update(i_p)
        testcouple = SyntheticCouple(master_slave=pair, method=method, use_common=use_common)
        testcouple.process(noise=noise)
        testcouples.append(testcouple)
    pb.finish()
    #outfn = 'testimage'
    #plt.gcf().savefig('output/%s.png' % outfn)
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    for testcouple in num.random.choice(testcouples, 10):
        testcouple.plot(infos=infos, colors=colors, noisy_Q=False)
    inverter.plot(q_threshold=600)
    fig = plt.gcf()
    fig.savefig('hist_db%s.png' %store_id, dpi=200)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    analyze(inverter.couples)
    plt.show()


def reset_events(markers, events):
    for e in events:
        marks = filter(lambda x: x.get_event_time()==e.time, markers)
        map(lambda x: x.set_event(e), marks)

def s2t(s, channel='Z', store_id=None):
    return Target(lat=s.lat, lon=s.lon, depth=s.depth, elevation=s.elevation,
                  codes=(s.network, s.station, s.location, channel), store_id=store_id)

def e2s(e):
    #s = SourceWithMagnitude.from_pyrocko_event(e)
    #s = DCSource.from_pyrocko_event(e)
    s = DCSourceWid.from_pyrocko_event(e)
    s.magnitude = e.magnitude
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
    builder = Builder()
    method = 'mtspec'
    fmax = 89
    fminrange = 20
    use_common = True
    fmin_by_magnitude = Magnitude2fmin.setup(lim=25)
    min_magnitude = 1.5
    mod = cake.load_model('models/earthmodel_malek_alexandrakis.nd')
    #markers = PhaseMarker.load_markers('/media/usb/webnet/meta/phase_markers2008_extracted.pf')
    #events = list(model.Event.load_catalog('/data/meta/events2008.pf'))
    markers = PhaseMarker.load_markers('/data/meta/webnet_reloc/hypo_dd_markers.pf')
    events = list(model.Event.load_catalog('/data/meta/webnet_reloc/hypo_dd_event.pf'))
    events = filter(lambda x: x.magnitude>= min_magnitude, events)
    print '%s events'% len(events)
    reset_events(markers, events)
    pie = PickPie(markers=markers, mod=mod, event2source=e2s, station2target=s2t)
    stations = model.load_stations('/data/meta/stations.pf')
    want_phase = 'P'
    #window_length = {'S': 0.4, 'P': 0.4}
    window_by_magnitude = Magnitude2Window.setup(0.05, 2.5)
    phase_position = {'S': 0.2, 'P': 0.25}
    #window_length = {'S': 0.4, 'P': 0.4}
    #phase_position = {'S': 0.2, 'P': 0.2}

    channels = {'P': 'SHZ', 'S': 'SHE' }
    channel = channels[want_phase]
    pie.process_markers(phase_selection=want_phase, stations=stations, channel=channel)
    p_chopper = Chopper(
        startphasestr=want_phase, by_magnitude=window_by_magnitude,
        phase_position=phase_position[want_phase], phaser=pie)
    tracers = []

    load_coupler = False
    #fn_coupler = 'dummy_webnet_pairing_%s.yaml' % want_phase
    fn_coupler = 'webnet_pairing_%s.yaml' % want_phase

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
        coupler.process(sources, targets, mod, [want_phase, want_phase.lower()], ignore_segments=True, dump_to=fn_coupler)

    print '%s sources' %len(sources)
    fig, ax = Animator.get_3d_ax()
    pairs_by_rays = coupler.filter_pairs(4., 1000, data=coupler.filtrate, ignore=ignore, max_mag_diff=0.5)
    paired_sources = []
    for p in pairs_by_rays:
        s1, s2, t, td, pd = p
        paired_sources.extend([s1, s2])

    paired_source_dict = paired_sources_dict(paired_sources)
    Animator.plot_sources(sources=paired_source_dict, reference=coupler.hookup, ax=ax, alpha=1)
    #pb = progressbar.ProgressBar(maxval=len(pairs_by_rays)-1, widgets=widgets).start()
    #for i_r, r in enumerate(pairs_by_rays):
    #    e1, e2, t, td, pd, segments = r
    #    Animator.plot_ray(segments, ax=ax)
    #    pb.update(i_r)
    #print 'done'
    #pb.finish()
    pairs = []

    for r in pairs_by_rays:
        s1, s2, t  = r[0:3]
        fmin1 = fmin_by_magnitude(s1.magnitude)
        tracer1 = DataTracer(data_pile=data_pile, source=s1, target=t, chopper=p_chopper, channel=channel, fmin=fmin1, fmax=fmax)

        fmin2 = fmin_by_magnitude(s2.magnitude)
        tracer2 = DataTracer(data_pile=data_pile, source=s2, target=t, chopper=p_chopper, channel=channel, fmin=fmin2, fmax=fmax)
        if fmax-fmin1<fminrange or fmax-fmin2<fminrange:
            continue
        else:
            pair = [tracer1, tracer2]
            tracers.extend(pair)
            pairs.append(pair)

    #for r in pairs_by_rays:
    #    s1, s2, t, td, pd = r
    #    tracer1 = DataTracer(
    #        data_pile=data_pile, source=s1, target=t, chopper=p_chopper, channel=channel)
    #    tracer2 = DataTracer(
    #        data_pile=data_pile, source=s2, target=t, chopper=p_chopper, channel=channel)
    #    pair = [tracer1, tracer2]
    #    tracers.extend(pair)
    #    pairs.append(pair)

    #tracers = builder.build(tracers, snuffle=True)
    colors = UniqueColor(tracers=tracers)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    #fig = plt.gcf()
    #fig.savefig('location_model_db1.png', dpi=200)
    #plt.show()
    testcouples = []
    pb = progressbar.ProgressBar(maxval=len(pairs), widgets=pb_widgets('processing couples')).start()
    for i_p, pair in enumerate(pairs):
        pb.update(i_p)
        testcouple = SyntheticCouple(master_slave=pair, method=method)
        testcouple.process()
        if testcouple.good:
            testcouples.append(testcouple)
    pb.finish()
    #testcouples = filter(lambda x: x.delta_onset()>0.06, testcouples)
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    for tc in num.random.choice(testcouples, 10):
        tc.plot(infos=infos, colors=colors)
    inverter.plot(q_threshold=500)
    fig = plt.gcf()
    fig.savefig('hist_databasetest.png', dpi=200)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    couples = inverter.couples
    analyze(couples)


def analyze(couples):
    # by meanmag
    Qs = []
    mags = []
    magdiffs = []
    by_target = {}
    for c in couples:
        Q = c.invert_data[-1]
        Qs.append(Q)
        meanmag = (c.master_slave[0].source.magnitude+c.master_slave[1].source.magnitude)/2.
        mags.append(meanmag)

        magdiff = abs(c.master_slave[0].source.magnitude-c.master_slave[1].source.magnitude)
        magdiffs.append(magdiff)
        try:
            by_target['%s-%s'%(c.master_slave[0].target.codes[1], c.master_slave[1].target.codes[1]) ].append(Q)
        except KeyError:
            by_target['%s-%s'%(c.master_slave[0].target.codes[1], c.master_slave[1].target.codes[1]) ] = [Q]
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(mags, Qs, 'bo')
    #ax.set_ylim(0, 2500)
    ax.set_title('mean magnitude vs Q')
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(magdiffs, Qs, 'bo')
    #ax.set_ylim(0, 2500)
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


if __name__=='__main__':
    #invert_test_1()
    #qpqs()

    # DIESER:
    #invert_test_2()
    invert_test_2D(noise_level=0.0000001)
    #invert_test_2D_parallel(noise_level=0.1)
    dbtest()
    # TODO: !!! !!!!!!!!!!!!!!! Synthetics in displacemtn!!!!!!!!!!!!!!!1
    #apply_webnet()
    plt.show()
    #noise_test()
    #qp_model_test()
    #constant_qp_test()
    #sdr_test()
    #vp_model_test()
