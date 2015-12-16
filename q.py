import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rc('ytick', labelsize=8) 
mpl.rc('xtick', labelsize=8) 

import multiprocessing
import copy
from pyrocko.gf import meta, DCSource, RectangularSource, Target
from pyrocko.guts import *
from pyrocko import gui_util
from pyrocko import orthodrome
from pyrocko import trace
from pyrocko import cake
from pyrocko import crust2x2
from pyrocko.fomosto import qseis
from pyrocko.parimap import parimap

from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as num
import sys
import os
from micro_engine import OnDemandEngine, Tracer, create_store, Builder, QResponse, add_noise, RandomNoise
from autogain.autogain import PhasePie
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pjoin = os.path.join
km = 1000.

methods_avail = ['fft']
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
    if not method in methods_avail:
        logger.exception("Method %s not available" % method)
        raise Exception("Method %s not available" % method)
    else:
        logger.debug('method used %s' %method)
        return True

def getattr_dot(obj, attr):
    v = reduce(getattr, attr.split('.'), obj)
    return v

def xy2targets(x, y, o_lat, o_lon, channels, **kwargs):
    targets = []
    for istat, xy in enumerate(zip(x,y)):
        for c in channels:
            lat, lon = orthodrome.ne_to_latlon(o_lat, o_lon, *xy)
            kwargs.update({'lat': float(lat), 'lon': float(lon),
                           'codes':('', '%i'%istat, '', c)})
            targets.append(Target(**kwargs))
    return targets

def regression(tr, method, fmin=0., fmax=9999, chopper=None):
    fx, fy = spectralize(tr, method=method, chopper=chopper)
    indx = num.where(num.logical_and(fx>=fmin, fx<=fmax))
    #slope, interc, r_value, p_value, stderr = linregress(fx, num.log(num.abs(fy)))
    return linregress(fx, num.log(num.abs(fy)))

def legend_clear_duplicates(ax):
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

def ax_if_needed(ax):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    return ax

def plot_traces(tr, ax=None, label='', color='r'):
    ax = ax_if_needed(ax)
    ax.plot(tr.get_xdata(), tr.get_ydata(), label=label, color=color)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('displ [m]')

def plot_model(mod, ax=None, label='', color=None, parameter='qp'):
    ax = ax_if_needed(ax)
    z = mod.profile('z')
    profile = mod.profile(parameter)
    ax.plot(profile, -z, label=label, c=color)
    ax.set_ylabel('depth [m]')
    ax.set_title(parameter)

def infos(ax, info_string):
    ax.axis('off')
    ax.text(0., 0, info_string, transform=ax.transAxes)

def plot_locations(items):
    f = plt.figure(figsize=(3,3))
    ax = f.add_subplot(111)
    lats = []
    lons = []
    for item in items:
        ax.plot(item.lon, item.lat, 'og')
        lats.append(item.lat)
        lons.append(item.lon)

    ax.plot(num.mean(lons), num.mean(lats), 'xg')
    ax.set_title('locations')
    ax.set_xlabel('lon', size=7)
    ax.set_ylabel('lat', size=7)

    y_range = num.max(lats)-num.min(lats)
    x_range = num.max(lons)-num.min(lons)
    ax.set_ylim([num.min(lats)-0.05*y_range, num.max(lats)+0.05*y_range])
    ax.set_xlim([num.min(lons)-0.05*x_range, num.max(lons)+0.05*x_range])
    f.savefig("locations_plot.png", dpi=160., bbox_inches='tight', pad_inches=0.01)

#def compile_slopes(slopes):
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for slope in slopes:
#        for s, ta, slope_data in slope.iterdd():
#            x, y, m, interc = slope_data
#            if s.attitude=='master':
#                ax.plot(x,y,'o',alpha=0.2)
#            else:
#                ax.plot(x,y,alpha=0.2)
#
#    ax.set_yscale("log")

def intersect_all(*args):
    for i, l in enumerate(args):
        if i==0:
            inters = num.intersect1d(l, args[1])
        else:
            try:
                inters = num.intersect1d(inters, args[i+1])
            except IndexError:
                return inters

def average_spectra(fxs, fys):
    x_intersect = intersect_all(*fxs)
    intersecting = num.zeros((len(x_intersect), len(fys)))
    for i, fy in enumerate(fys):
        indx = num.where(fxs[i]==x_intersect)
        intersecting[:, i] = fy[indx]

    return x_intersect, num.mean(intersecting, 1)

def extract(fxfy, upper_lim=0., lower_lim=99999.):
    indx = num.where(num.logical_and(fxfy[0]>=upper_lim, fxfy[0]<=lower_lim))
    indx = num.array(indx)
    return fxfy[:, indx].reshape(2, len(indx.T))

class DCSourceWid(DCSource):
    store_id = String.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)

class HaskellSourceWid(RectangularSource):
    #store_id = String.T(optional=True, default=None)
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

# Eine station aus dem webnet
# KVC

class DDContainer():
    """ A Double Dict Container..."""
    def __init__(self, key1=None, key2=None, value=None):
        self.content = defaultdict(dict)
        if key1 and key2 and value:
            self.content[key1][key2] = value

    def iterdd(self):
        for key1, key2_value in self.content.iteritems():
            for key2, value in key2_value.iteritems():
                yield key1, key2, value

    def __getitem__(self, keys):
        return self.content[keys[0]][keys[1]]

    def add(self, key1, key2, value):
        self.content[key1][key2] = value

    def as_lists(self):
        keys1 = []
        keys2 = []
        values = []
        for k1,k2,v in self.iterdd():
            keys1.append(k1)
            keys2.append(k2)
            values.append(v)

        return keys1, keys2, values

def linear_fit(x, y, m):
    return y+m*x

def exp_fit(x, y, m):
    return num.exp(y - m*x)

def spectralize(tr, method, chopper=None):
    if method=='fft':
        # fuehrt zum amplitudenspektrum
        f, a = tr.spectrum(tfade=chopper.get_tfade(tr.tmax-tr.tmin))

    elif method=='pymutt':
        # berechnet die power spectral density
        r = pymutt.mtft(tr.ydata, dt=tr.deltat)
        f = num.arange(r['nspec'])*r['df']
        a = r['power']

    elif method=='mtspec':
        a, f = mtspec(data=tr.ydata,
                      delta=tr.deltat,
                      number_of_tapers=5,
                      time_bandwidth=2.,
                      nfft=2**9,
                      statistics=False)

    return f, num.sqrt(a)

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

    def get_slopes(self):
        slopes = []
        for tracer, fxfy in self.spectra:
            fx, fy = num.vsplit(fxfy, 2)
            #fc = 1./(0.5*s.risetime)
            #fc = 50.
            slope_section = extract(fxfy, upper_lim=fc, lower_lim=230.)
            popt, pcov = curve_fit(self.fit_function, *slope_section)
            fy_fit = self.fit_function(slope_section[0], popt[0], popt[1]),
            slopes.append((tracer, (slope_section[0], num.array(fy_fit).T, popt[0], popt[1])))

        return slopes

    def plot_all(self, ax=None, colors=None, alpha=1., legend=True):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        count = 0
        for tracer, fxfy in self.spectra:
            fx, fy = num.vsplit(fxfy, 2)
            if colors:
                color = colors[tracer]
            ax.plot(fx.T, fy.T, label=tracer.config.id, color=color, alpha=alpha)
            count += 1
        ax.autoscale()
        ax.set_title("$\sqrt{PSD}$")
        ax.set_ylabel("A")
        ax.set_xlabel("f[Hz]")
        ax.set_yscale("log")

    def plot_all_with_linregress(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fmin = 1.
        fmax = 20.
        regs = self.linregress(fmin=fmin, fmax=fmax)
        count = 0
        slopes = []
        for s, ta, fxfy in self.iterdd():
            fx, fy = fxfy
            color = colors[count%len(colors)]
            ax.plot(fx, num.abs(fy), label=ta.store_id, color=color)
            slope, interc = regs[s][ta]
            slopes.append(slope)
            linreg = interc + fx*slope
            indx = num.where(num.logical_and(fx>=fmin, fx<=fmax))
            ax.plot(fx[indx],num.exp(linreg[indx]), label=ta.store_id,
                    color=color)
            count += 1

        slope_string = '\n'.join(map(str, slopes))
        ax.text(0.1,0.9,slope_string, verticalalignment='top',
                                horizontalalignment='left', 
                                transform=ax.transAxes)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()

    def linregress(self, fmin=-999, fmax=999):
        regressions = DDContainer()
        for s, ta, fxfy in self.iterdd():
            fx, fy = fxfy
            indx = num.where(num.logical_and(fx>=fmin, fx<=fmax))
            slope, interc, r_value, p_value, stderr = linregress(fx[indx],
                    num.log(num.abs(fy[indx])))
            regressions.add(s, ta, (slope, interc))

        return regressions

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
    def __init__(self, master_slave):
        self.master_slave = master_slave
        self.spectra = Spectra()
        self.noisy_spectra = Spectra()
        self.fit_function = None
        self.use_component = 'z'
        self.colors = None

        self.repeat = 1
        self.noise_level = 0

    def process(self, method='mtspec', **pp_kwargs):
        self.noise_level = pp_kwargs.pop('noise_level', self.noise_level)
        self.repeat = pp_kwargs.pop('repeat', self.repeat)
        check_method(method)
        for tracer in self.master_slave:
            tr = tracer.process(self.use_component, **pp_kwargs)
            f, a = self.get_spectrum(tr, method, tracer.chopper)
            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

            for i in xrange(self.repeat):
                tr = tracer.process(self.use_component, **pp_kwargs).copy()
                tr_noise = add_noise(tr, level=self.noise_level)
                f, a = self.get_spectrum(tr_noise, method, tracer.chopper)
                fxfy = num.vstack((f,a))
                self.noisy_spectra.spectra.append((tracer, fxfy))

    def get_spectrum(self, tr, method, chopper):
        return spectralize(tr, method, chopper)

    def plot(self, colors, **kwargs):
        fig = plt.figure(figsize=(5, 6))
        ax = fig.add_subplot(3, 2, 1)
        self.spectra.plot_all(ax, colors=colors, legend=False)
        self.noisy_spectra.plot_all(ax, colors=colors, alpha=0.05, legend=False)
        #legend_clear_duplicates(ax)
        ax = fig.add_subplot(3, 2, 2)
        for tracer in self.master_slave:
            plot_model(mod=tracer.config.earthmodel_1d,
                       label=tracer.config.id,
                       color=colors[tracer],
                       ax=ax,
                       parameter=kwargs.get('parameter', 'qp'))

        ax.set_xlim((ax.get_xlim()[0]*0.9,
                     ax.get_xlim()[1]*1.1))
        for tr in self.master_slave:
            ax.axhline(-tr.source.depth, ls='--', label='z %s' % tr.config.id,
                       color=colors[tr])
        if not kwargs.get('no_legend', False):
            ax.legend()
        ax = fig.add_subplot(3, 2, 3)
        for tracer in self.master_slave:
            plot_traces(tr=tracer.process(
                self.use_component, normalize=kwargs.get('normalize', False)),
                        ax=ax, label=tr.config.id,
                        color=colors[tracer])

        if self.noise_level!=0.:
            trs = tracer.process(self.use_component, normalize=kwargs.get('normalize', False)).copy()
            trs = add_noise(trs, level=self.noise_level)
            plot_traces(tr=trs, ax=ax, label=tracer.config.id, color=colors[tracer])

        ax = fig.add_subplot(3, 2, 4)
        if 'noisy_Q' in kwargs:
            fxs, fy_ratios = noisy_spectral_ratios(self, 50, 240)

            Qs = []
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

            std_q = num.std(Qs)
            maxQ = max(Qs)
            minQ = min(Qs)

            c_m = mpl.cm.jet
            norm = mpl.colors.Normalize(vmin=minQ, vmax=maxQ)
            s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
            s_m.set_array([])
            for i in xrange(len(xs)):
                c = s_m.to_rgba(Qs[i])
                ax.plot(xs[i], ys_ratio[i], color=c, alpha=0.2)
                ax.plot(xs[i], ys[i], color=c, alpha=0.2)
            ax.set_xlabel('f[Hz]')
            ax.set_ylabel('log(A1/A2)')
            ax.text(0.02,0.02, 'std deviation(Q)=%1.2f' % std_q, transform=ax.transAxes)
            cb = plt.colorbar(s_m)
            cb.set_label('Q')
            ax = fig.add_subplot(3, 2, 5)
            ax.hist(Qs, bins=20)
            ax.set_xlim([-10000., 10000.])
            ax.set_ylabel('Count')
            ax.set_xlabel('Q')
        ax = fig.add_subplot(3, 2, 6)
        infos(ax, kwargs.pop('infos'))
        plt.show()

    def delta_onset(self):
        ''' Get the onset difference between two (!) used phases'''
        diff = 0
        assert len(self.master_slave)==2
        for tr in self.master_slave:
            if not diff:
                diff = tr.onset()
            else:
                diff -= tr.onset()
        return diff

    def set_fit_function(self, func):
        self.spectra.set_fit_function(func)
        self.fit_function = func

    def get_slopes(self):
        return self.spectra.get_slopes()

    def get_spectra(self):
        return self.spectra

    def q_scherbaum(self, phasestr):

        sources, targets, fxfy = self.spectra.as_lists()
        dists = num.zeros(len(sources))
        arrivals = num.zeros(len(sources))
        freqs = []
        A = []
        for i in range(len(sources)):
            dists[i] = sources[i].distance_to(targets[i])
            arrivals[i] = self.engine.get_store(sources[i].store_id).t(phasestr,
                                                (sources[i].depth, dists[i]))
            freqs.append(fxfy[i][0])
            A.append(num.abs(fxfy[i][1]))
        assert all(fxfy[0][0]==fxfy[1][0])
        regresses = self.spectra.linregress(1,20)
        slope = regresses[1]/regresses[0]
        plt.plot(fxfy[0], _q(fxfy[1], arrivals[::-1], slope=slope))

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


class Chopper():
    def __init__(self, startphasestr, endphasestr=None, fixed_length=None,
                 phase_position=0.5, xfade=0.0, phaser=None):
        self.phase_pie = phaser or PhasePie()
        self.startphasestr = startphasestr
        self.endphasestr = endphasestr
        self.phase_position = phase_position
        self.fixed_length = fixed_length
        self.chopped = DDContainer()
        self.xfade = xfade

    def chop(self, s, t, tr):
        dist = s.distance_to(t)
        depth = s.depth
        tstart = self.phase_pie.t(self.startphasestr, (depth, dist))
        if self.endphasestr!=None and self.fixed_length==None:
            tend = self.phase_pie.t(self.endphasestr, (depth, dist))
        elif self.endphasestr==None and self.fixed_length!=None:
            tend = self.fixed_length+tstart
        else:
            raise Exception('Need to define exactly one of endphasestr and fixed length')
        tr_bkp = tr
        logger.info('lowpass filter with 0.5/nyquist')
        tr = tr.copy()
        tr.set_location('cp')
        trange = tend-tstart
        tstart -= trange*self.phase_position
        tend -= trange*self.phase_position
        tr.chop(tstart-self.get_tfade(trange), tend+self.get_tfade(trange))
        taperer = trace.CosFader(xfrac=self.get_xfade())
        tr.taper(taperer)
        self.chopped.add(s,t,tr)
        return tr

    def get_xfade(self):
        return self.xfade

    def get_tfade(self, t_span):
        return self.xfade*t_span

    def iter_results(self):
        return self.chopped.iterdd()

    def set_trange(self):
        _,_, times = self.table.as_lists()

        mintrange = 99999.
        for tstart, tend in times:
            trange = tstart-tend
            mintrange = trange if trange < mintrange else 99999.
        return mintrange

    def set_tfade(self, tfade):
        '''only needed for method fft'''
        self.tfade = tfade

    def onset(self, s, t):
        return self.phase_pie.t(self.startphasestr, (s.depth, s.distance_to(t)))

def noisy_spectral_ratios(pairs, fmin, fmax):
    '''Ratio of two overlapping spectra'''
    ratios = []
    indxs = []
    spectra_couples = defaultdict(list)
    for tr, fxfy in pairs.noisy_spectra.spectra:
        spectra_couples[tr].append(fxfy)
    one, two = spectra_couples.values()
    for i in xrange(len(one)):
        fmin = max(fmin, min(min(one[i][0]), min(two[i][0])))
        fmax = min(fmax, max(max(one[i][0]), min(two[i][0])))
        ind0 = num.where(num.logical_and(one[i][0]>=fmin, one[i][0]<=fmax))
        ind1 = num.where(num.logical_and(two[i][0]>=fmin, two[i][0]<=fmax))
        fy_ratio = one[i][1][ind0]/two[i][1][ind1]
        ratios.append(fy_ratio)
        indxs.append(one[i][0][ind0])
    return indxs, ratios


def spectral_ratio(pair, fmin, fmax):
    '''Ratio of two overlapping spectra'''
    assert len(pair.spectra.spectra)==2
    fx = []
    fy = []
    for tr, fxfy in pair.spectra.spectra:
        fmin = max(fmin, min(fxfy[0]))
        fmax = min(fmax, max(fxfy[0]))
        fx.append(fxfy[0])
        fy.append(fxfy[1])

    ind0 = num.where(num.logical_and(fx[0]>=fmin, fx[0]<=fmax))
    ind1 = num.where(num.logical_and(fx[1]>=fmin, fx[1]<=fmax))
    fy_ratio = fy[0][ind0]/fy[1][ind1]
    return fx[0][ind0], fy_ratio

class QInverter:
    def __init__(self, couples):
        self.couples = couples

    def invert(self, fmin=-9999, fmax=9999):
        self.ratios = []
        for couple in self.couples:
            fx, fy_ratio = spectral_ratio(couple, fmin, fmax)
            slope, interc, r_value, p_value, stderr = linregress(fx, num.log(fy_ratio))
            dt = couple.delta_onset()
            Q = -1.*num.pi*dt/slope
            #self.ratios.append((dt, fx, slope, interc, num.log(fy_ratio), Q))
            couple.invert_data = (dt, fx, slope, interc, num.log(fy_ratio), Q)

    def plot(self, ax=None):
        ax = ax_if_needed(ax)
        for couple in self.couples:
            dt, fx, slope, interc, log_fy_ratio, Q = couple.invert_data
            ax.plot(fx, interc+slope*fx)


def noise_test():
    print '-----------------------------------------noise_test------------------------'
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])
    levels = num.arange(0, 0.5, 0.1)
    lat = 50.2059
    lon = 12.5152
    source_depth = 12000.
    sampling_rate = 500
    time_window = 14
    noise_level = 0.02
    n_repeat = 50
    sources = []
    targets = []
    strike = 170.
    dip = 70.
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)
    method = 'pymutt'
    target_kwargs = {'elevation': 0,
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

    for i_fn, mod_fn in enumerate(['earthmodel1.nd', 'earthmodel2.nd', 'earthmodel3.nd']):
        config = qseis.QSeisConfigFull.example()

        mod = cake.load_model(pjoin('models', mod_fn))
        config.id='C0%s' % (i_fn)
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
        config.time_window = time_window
        config.nsamples = (sampling_rate*config.time_window)+1
        config.earthmodel_1d = mod
        config.source_mech = source_mech
        configs.append(config)
        p_chopper = Chopper('first(p|P)', fixed_length=0.1, phase_position=0.5,
                            phaser=PhasePie(mod=mod), xfade=0.)
        tracers.append(Tracer(s0, targets[0], p_chopper, config=config))

    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1
    extra = {'qseis': qs}

    tracers = builder.build(tracers)

    testcouple = SyntheticCouple(master_slave=tracers)
    colors = UniqueColor(tracers=tracers)

    for i_level, noise_level in enumerate(levels):
        testcouple.process(
            method=method, noise_level=noise_level, repeat=n_repeat)

        infos = '''    Noise Test
        Strike: %s
        Dip: %s
        Rake: %s
        Sampling rate [Hz]: %s
        dist_x: %s
        dist_y: %s
        source_depth: %s
        noise_level: %s
        method: %s
        ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, source_depth, noise_level, method)
        testcouple.plot(infos=infos, colors=colors, fontsize=8, no_legend=True)
        #fig = plt.gcf()
        #ax = fig.add_subplot(3, 2, 5)
        #slope_histogram(ax, testcouple.noisy_spectra, colors)
        outfn = 'test_nl%s.png'% noise_level
        fig.savefig('output/%s.png' % outfn, )

    plt.show()


def constant_qp_test():
    print '-----------------------------------------constant_qp_test------------------------'

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

    target_kwargs = {'elevation': 0,
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
    models = ['constantq1.nd', 'constantq2.nd', 'constantq3.nd', 'constantq4.nd']
    for i_fn, mod_fn in enumerate(models):
        config = qseis.QSeisConfigFull.example()

        mod = cake.load_model(pjoin('models', mod_fn))

        config.id='C0%s' % (i_fn)
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
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
    extra = {'qseis': qs}

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
    fig = plt.gcf()
    ax = fig.add_subplot(3, 2, 5)
    slope_histogram(ax, testcouple.noisy_spectra, colors)
    outfn = 'constantq'
    plt.gcf().savefig('output/%s.png' % outfn)
    plt.show()

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

    target_kwargs = {'elevation': 0,
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
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
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
    extra = {'qseis': qs}

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

def sdr_test():
    print '-------------------sdr_test-------------------------------'

    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    source_depth = 15000.
    sampling_rate = 500
    time_window = 15
    noise_level = 0.02
    n_repeat = 100
    sources = []
    targets = []

    strike = 170.
    dips = num.arange(0, 90, 10)
    rake = -30.
    source_mechs = [qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake) 
                    for dip in dips]

    target_kwargs = {'elevation': 0,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    tracers = []
    mod1 = cake.load_model('models/earthmodel1.nd')
    p_chopper = Chopper('first(p|P)', fixed_length=0.1, phase_position=0.5,
                        phaser=PhasePie(mod=mod1))
    for i_source_mech, source_mech in enumerate(source_mechs):
        config1 = qseis.QSeisConfigFull.example()

        config1.id='C%s' % i_source_mech
        config1.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
        config1.time_window = time_window
        config1.nsamples = (sampling_rate*config1.time_window)+1
        config1.earthmodel_1d = mod1
        config1.source_mech = source_mech

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
                              id='%s' % i_source_mech,
                              attitude='master')
        tracer0 = Tracer(source=s0,
                         target=targets[0],
                         config=config1,
                         chopper=p_chopper,
                         normalize=True)
        tracers.append(tracer0)

    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1
    extra = {'qseis': qs}

    tracers = builder.build(tracers)

    testcouple = SyntheticCouple(master_slave=tracers)

    testcouple.process( method='pymutt',
                       noise_level=noise_level,
                       repeat=n_repeat,
                       normalize=True)

    infos = '''    SDR test
    Strike: %s
    Dip: %s
    Rake: %s
    Sampling rate [Hz]: %s
    dist_x: %s
    dist_y: %s
    source_depth: %s
    noise_level: %s
    ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, source_depth, noise_level)
    colors = TracerColor(attr='config.source_mech.dip', tracers=tracers)
    testcouple.plot(infos=infos, colors=colors)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)
    plt.show()

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

    target_kwargs = {'elevation': 0,
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
        config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
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
    extra = {'qseis': qs}

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

    target_kwargs = {'elevation': 0,
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
            config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
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
    extra = {'qseis': qs}

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

def invert_test_2():
    print '-------------------invert_test_2-------------------------------'
    builder = Builder(cache_dir='test-cache')
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    sampling_rate = 500
    time_window = 25
    noise_level = 0.001
    n_repeat = 100
    sources = []
    targets = []
    method = 'mtspec'
    strike = 170.
    dip = 70.
    rake = -30.
    source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)

    target_kwargs = {'elevation': 0,
                     'codes': ('', 'KVC', '', 'Z'),
                     'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'NEZ',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []
    source_depth_pairs = [(8*km, 10*km), (10*km, 12*km), (12*km, 14*km), (8*km, 14*km)]
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
            mod = cake.load_model('models/inv_test.nd')
            config.id='C0%s' % (i_s)
            config.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
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
    extra = {'qseis': qs}

    tracers = builder.build(tracers)

    noise = RandomNoise(noise_level)
    testcouples = []
    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair)
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
        colors = UniqueColor(tracers=tracers)
        testcouple.plot(infos=infos, colors=colors, noisy_Q=True)
    outfn = 'testimage'
    plt.gcf().savefig('output/%s.png' % outfn)

    inverter = QInverter(couples=testcouples)
    inverter.invert(fmin=50, fmax=200)
    inverter.plot()

if __name__=='__main__':
    #invert_test_1()
    invert_test_2()
    noise_test()
    qp_model_test()
    constant_qp_test()
    sdr_test()
    vp_model_test()
    sys.exit(0)
