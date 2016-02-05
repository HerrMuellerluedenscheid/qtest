import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rc('ytick', labelsize=8)
mpl.rc('xtick', labelsize=8)

import copy
import progressbar
import multiprocessing
from pyrocko.gf import meta, DCSource, RectangularSource, Target, LocalEngine
from pyrocko.gf import ExplosionSource
from pyrocko.guts import String
from pyrocko import orthodrome
from pyrocko import cake
from pyrocko.fomosto import qseis
from pyrocko.trace import nextpow2
from collections import defaultdict
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as num
import os
from micro_engine import Tracer, Builder
from micro_engine import add_noise, RandomNoise, Chopper, DDContainer
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
    if method not in methods_avail:
        logger.exception("Method %s not available" % method)
        raise Exception("Method %s not available" % method)
    else:
        logger.debug('method used %s' % method)
        return True


def getattr_dot(obj, attr):
    v = reduce(getattr, attr.split('.'), obj)
    return v


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
        fig = plt.figure()
        ax = fig.add_subplot(111)
    return ax


def plot_traces(tr, ax=None, label='', color='r'):
    ax = ax_if_needed(ax)
    ax.plot(tr.get_xdata(), tr.get_ydata(), label=label, color=color)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('displ [m]')


def plot_model(mod, ax=None, label='', color=None, parameters=['qp']):
    ax = ax_if_needed(ax)
    z = mod.profile('z')
    colors = 'rgbcy'
    for ip, parameter in enumerate(parameters):
        profile = mod.profile(parameter)
        if ip>=1:
            ax = ax.twiny()
        ax.plot(profile, -z, label=label, c=colors[ip])
        ax.set_xlabel(parameter, color=colors[ip])

    ax.set_ylabel('depth [m]')


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
        a = num.sqrt(a)

    elif method=='mtspec':
        a, f = mtspec(data=tr.ydata,
                      delta=tr.deltat,
                      number_of_tapers=5,
                      time_bandwidth=2.,
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

    def plot_all(self, ax=None, colors=None, alpha=1., legend=True, fmin=None, fmax=None):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        count = 0
        if fmin and fmax:
            ax.axvspan(fmin, fmax, facecolor='0.5', alpha=0.25)
        elif fmin:
            ax.axvline(fmin)
        elif fmax:
            ax.axvline(fmax)
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
    def __init__(self, master_slave, fmin=-9999, fmax=9999, method='mtspec'):
        check_method(method)
        self.method = method
        self.master_slave = master_slave
        self.fmin = fmin
        self.fmax = fmax
        self.spectra = Spectra()
        self.noisy_spectra = Spectra()
        self.fit_function = None
        self.colors = None

        self.repeat = 1
        self.noise_level = 0

    def process(self, **pp_kwargs):
        for tracer in self.master_slave:
            self.noise_level = pp_kwargs.pop('noise_level', self.noise_level)
            self.repeat = pp_kwargs.pop('repeat', self.repeat)
            tr = tracer.process(**pp_kwargs)
            f, a = self.get_spectrum(tr, self.method, tracer.chopper)
            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

            for i in xrange(self.repeat):
                #tr = tracer.process(**pp_kwargs).copy()
                tr = tracer.process(**pp_kwargs)
                tr_noise = add_noise(tr, level=self.noise_level)
                f, a = self.get_spectrum(tr_noise, self.method, tracer.chopper)
                fxfy = num.vstack((f,a))
                self.noisy_spectra.spectra.append((tracer, fxfy))

    def get_spectrum(self, tr, method, chopper):
        return spectralize(tr, method, chopper)

    def plot(self, colors, **kwargs):
        fig = plt.figure(figsize=(3, 6))
        ax = fig.add_subplot(4, 1, 2)
        self.spectra.plot_all(ax, colors=colors, legend=False, fmin=self.fmin,
                              fmax=self.fmax)
        self.noisy_spectra.plot_all(ax, colors=colors, alpha=0.05, legend=False)
        #legend_clear_duplicates(ax)
        
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
        #if not kwargs.get('no_legend', False):
        #    ax.legend()
        ax = fig.add_subplot(4, 1, 1)
        for tracer in self.master_slave:
            plot_traces(tr=tracer.process(normalize=kwargs.get('normalize', False)),
                        ax=ax, label=tracer.config.id,
                        color=colors[tracer])

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
            ax.hist(num.array(Qs), bins=25)
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


def spectral_ratio(couple):
    '''Ratio of two overlapping spectra'''
    assert len(couple.spectra.spectra)==2
    fx = []
    fy = []
    for tr, fxfy in couple.spectra.spectra:
        fs, a = fxfy
        fmin = max(couple.fmin, fs.min())
        fmax = min(couple.fmax, fs.max())
        fx.append(fxfy[0])
        fy.append(fxfy[1])

    ind0 = limited_frequencies_ind(fmin, fmax, fx[0])
    ind1 = limited_frequencies_ind(fmin, fmax, fx[1])
    assert all(fx[0][ind0]==fx[1][ind0])
    fy_ratio = fy[0][ind0]/fy[1][ind1]
    return fx[0][ind0], fy_ratio


class QInverter:
    def __init__(self, couples):
        self.couples = couples

    def invert(self):
        self.allqs = []
        self.ratios = []
        self.p_values = []
        self.stderrs = []
        widgets = ['regression analysis', progressbar.Percentage(), progressbar.Bar()]
        pb = progressbar.ProgressBar(maxval=len(self.couples)-1, widgets=widgets).start()
        for i_c, couple in enumerate(self.couples):
            pb.update(i_c)
            fx, fy_ratio = spectral_ratio(couple)
            slope, interc, r_value, p_value, stderr = linregress(fx, num.log(fy_ratio))
            dt = couple.delta_onset()
            Q = -1*num.pi*dt/slope
            if num.isnan(Q):
                logger.warn('Q is nan')
                continue
            couple.invert_data = (dt, fx, slope, interc, num.log(fy_ratio), Q)
            self.allqs.append(Q)
            self.p_values.append(p_value)
            self.stderrs.append(stderr)
        pb.finish()

    def plot(self, ax=None, q_threshold=999999.):
        fig, axs = plt.subplots(2,1)
        median = num.median(self.allqs)
        filtered = filter(lambda x: x>median-q_threshold and x<median+q_threshold, self.allqs)
        axs[0].hist(filtered, bins=100)
        txt ='median: %1.1f\n$\sigma$: %1.1f' % (median, num.std(self.allqs))
        axs[0].text(0.01, 0.99, txt, transform=axs[0].transAxes, horizontalalignment='left', verticalalignment='top')
        axs[1].plot(norm.pdf(filtered), '-r', label='pdf')
        #for couple in self.couples:
        #   dt, fx, slope, interc, log_fy_ratio, Q = couple.invert_data
        #   ax.plot(fx, interc+slope*fx)


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
        ax.plot(x, z[0].ravel(), color=colors[tr])
        minx[itr] = num.min(x)
        maxx[itr] = num.max(x)
        miny[itr] = num.min(z)
        maxy[itr] = num.max(z)
    pb.finish()
    minx = min(minx.min(), -100)-100
    maxx = max(maxx.max(), 100)+100
    miny = min(miny.min(), -100)-100
    maxy = max(maxy.max(), 100)+100
    xlims=(minx, maxx)
    ylims=(miny, maxy)
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
    d1 = 20000.
    d2 = 20000.*z2/z1
    x_targets = num.array([d2, d1])
    y_targets = num.array([0.]*len(x_targets))

    lat = 50.2059
    lon = 12.5152
    sampling_rate = 500.
    #sampling_rate = 20.
    time_window = 20.
    n_repeat = 100
    sources = []
    method = 'mtspec'
    #method = 'pymutt'
    strike = 170.
    dip = 70.
    rake = -30.
    fmin = 30.
    fmax = 130.
    source_mech = qseis.QSeisSourceMechMT(mnn=1E6, mee=1E6, mdd=1E6)
    #source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)
    #source_mech.m_dc *= 1E10
    earthmodel = 'models/inv_test2_simple.nd'
    #earthmodel = 'models/constantall.nd'
    mod = cake.load_model(earthmodel)
    component = 'r'
    target_kwargs = {
        'elevation': 0, 'codes': ('', 'KVC', '', 'Z'), 'store_id': None}

    targets = xy2targets(x_targets, y_targets, lat, lon, 'Z',  **target_kwargs)
    logger.info('Using %s targets' % (len(targets)))

    configs = []
    tracers = []
    #source_depth_pairs = [(8*km, 10*km), (10*km, 12*km), (12*km, 14*km), (8*km, 14*km)]
    #source_depth_pairs = [(9*km, 11*km)]
    source_depth_pairs = [(11*km, 13*km)]
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

    location_plots(tracers, colors=colors, background_model=mod)

    noise = RandomNoise(noise_level)
    testcouples = []

    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair, fmin=fmin, fmax=fmax)
        testcouple.process(method=method, repeat=n_repeat, noise=noise)
        testcouples.append(testcouple)
        infos = '''
        Strike: %s\nDip: %s\n Rake: %s\n Sampling rate [Hz]: %s\n dist_x: %s\n dist_y: %s
        noise_level: %s\nmethod: %s
        ''' % (strike, dip, rake, sampling_rate, x_targets, y_targets, noise_level, method)
        testcouple.plot(infos=infos, colors=colors, noisy_Q=True, fmin=fmin, fmax=fmax)
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
    builder = Builder(cache_dir='cache-parallel')
    #builder = Builder(cache_dir='muell-cache')
    #x_targets = num.array([1000., 10000., 20000., 30000., 40000., 50000.])
    #d1s = num.arange(5000., 50000., 300.)
    d1s = num.linspace(1000., 100000., 16)[2:]
    ##### ready
    #z1 = 10.*km
    #z2 = 14.*km
    z1 = 11.*km
    z2 = 13.*km
    #z1 = 12.4*km
    #z2 = 12.8*km
    #z1 = 12.5*km
    #z2 = 12.6*km
    ############
    #z1 = 11.*km
    #z2 = 13.*km
    #z1 = 12.*km
    #z2 = 12.1*km

    #if True:
    #    d2s = d1s*z2/z1
    parallel = True

    #x_targets = num.array([d1, d2])
    #y_targets = num.array([0.]*len(x_targets))

    lat = 50.2059
    lon = 12.5152
    sampling_rate = 500.
    #sampling_rate = 20.
    time_window = 24.
    n_repeat = 1000
    sources = []
    method = 'mtspec'
    #method = 'pymutt'
    strike = 170.
    dip = 70.
    rake = -30.
    fmin = 50.
    fmax = 150.
    source_mech = qseis.QSeisSourceMechMT(mnn=1E6, mee=1E6, mdd=1E6)
    #source_mech = qseis.QSeisSourceMechSDR(strike=strike, dip=dip, rake=rake)
    #source_mech.m_dc *= 1E10
    earthmodel = 'models/constantall.nd'
    #earthmodel = 'models/inv_test2_simple.nd'
    #earthmodel = 'models/inv_test6.nd'
    mod = cake.load_model(earthmodel)
    component = 'r'
    target_kwargs = {
        'elevation': 0, 'codes': ('', 'KVC', '', 'Z'), 'store_id': None}

    #targets = xy2targets(x_targets, y_targets, lat, lon, 'Z',  **target_kwargs)
    #logger.info('Using %s targets' % (len(targets)))

    tracers = []
    source_depths = [z1, z2]
    p_chopper = Chopper('first(p|P)', fixed_length=0.4, phase_position=0.5,
                        phaser=PhasePie(mod=mod))

    sources = [HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=sd,
                          strike=strike,
                          dip=dip,
                          rake=rake,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='0%i'%(sd/1000.),
                          attitude='master') for sd in source_depths]
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
            config.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
            config.time_window = time_window
            config.nsamples = (sampling_rate*config.time_window)+1
            config.earthmodel_1d = mod
            config.source_mech = source_mech

            slow = p_chopper.arrival(sources[itarget], target).p/(cake.r2d*cake.d2m/cake.km)
            config.slowness_window = [slow*0.5, slow*0.9, slow*1.1, slow*1.5]

            tracer = Tracer(sources[itarget], target, p_chopper, config=config,
                            component=component)
            tracers.append(tracer)
            pair.append(tracer)
        pairs.append(pair)


    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1

    colors = UniqueColor(tracers=tracers)
    location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    fig = plt.gcf()
    fig.savefig('locations_model_parallel%s.png'%parallel)
    plt.show()
    tracers = builder.build(tracers, snuffle=False)
    noise = RandomNoise(noise_level)
    testcouples = []
    for pair in pairs:
        testcouple = SyntheticCouple(master_slave=pair, fmin=fmin, fmax=fmax)
        testcouples.append(testcouple)

    #pool = multiprocessing.Pool()
    args = [(tc, n_repeat, noise) for tc in testcouples]
    for arg in args:
        process_couple(arg)
    print 'Fix parallel version'
    #pool.map(process_couple, args)
    dist_vs_Q = []
    for i_tc, testcouple in enumerate(testcouples):

        infos = '''
        Strike: %s\nDip: %s\n Rake: %s\n Sampling rate [Hz]: %s\n 
        noise_level: %s\nmethod: %s
        ''' % (strike, dip, rake, sampling_rate, noise_level, method)
        
        # return the list of noisy Qs. This should be cleaned up later.....!
        Qs = testcouple.plot(infos=infos, colors=colors, noisy_Q=True, fmin=fmin, fmax=fmax)
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


def dbtest(noise_level=0.001):
    print '-------------------db test-------------------------------'
    builder = Builder()
    lat = 50.2059
    lon = 12.5152
    n_repeat = 10
    sources = []
    method = 'mtspec'
    strike = 170.
    dip = 70.
    rake = -30.
    fmin = 10.
    fmax = 40.

    store_id = 'qplayground_30000m'

    engine = LocalEngine(store_superdirs=['/data/stores'])
    store = engine.get_store(store_id)
    config = engine.get_store_config(store_id)
    mod = config.earthmodel_1d
    component = 'Z'
    target_kwargs = {
        'elevation': 0, 'codes': ('', 'KVC', '', component), 'store_id': store_id}
    targets = [Target(lat=lat, lon=lon, **target_kwargs)]
    p_chopper = Chopper('first(p|P)', fixed_length=0.8, phase_position=0.5,
                        phaser=PhasePie(mod=mod))
    tracers = []
    source_depths = num.arange(10100, 14000, 200)
    distances = num.arange(28000., 32000., 200)

    #source_depths = num.arange(config.source_depth_min,
    #                           config.source_depth_max+config.source_depth_delta,
    #                           config.source_depth_delta*4)
    #distances = num.arange(config.distance_min,
    #                       config.distance_max+config.distance_delta,
    #                       config.distance_delta*4)

    #sources = [DCSource(lat=float(lat), lon=float(lon), depth=sd, strike=strike,
    #               dip=dip, rake=rake, magnitude=1.5) for sd in source_depths]
    sources = []
    for d in distances:
        sources.extend([ExplosionSource(
            lat=float(lat),
            lon=float(lon),
            depth=sd,
            magnitude=0.,
            north_shift=d) for sd in source_depths])

    plot_locations(sources, use_effective_latlon=True)
    plt.show()
    from distance_point2line import process as event_pairing
    from distance_point2line import plot_segments, get_3d_ax, filter_pairs
    ### Should use die Pie here:
    pairs_by_rays = event_pairing(sources, targets, mod, cake.PhaseDef('p'))

    pairs_by_rays = filter_pairs(pairs_by_rays, 10, 2000)
    fig, ax = get_3d_ax()
    #widgets = ['plotting segments: ', progressbar.Percentage(), progressbar.Bar()]
    #pb = progressbar.ProgressBar(maxval=len(pairs_by_rays)-1, widgets=widgets).start()
    #for i_r, r in enumerate(pairs_by_rays):
    #    s, e1, e2, segments = r
    #    plot_segments(segments, ax=ax)
    #    pb.update(i_r)
    #pb.finish()
    pairs = []
    for r in pairs_by_rays:
        t, s1, s2, segments = r
        tracer1 = Tracer(s1, t, p_chopper, component=component)
        tracer2 = Tracer(s2, t, p_chopper, component=component)
        pair = [tracer1, tracer2]
        tracers.extend(pair)
        pairs.append(pair)

    tracers = builder.build(tracers, engine=engine, snuffle=False)
    colors = UniqueColor(tracers=tracers)
    noise = RandomNoise(noise_level)
    testcouples = []
    widgets = ['processing couples: ', progressbar.Percentage(), progressbar.Bar()]
    pb = progressbar.ProgressBar(maxval=len(pairs)-1, widgets=widgets).start()
    for i_p, pair in enumerate(pairs):
        pb.update(i_p)
        testcouple = SyntheticCouple(master_slave=pair, fmin=fmin, fmax=fmax)
        testcouple.process(method=method, repeat=n_repeat, noise=noise)
        testcouples.append(testcouple)
    pb.finish()
    #outfn = 'testimage'
    #plt.gcf().savefig('output/%s.png' % outfn)
    testcouples = filter(lambda x: x.delta_onset()>0.06, testcouples)
    inverter = QInverter(couples=testcouples)
    inverter.invert()
    inverter.plot(q_threshold=500)
    #location_plots(tracers, colors=colors, background_model=mod, parameter='vp')
    plt.show()


if __name__=='__main__':
    #invert_test_1()
    #qpqs()
    
    # DIESER:
    #invert_test_2()
    #invert_test_2D(noise_level=0.0000001)
    invert_test_2D_parallel(noise_level=0.01)
    #dbtest(noise_level=0.00000001)
    #noise_test()
    #qp_model_test()
    #constant_qp_test()
    #sdr_test()
    #vp_model_test()
    #sys.exit(0)
