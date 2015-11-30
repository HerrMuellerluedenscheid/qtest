import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rc('ytick', labelsize=8) 
matplotlib.rc('xtick', labelsize=8) 

import multiprocessing

from pyrocko.gf import meta, DCSource, RectangularSource, Target
from pyrocko.guts import *
from pyrocko import gui_util
from pyrocko import orthodrome
from pyrocko import trace
from pyrocko import cake
from pyrocko import crust2x2
from pyrocko.fomosto import qseis

from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as num
import sys
import os
from micro_engine import OnDemandEngine, Tracer, create_store
from autogain.autogain import PhasePie
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

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

def _q(freqs, arrivals, slope):
    # nach Scherbaum 1990
    t1, t2 = arrivals
    t = t2-t1

    return -num.pi*num.array(freqs)*t/slope 

def add_noise(t, level):
    ydata = t.get_ydata()
    noise = num.random.random(len(ydata))-0.5
    noise *= ((num.max(num.abs(ydata))) * level)
    ydata += noise
    t.set_ydata(ydata)
    return t

def legend_clear_duplicates(ax):
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def xy2targets(x, y, o_lat, o_lon, channels, **kwargs):
    targets = []
    for istat, xy in enumerate(zip(x,y)):
        print istat
        for c in channels:
            lat, lon = orthodrome.ne_to_latlon(o_lat, o_lon, *xy)
            kwargs.update({'lat': float(lat), 'lon': float(lon),
                           'codes':('', '%i'%istat, '', c)})
            targets.append(Target(**kwargs))
    return targets

def ax_if_needed(ax):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    return ax

def plot_traces(tr, ax=None, label='', color='r'):
    ax = ax_if_needed(ax)
    ax.plot(tr.get_xdata(), tr.get_ydata(), label=label, color=color)

def plot_model(mod, ax=None, label='', color=None):
    ax = ax_if_needed(ax)
    z = mod.profile('z')
    profile = mod.profile('qp')
    ax.plot(profile, -z, label=label, c=color)
    ax.set_title('Qp')

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

def check_method(method):
    if not method in methods_avail:
        logger.exception("Method %s not available" % method)
        raise Exception("Method %s not available" % method)
    else:
        logger.debug('method used %s' %method)
        return True

def compile_slopes(slopes):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for slope in slopes:
        for s, ta, slope_data in slope.iterdd():
            x, y, m, interc = slope_data
            if s.attitude=='master':
                ax.plot(x,y,'o',alpha=0.5)
            else:
                ax.plot(x,y,alpha=0.5)

    ax.set_yscale("log")

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

class Spectra(DDContainer):
    def __init__(self, *args, **kwargs):
        DDContainer.__init__(self, args, kwargs)
        self.fit_function = None
        self.spectra = []

    def set_fit_function(self, func):
        self.fit_function = func

    def get_slopes(self):
        slopes = DDContainer()
        for s, ta, fxfy in self.iterdd():
            fx, fy = num.vsplit(fxfy, 2)
            fc = 1./(0.5*s.risetime)
            slope_section = extract(fxfy, upper_lim=fc, lower_lim=22)
            popt, pcov = curve_fit( self.fit_function, *slope_section)
            fy_fit = self.fit_function(slope_section[0], popt[0], popt[1]),
            slopes.add(s, ta, [slope_section[0], num.array(fy_fit).T, popt[0], popt[1]])

        return slopes

    def plot_all(self, ax=None, colors=None, alpha=1.):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        count = 0
        #slopes = self.get_slopes()
        #for s, ta, fxfy in self.iterdd():
        for tracer, fxfy in self.spectra:
            fx, fy = num.vsplit(fxfy, 2)
            if colors:
                color = colors[tracer.config.id]
            ax.plot(fx.T, fy.T, label=tracer.config.id, color=color, alpha=alpha)
            #slope = slopes[(s,ta)]
            #print slope[0].shape
            #print slope[1].shape
            #ax.plot(slope[0], slope[1], 'o', label=s.store_id, color=color)
            #slopes.append(popt[0])
            count += 1
        ax.autoscale()
        ax.set_yscale("log")
        # slopes hat jetzt 4 eintraege. von interesse: 3 
        slope_string = ''
        #for s, ta, slope_data in slopes.iterdd():
        #    print slopes
        #    slope_string += '%s\n'%slope_data[2]
#        ax.text(0.1,0.9,slope_string, verticalalignment='top',
#                                horizontalalignment='left', 
#                                transform=ax.transAxes)
#
        ax.legend()

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

class UniqueColors():
    def __init__(self):
        self.all_colors = "rgbcmyb"
        self.colors = {}

    def __getitem__(self, key):
        if not key in self.colors.keys():
            for c in self.all_colors:
                if c not in self.colors.values():
                    self.colors[key] = c
                    break
            else:
                raise Exception('AllColorsInUse')
        return self.colors[key]


class SyntheticCouple():
    def __init__(self, master_slave):
        self.master_slave = master_slave
        self.spectra = Spectra()
        self.noisy_spectra = Spectra()
        self.phase_table = None
        self.fit_function = None
        self.use_trace_index = 2
        self.colors = None

    def process(self, chopper, method='mtspec', noise_level=0.0, repeat=1):
        check_method(method)
        #for i, s in enumerate(self.master_slave):
        #    t.store_id = s.config.id
        #    response = self.engine.process(sources=[s.source], targets=[s.target])
        #    #chopper.update(self.engine, s, t)
        #    responses.append(response)

        for tracer in self.master_slave:
            s = tracer.source
            t = tracer.target
            tr_raw = tracer.traces[self.use_trace_index]
            tracer.processed = chopper.chop(s, t, tr_raw)
            f, a = self.get_spectrum(tracer.processed, method, chopper)
            fxfy = num.vstack((f,a))
            self.spectra.spectra.append((tracer, fxfy))

            for i in xrange(repeat):
                tr_noise = add_noise(tracer.processed, level=.05)
                f, a = self.get_spectrum(tr_noise, method, chopper)
                fxfy = num.vstack((f,a))
                self.noisy_spectra.spectra.append((tracer, fxfy))


    def get_spectrum(self, tr, method, chopper):
        if method=='fft':
            # fuehrt zum amplitudenspektrum
            f, a = tr.spectrum(tfade=chopper.get_tfade(tr.tmax-tr.tmin))

        elif method=='pymutt':
            # berechnet die power spectral density
            r = pymutt.mtft(tr.ydata, dt=tr.deltat)
            f = num.arange(r['nspec'])*r['df']
            a = r['power']

        elif method=='mtspec':
            # nfft -> zero padding
            a, f = mtspec(data=tr.ydata,
                          delta=tr.deltat,
                          number_of_tapers=10,
                          time_bandwidth=1.2,
                          nfft=2**9,
                          #nfft=250,
                          statistics=False)

        return f, a

    def plot(self, **kwargs):
        colors = kwargs.pop('colors', UniqueColors())
        fig = plt.figure()
        ax = fig.add_subplot(2,2,1)
        self.spectra.plot_all(ax, colors=colors)
        self.noisy_spectra.plot_all(ax, colors=colors, alpha=0.06)
        legend_clear_duplicates(ax)
        ax = fig.add_subplot(2,2,2)
        master, slave = self.master_slave 
        plot_model(mod=master.config.earthmodel_1d,
                   label=master.config.id,
                   color=colors[master.config.id],
                   ax=ax)

        plot_model(mod=slave.config.earthmodel_1d,
                   label=slave.config.id,
                   color=colors[slave.config.id],
                   ax=ax)

        for tr in self.master_slave:
            ax.axhline(-tr.source.depth, ls='--', label='z %s' % tr.config.id,
                       color=colors[tr.config.id])
        ax.legend()
        ax = fig.add_subplot(2, 2, 3)
        for tracer in self.master_slave:
            plot_traces(tr=tracer.processed, ax=ax, label=tr.config.id,
                        color=colors[tracer.config.id])

    def set_fit_function(self, func):
        self.spectra.set_fit_function(func)
        self.fit_function = func

    def slope(self, phasestr):
        f_corner = 3
        for s,ta, fit in self.spectra.linefit(self.fit_function):
            print fit
        #regressions = self.spectra.linregress(fmin=f_corner)
        #slopes = []
        #for s, ta, regress in regressions.iterdd():
        #    slopes.append(regresses)

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
        #print 'WARNING: hard coded slope'
        #slope = 0.3049/0.25
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
                 phase_position=0., xfade=0.2):
        self.phase_pie = PhasePie()
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
        # aufraeumen!
        logger.info('lowpass filter with 0.5/nyquist')
        tr = tr.copy()
        tr.set_location('cp')
        tr.lowpass(4, 0.5/tr.deltat)

        trange = tend-tstart
        tstart -= trange*self.phase_position
        tend -= trange*self.phase_position
        mark = gui_util.PhaseMarker(tmin=tstart, tmax=tend,
                                    nslc_ids=[tr.nslc_id])
        tr.chop(tstart*(1-self.get_xfade()), tend*(1+self.get_xfade()))
        taperer = trace.CosFader(xfrac=self.get_xfade())
        tr.taper(taperer)
        self.chopped.add(s,t,tr)
        trace.snuffle([tr, tr_bkp])
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




if __name__=='__main__':
    x_targets = num.array([10.])
    y_targets = num.array([0.])

    lat = 50.2059
    lon = 12.5152
    sdepth = 10000.
    sampling_rate = 150.
    time_window = 20
    noise_level = 0.00001
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

    slopes = []
    superdirs = ['/home/marius']
    superdirs.append(os.environ['STORES'])

    config1 = qseis.QSeisConfigFull.example()
    config2 = qseis.QSeisConfigFull.example()
    mod1 = cake.load_model('earthmodel1.nd')
    mod2 = cake.load_model('earthmodel2.nd')

    config1.id='C00'
    config1.time_region = [meta.Timing(-time_window/2.), meta.Timing(-time_window/2.)]
    config1.time_window = time_window
    config1.nsamples = (sampling_rate*config1.time_window)+1
    config1.earthmodel_1d = mod1
    config1.source_mech = source_mech

    config2.id='C01'
    config2.time_region = [meta.Timing(-time_window/2.), meta.Timing(time_window/2.)]
    config2.time_window = time_window
    config2.nsamples = (sampling_rate*config2.time_window)+1
    config2.earthmodel_1d = mod2
    config2.source_mech = source_mech

    #on_demand_stores = []
    #for i in range(len(configs)):
    #    on_demand_stores.append(OnDemandStore(config=configs[i]))

    qs = qseis.QSeisConfig()
    qs.qseis_version = '2006a'
    qs.sw_flat_earth_transform = 1
    extra = {'qseis': qs}

    s0 = HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=sdepth,
                          strike=170.,
                          dip=80.,
                          rake=-30.,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='00',
                          attitude='master')

    s1 = HaskellSourceWid(lat=float(lat),
                          lon=float(lon),
                          depth=sdepth,
                          strike=170.,
                          dip=80.,
                          rake=-30.,
                          magnitude=.5,
                          length=0.,
                          width=0.,
                          risetime=0.02,
                          id='01',
                          attitude='slave')

    tracer0 = Tracer(source=s0, target=targets[0], config=config1)
    tracer1 = Tracer(source=s1, target=targets[0], config=config2)

    #p = multiprocessing.Pool()
    #p.map(lambda x: x.run, [tracer0, tracer1])
    tracer0.run(cache_dir='test-cache')
    tracer1.run(cache_dir='test-cache')

    testcouple = SyntheticCouple(master_slave=[tracer0, tracer1])

    #s_chopper = Chopper('first(s|S)', fixed_length=0.2, phase_position=0.1)
    p_chopper = Chopper('first(p|P)', fixed_length=0.1, phase_position=0.1)
    testcouple.process(chopper=p_chopper,
                       method='pymutt',
                       noise_level=noise_level,
                       repeat=500)

    testcouple.plot()
    plt.show()
    #testcouple.process(chopper=p_chopper, method='fft')
    #testcouple.set_fit_function(exp_fit)
    #testcouple.plot()
    fx_ma, fy_ma, fx_sl, fy_sl = testcouple.get_average_spectra()

    slope_ratio = m_slopes/s_slopes

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fx_ma, fy_ma, 'x', label="master")
    ax.plot(fx_sl, fy_sl, 'o', label="slave")
    ax.set_yscale('log')
    ax.set_xscale('log')

    ##couples.append(testcouple)
    #slopes.append(testcouple.get_slopes())
    #
    #compile_slopes(slopes)

    #x, y = couples.get_average_spectrum()
    # END
    # ------------------------------------------------------------------------------------------
    sys.exit(0)
