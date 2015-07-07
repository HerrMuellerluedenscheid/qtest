import matplotlib
matplotlib.use('Qt4Agg')
from pyrocko.gf import *
from pyrocko.guts import *
from pyrocko import gui_util
from pyrocko import orthodrome
from pyrocko import trace
from pyrocko.fomosto import qseis

from collections import defaultdict
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as num
import sys
import os
from micro_engine import OnDemandEngine, OnDemandStore

import logging 
logging.basicConfig(level=logging.DEBUG)
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

def xy2targets(x, y, o_lat, o_lon, **kwargs):
    targets = []
    for xy in zip(x,y):
        lat, lon = orthodrome.ne_to_latlon(o_lat, o_lon, *xy)
        kwargs.update({'lat':lat, 'lon':lon})
        targets.append(Target(**kwargs))
                   
    return targets

def plot_locations(items):
    f = plt.figure()
    ax = f.add_subplot(111)
    for item in items:
        ax.plot(item.lon, item.lat, 'o')
    ax.set_title('locations')
    plt.show()

def check_method(method):
    if not method in methods_avail:
        logger.exception("Method %s not available" % method)
        raise Exception("Method %s not available" % method)
    else: 
        logger.debug('method used %s' %method)
        return True

def test_data(f, deltat=1., samples=100, noise=False):
    t = num.arange(0., samples*deltat, deltat)
    ydata = num.sin(2*num.pi*f*t)
    if noise:
        ydata += num.random.randn(samples)
    return ydata

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

class DCSourceWid(DCSource):
    store_id = String.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)

class HaskellSourceWid(RectangularSource):
    store_id = String.T(optional=True, default=None)
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
colors = "rgbcmyb"

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

    def set_fit_function(self, func):
        self.fit_function = func

    def get_slopes(self):
        slopes = DDContainer()
        for s, ta, fxfy in self.iterdd():
            fx, fy = num.vsplit(fxfy, 2)
            fc = 1./(0.5*s.risetime)
            slope_section = self.extract(fxfy, upper_lim=fc, lower_lim=22)
            popt, pcov = curve_fit( self.fit_function, *slope_section)
            fy_fit = self.fit_function(slope_section[0], popt[0], popt[1]),
            slopes.add(s, ta, [slope_section[0], num.array(fy_fit).T, popt[0], popt[1]])

        return slopes

    def plot_all(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        count = 0
        slopes = self.get_slopes()
        for s, ta, fxfy in self.iterdd():
            fx, fy = num.vsplit(fxfy, 2)
            
            #redundant
            #fc = 1./(0.5*s.risetime)
            #slope_section = self.extract(fxfy, upper_lim=fc, lower_lim=22)
            #popt, pcov = curve_fit( self.fit_function, *slope_section)
            color = colors[count%len(colors)]
            ax.plot(fx.T, fy.T, label=s.store_id, color=color)
            slope = slopes[(s,ta)]
            print slope[0].shape
            print slope[1].shape
            ax.plot(slope[0], slope[1], 'o', label=s.store_id, color=color)
            #slopes.append(popt[0])
            count += 1
        ax.autoscale()
        ax.set_yscale("log")
        # slopes hat jetzt 4 eintraege. von interesse: 3 
        slope_string = ''
        for s, ta, slope_data in slopes.iterdd():
            print slopes
            slope_string += '%s\n'%slope_data[2]
        ax.text(0.1,0.9,slope_string, verticalalignment='top',
                                horizontalalignment='left', 
                                transform=ax.transAxes)

        ax.legend()

    def extract(self, fxfy, upper_lim=0., lower_lim=99999.):
        indx = num.where(num.logical_and(fxfy[0]>=upper_lim, fxfy[0]<=lower_lim))
        indx = num.array(indx)
        return fxfy[:, indx].reshape(2, len(indx.T))


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

class Couples():
    def __init__(self, couples=None):
        if couples==None:
            couples= []
        self.couples = couples 

    def append(self, item):
        self.couples.append(item)

    def get_average_spectrum(self):
        fxs = []
        fys = []
        for couple in self.couples:
            spectra = couple.get_spectra()
            for s, ta, fxfy in spectra.iterdd():
                if s.is_slave:
                    fxs.append(list(fxfy[0]))
                    fys.append(fxfy[1])

        #fxs = num.asarray(fxs)
        fys = num.asarray(fys)
        intersecting, average_spectrum = average_spectra(fxs, fys)
        return intersecting ,average_spectrum

class SyntheticCouple():
    def __init__(self, master_slave, targets, engine):
        self.master_slave = master_slave
        self.targets = targets
        self.engine = engine
        self.spectra = Spectra()
        self.phase_table = None

    def process(self, chopper, method='mtspec'):
        check_method(method)
        responses = []
        for i, s in enumerate(self.master_slave):
            for t in self.targets:
                t.store_id = s.store_id
                response = self.engine.process(sources=[s], targets=[t])
                chopper.update(self.engine, s, t)
                responses.append(response)
        
        chopper.chop(responses)
        for s, t, tr in chopper.iter_results():
            if method=='fft':
                # fuehrt zum amplitudenspektrum
                f, a = tr.spectrum(tfade=chopper.get_tfade())

            elif method=='pymutt':
                # berechnet die power spectral density
                r = pymutt.mtft(tr.ydata, dt=tr.deltat)
                f = num.arange(r['nspec'])*r['df']
                a = r['power']

            elif method=='mtspec':
                # nfft -> zero padding
                a, f = mtspec(data=tr.ydata,
                              delta=tr.deltat,
                              number_of_tapers=5,
                              time_bandwidth=2,
                              nfft=150,
                              statistics=False)

            fxfy = num.vstack((f,a))
            self.spectra.add(s, t, fxfy)
        
        self.phase_table = chopper.table

    def plot(self):
        self.spectra.plot_all()
    
    def set_fit_function(self, func):
        self.spectra.set_fit_function(func)

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
        #regresses = self.spectra.linregress(1,20)
        #slope = regresses[1]/regresses[0]
        print 'WARNING: hard coded slope'
        slope = 0.3049/0.25
        plt.plot(fxfy[0], _q(fxfy[1], arrivals[::-1], slope=slope))

    def _q_1(self, fxfy, A, dists, arrivals):
        # following eqn:
        # http://www.ga.gov.au/corporate_data/81414/Jou1995_v15_n4_p511.pdf (5)
        t1, t2 = arrivals
        return num.pi*num.abs(num.array(fxfy[0][0]))*(t2-t1)/(num.log(A[0])+num.log(dists[0])-num.log(A[1])-num.log(dists[1]))

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
        intersecting_master, average_spectrum_master = average_spectra(fx_master, fy_master)
        intersecting_slave, average_spectrum_slave = average_spectra(fx_slave, fy_slave)
        return intersecting_master, average_spectrum_master, intersecting_slave , average_spectrum_slave
        

class Chopper():
    def __init__(self, startphasestr, endphasestr=None, fixed_length=None, phase_position=0., default_store=None):
        self.startphasestr = startphasestr
        self.endphasestr = endphasestr
        self.phase_position = phase_position
        self.fixed_length = fixed_length
        self.current_store = default_store
        self.table = DDContainer()
        self.chopped = DDContainer()

    def chop(self, responses):
        sources = []
        targets = [] 
        traces = []

        for response in responses:
            for so,ta,tr in response.iter_results():
                sources.append(so)
                targets.append(ta)
                traces.append(tr)

        #deltat = max(traces, key=lambda x: x.deltat).deltat
        trange = self.set_trange()
        returntraces = []
        for i in range(len(sources)):
            s = sources[i]
            t = targets[i]
            tr = traces[i]
            dist = s.distance_to(t)
            depth = s.depth
            tstart, end = self.table[s.store_id, (depth, dist)]
            if self.endphasestr!=None and self.fixed_length==None:
                tend = tend
            elif self.endphasestr==None and self.fixed_length!=None:
                tend = self.fixed_length+tstart
            else:
                raise Exception('Need to define exactly one of endphasestr and fixed length')

            
            # aufraeumen! 
            logger.info('lowpass filter with 0.5/nyquist')
            tr.lowpass(4, 0.4/tr.deltat)
            trange = tend-tstart
            tstart -= trange*self.phase_position
            tend -= trange*self.phase_position
            mark = gui_util.PhaseMarker(tmin=tstart, tmax=tend,
                                        nslc_ids=[tr.nslc_id])
            tr.chop(tstart, tend)
            #tr.downsample_to(deltat)
            returntraces.append(tr)
            self.chopped.add(s,t,tr)
        
    def iter_results(self):
        return self.chopped.iterdd()

    def update(self, engine, source, target):
        depth = source.depth
        dist = source.distance_to(target)
        if isinstance(engine, LocalEngine):
            store = engine.get_store(source.store_id)
            tstart = store.t(self.startphasestr, (depth, dist))
            tend = store.t(self.endphasestr, (depth, dist))
            model_id = store.store_id
        elif isinstance(engine, OnDemandEngine):
            tstart = engine.config.earthmodel_1d.arrivals((self.startphasestr,
                                                           (depth, dist)))
            tend = engine.config.earthmodel_1d.arrivals((self.startphasestr,
                                                           (depth, dist)))
            model_id = engine.config.earthmodel_id

        self.table.add(model_id, (depth, dist), (tstart, tend))


    def set_trange(self):
        _,_, times = self.table.as_lists()
        
        mintrange = 99999.
        for tstart, tend in times:
            trange = tstart-tend 
            mintrange = trange if trange < mintrange else 99999.
        return mintrange
    

#x_targets = num.array([0,0,0,0,-10,-5,5,10])
#y_targets = num.array([-10,-5,5,10,0,0,0,0])
x_targets = num.array([0])
y_targets = num.array([0])
x_targets *= km
y_targets *= km

#100Hz
#store_ids = ['vogtland_%s'%i for i in [1,2,3]]
# 50Hz
#store_ids = ['vogtland_%s'%i for i in [5,6,7]]
store_ids = ['vogtland_%s'%i for i in [7, 6]]
lat = 50.2059
lon = 12.5152
depth = 8500.

sources = []
targets = [] 
#for store_id in store_ids:
target_kwargs = {'elevation': 0,
                 'codes': ('', 'KVC', '', 'Z'),
                 'store_id': None}

targets = xy2targets(x_targets, y_targets, lat, lon, **target_kwargs)
#plot_locations(targets)

slopes = [] 
couples = Couples()
superdirs = ['/home/marius']
superdirs.append(os.environ['STORES'])
#e = LocalEngine(store_superdirs=superdirs)
chopper = Chopper('first(p|P)', fixed_length=1.5, phase_position=0.4)

store_ids = ['test0', 'test1']

config1 = qseis.QSeisConfigFull.example()
config1.time_region = [meta.Timing(-10.), meta.Timing(30.)]
config1.time_window = 40. 
config1.nsamples = 40/0.5
config1.earthmodel_id = store_ids[0]
config1.earthmodel_1d = config1.earthmodel_1d.extract(0, 80000)

config2 = qseis.QSeisConfigFull.example()
config2.time_region = [meta.Timing(-10.), meta.Timing(30.)]
config2.time_window = 40. 
config2.nsamples = 40/0.5
config2.earthmodel_id = [store_ids[1]]
config2.earthmodel_1d = config2.earthmodel_1d.extract(0, 80000)


configs = [config1, config2]
on_demand_stores = []
for i in range(2):
    on_demand_stores.append(OnDemandStore(config=configs[i],
                                          store_id=store_ids[i]))

s1 = HaskellSourceWid(lat=float(lat),
                      lon=float(lon),
                      depth=depth,
                      strike=170.,
                      dip=80.,
                      rake=-30.,
                      magnitude=1.5, 
                      length=400.,
                      width=250.,
                      risetime=0.6,
                      store_id=store_ids[0],
                      attitude='master')

s2 = HaskellSourceWid(lat=float(lat),
                      lon=float(lon),
                      depth=depth,
                      strike=170.,
                      dip=80.,
                      rake=-30.,
                      magnitude=1.5,
                      length=400.,
                      width=250.,
                      risetime=0.6,
                      store_id=store_ids[1], 
                      attitude='slave')
    
traces_on_demand = OnDemandEngine(on_demand_stores=on_demand_stores)
traces_on_demand.process(targets=targets, 
                         sources=[s1, s2])
testcouple = SyntheticCouple(master_slave=[s1, s2], targets=targets, engine=e)


testcouple = SyntheticCouple(master_slave=[s1, s2], targets=targets,
                             engine=traces_on_demand)
#chopper = Chopper('first(s|S)', fixed_length=1.5, phase_position=0.4)
testcouple.process(chopper=chopper)
testcouple.set_fit_function(exp_fit)
#testcouple.plot()
fx_ma, fy_ma, fx_sl, fy_sl = testcouple.get_average_spectra()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fx_ma, fy_ma, 'x', label="master")
ax.plot(fx_sl, fy_sl, 'x', label="slave")
ax.set_yscale('log')
plt.show()
couples.append(testcouple)
slopes.append(testcouple.get_slopes())

compile_slopes(slopes)

#x, y = couples.get_average_spectrum()
# END
# ------------------------------------------------------------------------------------------
sys.exit(0)
