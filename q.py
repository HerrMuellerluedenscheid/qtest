from pyrocko.gf import *
from pyrocko.guts import *
from pyrocko import orthodrome
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as num
import sys

class DCSourceWid(DCSource):
    store_id = String.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)

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

class Spectra(DDContainer):
    def __init__(self, *args, **kwargs):
        DDContainer.__init__(self, args, kwargs)
        self.specs = DDContainer()

    def plot_all(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fmin = 1.
        fmax = 20.
        regs = self.linregress(fmin=fmin, fmax=fmax)
        count = 0
        slopes = []
        for s, ta, fxfy in self.specs.iterdd():
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

        handles, labels = ax.get_legend_handles_labels()
        slope_string = '\n'.join(map(str, slopes))
        ax.text(0.1,0.9,slope_string, verticalalignment='top',
                                horizontalalignment='left', 
                                transform=ax.transAxes)

        ax.set_yscale("log")
        ax.legend()
        plt.show()

    def linregress(self, fmin=-999, fmax=999):
        regressions = defaultdict(dict)
        for s, ta, fxfy in self.specs.iterdd():
            fx, fy = fxfy
            indx = num.where(num.logical_and(fx>=fmin, fx<=fmax))
            slope, interc, r_value, p_value, stderr = linregress(fx[indx],
                    num.log(num.abs(fy[indx])))
            regressions[s][ta] = (slope, interc)
        
        return regressions


class Spectrum():
    """
    TODO... 
    """
    def __init__(self, fxfy, target=None, source=None):
        self.fx, self.fy = fxfy
        self.target = target
        self.source = source


class SyntheticCouple():
    def __init__(self, master_slave, target, engine):
        self.master_slave = master_slave
        self.target = target
        self.engine = engine
        self.spectra = Spectra()

    def process(self, chopper):

        chopper.prepare(self.master_slave)
        for i,s in enumerate(self.master_slave):
            self.target.store_id = s.store_id
            chopper.update(self.engine.get_store(s.store_id))
            response = e.process(sources=[s], targets=self.target)
            tr = response.pyrocko_traces()
            assert len(tr)==1
            tr = tr[0]
            chopper.chop(s, self.target, tr)
            self.spectra.add(s, self.target,
                    tr.spectrum(tfade=chopper.get_tfade()))
        
    def plot(self):
        self.spectra.plot_all()
        
    def q(self, phasestr):
        # following eqn:
        # http://www.ga.gov.au/corporate_data/81414/Jou1995_v15_n4_p511.pdf (5)
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
            A.append(fxfy[i][1])
        print freqs
        t2, t1 = arrivals[::-1]
        q = num.pi*fx*(t2-t1)
        #/(num.log(num.abs(specs[0]))+num.log(dists[0]-num.log(num.abs(specs[1]))-num.log(dists[1])))


class Holder():
    def __init__(self, func):
        self.func = func
        self.items = []

    def add(self, item):
        self.items.append(item)

    def go(self):
        for i in items:
            if caller:
                result = caller.func(**kwargs)
            else:
                result = func(**kwargs)

            yield 


class Chopper():
    def __init__(self, startphasestr, endphasestr=None, fixed_length=None, fade_factor=0., default_store=None):
        self.startphasestr = startphasestr
        self.endphasestr = endphasestr
        self.fade_factor = fade_factor
        self.fixed_length = fixed_length
        self.current_store = default_store

        self.tfade = 0.

    def chop(self, source, target, tr):

        dist = source.distance_to(target)
        tstart = self.current_store.t(self.startphasestr, (source.depth, dist))
        if self.endphasestr!=None and self.fixed_length==None:
            tend = self.current_store.t(self.endphasestr, (source.depth, dist))
        elif self.endphasestr==None and self.fixed_length!=None:
            tend = self.fixed_length+tstart
        else:
            raise Exception('Need to define exactly one of endphasestr and fixed length')
        self.tfade = (tend-tstart)*self.fade_factor
        tr.chop(source.time+tstart-self.tfade, source.time+tstart+tend+self.tfade)
        tr.run_chain()
        return tr
        
    def update(self, store):
        self.current_store = store

    def get_tfade(self):
        return self.tfade
    
#100Hz
#store_ids = ['vogtland_%s'%i for i in [1,2,3]]
# 50Hz
#store_ids = ['vogtland_%s'%i for i in [5,6,7]]
store_ids = ['vogtland_%s'%i for i in [7, 6]]
lat = 50.2059
lon = 12.5152
depth = 8500.
dist1 = 10000.
dist2 = 5000.

sources = []
targets = [] 
#for store_id in store_ids:
t = Target(lat=lat,
           lon=lon,
           elevation=0,
           codes=('', 'KVC', '', 'Z'),
           store_id=None)
#targets.append(t)

lat_s, lon_s = orthodrome.ne_to_latlon(lat, lon, dist1, 0.)
lat_s2, lon_s2 = orthodrome.ne_to_latlon(lat, lon, dist2, 0.)

s1 = DCSourceWid(lat=float(lat_s),
             lon=float(lon_s),
             depth=depth,
             strike=170.,
             dip=80.,
             rake=-30.,
             magnitude=1.5, 
             store_id=store_ids[0])

s2 = DCSourceWid(lat=float(lat_s2),
             lon=float(lon_s2),
             depth=depth,
             strike=170.,
             dip=80.,
             rake=-30.,
             magnitude=1.5,
             store_id=store_ids[0])

t = Target(lat=lat,
           lon=lon,
           elevation=0,
           codes=('', 'KVC', '', 'Z'),
           store_id=None)

superdirs = ['/home/marius']
e = LocalEngine(store_superdirs=superdirs)

testcouple = SyntheticCouple(master_slave=[s1, s2], target=t, engine=e)
#chopper = Chopper('first(p|P)', 'first(s|S)', fade_factor=0.3)
chopper = Chopper('first(p|P)', fixed_length=0.3, fade_factor=0.3)
testcouple.process(chopper=chopper)
testcouple.q('p')
testcouple.plot()
plt.show()
# END
# ------------------------------------------------------------------------------------------
sys.exit(0)
traces = {}
response = e.process(sources=sources, targets=targets)
spectra = Spectra()

traces = []
fade_factor = 0.5
for source, target, tr in response.iter_results():
    store = e.get_store(target.store_id)
    tstart = store.t('first(p|P)', (depth, dist))
    tstart += source.time
    #tend = store.t('first(s|S)', (depth, dist))
    tend = 1.

    tend += tstart
    tfade = (tend-tstart)*fade_factor
    tend += tfade
    tstart -= tfade
    tr.station = target.store_id[-1]
    traces.append(tr.copy())
    tr.chop(tstart, tend)
    traces.append(tr)
    fx, fy = tr.spectrum(tfade=tfade)

    spectra.add(source, target, (fx, fy))
spectra.plot_all()
trace.snuffle(traces)
    
