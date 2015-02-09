from pyrocko.gf import *
from pyrocko.guts import *
from pyrocko import orthodrome
from pyrocko import trace
from collections import defaultdict
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as num
import sys
import os

try:
    import pymutt
except as e:
    print e
try:
    from mtspec import mtspec
except as e:
    print e


def test_data(f, deltat=1., samples=100, noise=False):
    t = num.arange(0., samples*deltat, deltat)
    ydata = num.sin(2*num.pi*f*t)
    if noise:
        ydata += num.random.randn(samples)
    return ydata

def multitaper_spectrum(tr):
    ydata = tr.get_ydata()
    xdata = tr.get_xdata()

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

class Spectra(DDContainer):
    def __init__(self, *args, **kwargs):
        DDContainer.__init__(self, args, kwargs)
        self.specs = DDContainer()

    def plot_all(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        count = 0
        for s, ta, fxfy in self.iterdd():
            fx, fy = fxfy
            color = colors[count%len(colors)]
            ax.plot(fx, fy, label=ta.store_id, color=color)
            count += 1
        ax.autoscale()
        #ax.set_yscale("log")
        #ax.set_xscale("log")
        #ax.legend()

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

        handles, labels = ax.get_legend_handles_labels()
        slope_string = '\n'.join(map(str, slopes))
        ax.text(0.1,0.9,slope_string, verticalalignment='top',
                                horizontalalignment='left', 
                                transform=ax.transAxes)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()

    def linregress(self, fmin=-999, fmax=999):
        regressions = defaultdict(dict)
        for s, ta, fxfy in self.specs.iterdd():
            fx, fy = fxfy
            indx = num.where(num.logical_and(fx>=fmin, fx<=fmax))
            slope, interc, r_value, p_value, stderr = linregress(fx[indx],
                    num.log(num.abs(fy[indx])))
            regressions[s][ta] = (slope, interc)
        
        return regressions


class SyntheticCouple():
    def __init__(self, master_slave, target, engine):
        self.master_slave = master_slave
        self.target = target
        self.engine = engine
        self.spectra = Spectra()
        self.phase_table = None

    def process(self, chopper, method='fft'):
        responses = []
        for i,s in enumerate(self.master_slave):
            self.target.store_id = s.store_id
            response = e.process(sources=[s], targets=self.target)
            store = self.engine.get_store(s.store_id)
            chopper.update(store, s, self.target)
            responses.append(response)
        
        chopper.chop(responses)
        for s, t, tr in chopper.iter_results():
            tr.set_ydata(test_data(f=0.3, deltat=tr.deltat, samples=len(tr.ydata),
                                   noise=False))
            if method=='fft':
                f, a = tr.spectrum(tfade=chopper.get_tfade())
            elif method=='mt':
                r = pymutt.mtft(tr.ydata, dt=tr.deltat)
                f = num.arange(r['nspec'])*r['df']
                a = r['power']
            elif method=='mtspec'
            self.spectra.add(s, t, [f, num.abs(a)])
        
        self.phase_table = chopper.table

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
            A.append(num.abs(fxfy[i][1]))
        t1, t2 = arrivals[::-1]
        assert all(fxfy[0][0]==fxfy[1][0])
        q = num.pi*num.abs(num.array(fxfy[0][0]))*(t2-t1)/(num.log(A[0])+num.log(dists[0])-num.log(A[1])-num.log(dists[1]))

class Chopper():
    def __init__(self, startphasestr, endphasestr=None, fixed_length=None, fade_factor=0., default_store=None):
        self.startphasestr = startphasestr
        self.endphasestr = endphasestr
        self.fade_factor = fade_factor
        self.fixed_length = fixed_length
        self.current_store = default_store
        self.table = DDContainer()
        self.tfade = 0.
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

        deltat = max(traces, key=lambda x: x.deltat).deltat
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
            self.tfade = (tend-tstart)*self.fade_factor
            tr.chop(tstart, tend)
            tr.downsample_to(deltat)
            returntraces.append(tr)
            self.chopped.add(s,t,tr)
        
    def iter_results(self):
        return self.chopped.iterdd()

    def update(self, store, source, target):
        depth = source.depth
        dist = source.distance_to(target)
        tstart = store.t(self.startphasestr, (depth, dist))
        tend = store.t(self.endphasestr, (depth, dist))
        self.table.add(store.config.id, (depth, dist), (tstart, tend))

    def set_trange(self):
        _,_, times = self.table.as_lists()
        
        mintrange = 99999.
        for tstart, tend in times:
            trange = tstart-tend 
            mintrange = trange if trange < mintrange else 99999.
        return mintrange

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
             store_id=store_ids[1])

t = Target(lat=lat,
           lon=lon,
           elevation=0,
           codes=('', 'KVC', '', 'Z'),
           store_id=None)

superdirs = ['/home/marius']
superdirs.append(os.environ['STORES'])
e = LocalEngine(store_superdirs=superdirs)

testcouple = SyntheticCouple(master_slave=[s1, s2], target=t, engine=e)
#chopper = Chopper('first(p|P)', 'first(s|S)', fade_factor=0.3)
chopper = Chopper('first(s|S)', fixed_length=3, fade_factor=0.3)
testcouple.process(chopper=chopper)
#testcouple.q('p')
testcouple.plot()
#plt.show()
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
    
