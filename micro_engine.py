import numpy as num
import os
import tempfile
import glob
from autogain.autogain import PhasePie
from pyrocko.fomosto import qseis
from pyrocko import trace
from pyrocko import guts
from pyrocko import io
from pyrocko.gf import meta, Engine, Target, DCSource
from pyrocko.gf.store import Store
from pyrocko.parimap import parimap
import time
import argparse
import logging
import multiprocessing

pjoin = os.path.join

logger = logging.getLogger()


def add_noise(t, level):
    ydata = t.get_ydata()
    noise = num.random.random(len(ydata))-0.5
    noise *= ((num.max(num.abs(ydata))) * level)
    ydata += noise
    t.set_ydata(ydata)
    return t


class RandomNoise:
    def __init__(self, level):
        self.level = level

    def noisify(self, tr):
        tr = tr.copy()
        return add_noise(tr, self.level)


class Builder:
    def __init__(self, cache_dir=False):
        self.runners = []
        self.cache_dir = cache_dir

    def build(self, tracers):
        ready = []
        need_work = []
        for tr in tracers:
            found_cache = self.cache(tr, load=True)
            if not found_cache:
                need_work.append(tr)
            else:
                ready.append(tr)

        for tr, runner in parimap(self.work, need_work):
            tmpdir = runner.tempdir
            tmin = tr.config.time_start
            tmax = tr.config.time_window+tmin

            traces = self.load_seis(tmpdir, tmin=tmin, tmax=tmax)
            tr.traces = traces
            ready.append(tr)
            if self.cache_dir:
                self.cache(tr, load=False)
            del(runner)

        return ready

    def work(self, tr):
        runner = qseis.QSeisRunner(keep_tmp=True)
        runner.run(config=tr.config)
        return tr, runner

    def cache(self, tr, load):
        configfn='qseis-config.yaml'
        fn = 'traces.mseed'
        found_cache = False
        if self.cache_dir and load:
            subdirs = os.listdir(self.cache_dir)
            for sdir in subdirs:
                config = guts.load(filename=pjoin(self.cache_dir, sdir, configfn))
                tr.config.regularize()
                if str(config)==str(tr.config):
                    tr.read_files(pjoin(self.cache_dir, sdir, fn))
                    found_cache = True

        elif self.cache_dir and not load:
            tmpdir = tempfile.mkdtemp(dir=self.cache_dir)
            tr.config.dump(filename=pjoin(tmpdir, configfn))
            io.save(tr.traces, pjoin(tmpdir, fn))
            logger.info('cached under: %s' % self.cache_dir)
        return found_cache

    def load_seis(self, directory, tmin, tmax):
        # Hier mussen die files seis.tr, seis.... etc. aus der qseis temp directory 
        # geladen werden und wieder in den tracer gefuettert werden.
        fns = glob.glob(pjoin(directory, 'seis.*'))
        traces = []
        for fn in fns:
            data = num.loadtxt(fn, skiprows=1).T
            dt = (tmax-tmin)/(len(data[1])-1)
            trc = trace.Trace(
                ydata=data[1], tmin=tmin, deltat=dt, channel=fn.split('.')[-1][-1])
            traces.append(trc)
        trace.snuffle(traces)

        return traces


class Tracer:
    def __init__(self, source, target, chopper, *args, **kwargs):
        self.runner = None
        self.source = source
        self.target = target
        self.config = kwargs.pop('config', None)
        self.traces = None
        self.processed_cache = {}
        self.chopper = chopper
        self.kwargs = kwargs

        self.config.receiver_distances = [source.distance_to(target)/1000.]
        self.config.receiver_azimuths = [source.azibazi_to(target)[0]]
        self.config.source_depth = source.depth/1000.
        self.config.id+= "_%s" % (self.source.id)

    def read_files(self, dir):
        self.traces = io.load(dir)

    def process(self, component, **pp_kwargs):
        tr = self.processed_cache.get(component, False)
        if not tr:
            tr_raw = self.filter_by_component(component).copy()
            tr = self.chopper.chop(
                self.source, self.target, tr_raw)
            self.processed_cache[component] = tr
        return self.post_process(tr, **pp_kwargs)

    def post_process(self, tr, normalize=False, response=False, noise=False):
        if normalize:
            tr.set_ydata(tr.ydata/num.max(num.abs(tr.ydata)))
        if response:
            tr1 = tr.copy()
            #tr = tr.transfer(transfer_function=response)
            tr = response.convolve(tr)
            #trace.snuffle([tr1, tr])
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #ax.plot(tr1.ydata)
            #ax.plot(tr.ydata)
            #plt.show()
        if noise:
            tr = noise.noisify(tr)

        return tr

    def filter_by_component(self, component):
        tr = filter(lambda t: t.nslc_id[3]==component, self.traces)[0]
        return tr

    def set_geometry(self, distances, azimuths, depths):
        self.config.receiver_distances = distances
        self.config.receiver_azimuths = azimuths
        self.config.source_depths = depths

    def set_config(self, config):
        self.config = config

    def snuffle(self):
        trace.snuffle(self.traces)

    def onset(self):
        return self.chopper.onset(self.source, self.target)

class QResponse(trace.FrequencyResponse):
    def __init__(self, Q, x, v):
        self.Q = Q
        self.x = x
        self.v = v

    def convolve(self, tr):
        new_tr = tr.copy()
        ydata = new_tr.get_ydata() 
        #freqs = num.fft.rfftfreq(len(ydata), d=new_tr.deltat)
        f, a = tr.spectrum()
        B = num.fft.rfft(self.evaluate(a, f))
        new_tr.set_ydata(num.fft.irfft(B))
        new_tr.snuffle()
        return new_tr

    def evaluate(self,A,  freqs):
        i = num.complex(0,1)
        e = num.e
        pi = num.pi
        return A* num.exp(-pi*freqs*self.x/self.v/self.Q)
        #return num.exp(e*pi*freqs*self.x/(self.Q*self.v)) * num.exp(e*i*2*pi*freqs*self.x/self.v)


class OnDemandEngine():
    def __init__(self, *args, **kwargs):
        self.stores = kwargs.pop('stores')

    def process(self, *args, **kwargs):
        targets = kwargs.get('targets', [])
        sources = kwargs.get('sources', [])
        wanted_ids = []
        try:
            store = filter(lambda x: x.config.store_id==store_id, self.stores)[0]
            # GIVE BACK THE RESPIONSE
            raise
        except:
            # IF NOT READY YET, GO ON CALCULATE:

            distances = {}
            depths = {}
            azimuths = {}

            for s in sources:
                wanted_ids.append(s.store_id)
                for t in targets:
                    if not s.store_id in distances.keys():
                        distances[s.store_id] = set()
                    if not s.store_id in azimuths.keys():
                        azimuths[s.store_id] = set()
                    if not s.store_id in depths.keys():
                        depths[s.store_id] = set()
                    distances[s.store_id].add(s.distance_to(t))

                    print 'seen from the other side???'
                    azimuths[s.store_id].add(s.azibazi_to(t)[0])
                    depths[s.store_id].add(s.depth)

            wanted_stores = []
            for store_id in wanted_ids:
                #store_id = store.config.store_id
                store = filter(lambda x: x.config.store_id==store_id, self.stores)[0]
                store.set_geometry(distances[store_id],
                                   azimuths[store_id],
                                   depths[store_id])
                wanted_stores.append(store)
            logging.debug("start QSeisRunner ")

            #p = multiprocessing.Pool()
            #p.map(self.stores, lambda x: x.run)
            for s in wanted_stores:
                s.run()
            logging.debug("done QSeisRunner ")

    def get_store(self, store_id):
        return filter(lambda x: x.config.store_id==store_id, self.stores)[0]


def create_store(superdir, store_dir, config, source, target, force=False, nworkers=4, extra=None):
    '''
    :param sources: list of max 2 sources which should be served
    :param target: target which should be served

    The stores config id will be created from the *config_id*, the source's id and the target's station code,
    all separated by underscores.
    '''
    from pyrocko.fomosto import qseis
    dists = []
    depths = []

    dists.append(source.distance_to(target))
    depths.append(source.depth)

    config.id ="_".join((config.id, source.id, target.codes[1]))
    config.distance_min = min(dists)
    config.distance_max = max(dists)
    config.distance_delta = config.distance_max-config.distance_min
    config.source_depth_min = min(depths)
    config.source_depth_max = max(depths)
    config.source_depth_delta = config.source_depth_max-config.source_depth_min

    Store.create_editables(store_dir, config=config, force=force, extra=extra)
    module = qseis
    #gf.Store(store_dir)
    module.build(store_dir, force=force, nworkers=nworkers)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", dest="debug",  help="debug")
    args = parser.parse_args()

    config = qseis.QSeisConfigFull.example()
    #config.receiver_distances = [1000., 2000.]
    #config.receiver_azimuths = [0., 0.]
    config.time_region = [meta.Timing(-10.), meta.Timing(30.)]
    config.time_window = 40.
    config.nsamples = config.time_window/0.5
    config.earthmodel_1d = config.earthmodel_1d.extract(0, 80000)
    config.store_id = 'test'
    s = OnDemandStore(config=config)
    e = OnDemandEngine(stores=[s])
    target = Target()
    source = DCSource(lat=target.lat+0.1)
    e.process(targets=[target],
              sources=[source])
    tstart = time.time()
    e.process()
    print "%s seconds"%(time.time()-tstart)

    if args.debug:
        import pdb 
        pdb.set_trace()
    print 'start'
    print config

    trace.snuffle(e.runner.get_traces())

