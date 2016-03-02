import numpy as num
import os
import tempfile
import glob
from collections import defaultdict
from pyrocko.fomosto import qseis
from pyrocko import trace
from pyrocko import guts
from pyrocko import io
from pyrocko import util
from pyrocko.gf import meta, Engine, Target, DCSource, STF, Filter
from pyrocko.gf.store import Store
from pyrocko.parimap import parimap
from pyrocko.guts import Float
from autogain.autogain import PhasePie, PickPie
import time
import argparse
import logging
import multiprocessing
import progressbar
from rupture_size import radius as source_radius


pjoin = os.path.join

logger = logging.getLogger()
diff_response = trace.DifferentiationResponse()


class EvalrespFilter(Filter):
    response = trace.Evalresp.T()
    #def __init__(self, response):
    #    Filter.__init__(self)


def associate_responses(directory, targets, time=0):
    '''
    :param instant: time for which response to be evaluated
    '''
    fns = glob.glob(directory)
    if len(fns)==0:
        logger.warn('no files found: %s',  directory)
    for fn in fns:
        codes = tuple(fn.split('/')[-1].split('.')[1:])
        logger.warn('Frage an Sebastian: Was soll das target, bzw. wie herum benutzen?')
        response = trace.Evalresp(respfile=fn,
                                    nslc_id=codes,
                                    target='dis',
                                    time=time)
        for t in targets:
            if t.codes==codes:
                t.filter = EvalrespFilter(response=response)
                logger.info('resp-file: %s <---> target: %s' %(codes, t.codes))
                break
        else:
            logger.info('resp-file: %s unassociated' % str(codes))
    
    for t in targets:
        if not t.filter:
            logger.warn('Target %s does not carry response info' % str(t.codes))


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


def WEBNETMl2M0(Ml):
    # for WEBNET events 2000 swarm
    # Horalek, Sileny 2015
    return 10**(1.12*Ml)


#class Brune(trace.FrequencyResponse):
#    '''
#    Brunes source model type
#
#    as in Lion Krischer's moment_magnitude_calculator:
#    https://github.com/krischer/moment_magnitude_calculator
#    '''
#    duration = Float.T()
#    #sampling_rate = Float.T()
#    variation_signal = Float.T()
#    stress_drop = Float.T(default=2.9E6, optional=True, help='')
#    shear_module = Float.T()
#    v_s = Float.T()
#    depth = Float.T()
#    distance = Float.T()
#
#
#    def evaluate(self, freqs):
#        mu = self.vs**2 * self.rho
#        # freqs in rad/s ????
#        b = num.zeros(len(freqs))
#        b[:] = self.b
#        return self.stressdrop*self.vs/ mu / (freqs**2 + b**2)
#
#    @property
#    def b(self):
#        print self.magnitude, self.a
#        return 2.33*self.vs / self.a
#
#    @property
#    def a(self):
#        return source_radius([self.magnitude])
#
#    def discretize_t(self, deltat, tref):
#        t = num.linspace(0, self.duration, self.duration * 1./deltat)
#        return t, 2.0 * self.variation_signal * self.stress_drop / self.shear_module * self.v_s * \
#                    self.distance / self.depth * t * num.exp(-2.34 * (self.v_s / self.distance) * t)


class Builder:
    def __init__(self, cache_dir=False):
        self.runners = []
        self.cache_dir = cache_dir
        self.configfn='qseis-config.yaml'
        if self.cache_dir:
            self.subdirs = os.listdir(self.cache_dir)
            self.config_str = self.load_configs()
        else:
            self.subdirs = None

    def build(self, tracers, engine=None, snuffle=False):
        ready = []
        need_work = []
        logger.info('prepare or load tracers....')
        pb = progressbar.ProgressBar(maxval=len(tracers)).start()
        for i_tr, tr in enumerate(tracers):
            if tr.setup_data():
                ready.append(tr)
            elif tr.setup_from_engine(engine):
                ready.append(tr)
            elif self.cache(tr, load=True):
                ready.append(tr)
            else:
                need_work.append(tr)
            pb.update(i_tr)
        pb.finish()
        logger.info('run %s tracers' % len(need_work))
        for tr, runner in parimap(self.work, need_work):
            tmpdir = runner.tempdir
            traces = self.load_seis(tmpdir, tr.config)
            tr.traces = traces
            ready.append(tr)
            if self.cache_dir:
                self.cache(tr, load=False)
            del(runner)

        if snuffle:
            traces = []
            for tracer in ready:
                dist, z = tracer.get_geometry()
                trs = [t.copy() for t in tracer.traces if t ]
                map(lambda x: x.set_codes(network='%2i-%2i'%(dist, z)), trs)
                traces.extend(trs)
            trace.snuffle(traces)

        return ready

    def work(self, tr):
        runner = qseis.QSeisRunner(keep_tmp=True)
        runner.run(config=tr.config)
        return tr, runner

    def load_configs(self):
        config_str = {}
        for sdir in self.subdirs:
            config_str[str(guts.load(filename=pjoin(self.cache_dir, sdir,
                                                    self.configfn)))] = sdir
        return config_str

    def cache(self, tr, load):
        fn = 'traces.mseed'
        found_cache = False
        if self.cache_dir and load and str(tr.config) in self.config_str.keys():
            file_path = pjoin(self.cache_dir, self.config_str[str(tr.config)], fn)
            tr.read_files(file_path)
            found_cache = True
        elif self.cache_dir and not load:
            tmpdir = tempfile.mkdtemp(dir=self.cache_dir)
            tr.config.dump(filename=pjoin(tmpdir, self.configfn))
            io.save(tr.traces, pjoin(tmpdir, fn))
            logger.info('cached under: %s' % self.cache_dir)
        return found_cache

    def load_seis(self, directory, config):
        # Hier mussen die files seis.tr, seis.... etc. aus der qseis temp directory 
        # geladen werden und wieder in den tracer gefuettert werden.
        fns = glob.glob(pjoin(directory, 'seis.*'))
        traces = []
        tmin = config.time_start
        vred = config.time_reduction_velocity
        distances = config.receiver_distances
        assert len(distances)==1
        if vred != 0.:
            tmin += distances[0]/vred

        for fn in fns:
            data = num.loadtxt(fn, skiprows=1, dtype=num.float)
            nsamples, ntraces = data.shape
            deltat = (data[-1, 0] - data[0, 0])/(nsamples-1)

            trc = trace.Trace(
                ydata=data[:, 1], tmin=tmin+deltat, deltat=deltat, channel=fn.split('.')[-1][-1])
            traces.append(trc)

        return traces

def _do_nothing(x):
    return x

class Tracer:
    def __init__(self, source, target, chopper, channel='v', *args, **kwargs):
        ''':param channel: qseis channel'''
        self.runner = None
        self.source = source
        self.target = target
        self.tinc = kwargs.pop('tinc', None)
        self.config = kwargs.pop('config', None)
        self.traces = None
        self.processed = False
        self.chopper = chopper
        self.kwargs = kwargs
        self.channel = channel
        self.want = kwargs.pop('want', 'displacement')
        self.fmin = kwargs.pop('fmin', -99999.)
        self.fmax = kwargs.pop('fmax', 99999.)

        if self.want == 'velocity':
            self._apply_transfer = self.differentiate
        elif self.want == 'displacement':
            self._apply_transfer = _do_nothing
        else:
            raise Exception('unknown wanted quantity: %s' % self.want)

        if not self.target.store_id and not self.kwargs.pop('no_config', False):
            self.config.receiver_distances = [source.distance_to(target)/1000.]
            self.config.receiver_azimuths = [source.azibazi_to(target)[0]]
            self.config.source_depth = source.depth/1000.
            self.config.id+= "_%s" % (self.source.id)
            #self.config.id = ""
            self.config.regularize()

    def read_files(self, dir):
        self.traces = io.load(dir)

    def setup_data(self):
        return False

    def differentiate(self, tr):
        t = tr.tmax - tr.tmin
        sr = 0.5/tr.deltat
        return tr.transfer(transfer_function=diff_response, freqlimits=(1./t, 1.1/t, sr*0.7, sr*0.9), tfade=t*0.15)

    def simulate(self, tr):
        return tr.transfer(transfer_function=self.target.filter.response)

    def setup_from_engine(self, engine=None):
        if engine and self.target.store_id:
            response = engine.process(sources=[self.source], targets=[self.target])
            traces = response.pyrocko_traces()
            processed = []
            for tr in traces:
                if self.source.brunes:
                    self.source.brunes.preset(source=self.source, target=self.target)
                    tr = tr.transfer(transfer_function=self.source.brunes)
                tr = self.simulate(tr)
                processed.append(tr)
            self.traces = processed

            #self.traces = rotate_rtz(self.traces)
            self.config = engine.get_store_config(self.target.store_id)
            return True
        return

    def process(self, **pp_kwargs):
        tr = self.processed
        if not tr:
            tr_raw = self.filter_by_channel(self.channel).copy()
            tr = self.chopper.chop(self.source, self.target, tr_raw)
            tr = self._apply_transfer(tr)
        self.processed = tr
        return self.post_process(tr, **pp_kwargs)

    def post_process(self, tr, normalize=False, response=False, noise=False):
        if normalize:
            tr.set_ydata(tr.ydata/num.max(num.abs(tr.ydata)))
        #if response:
        #    tr = tr.transfer(transfer_function=response)
        if noise:
            tr = noise.noisify(tr)

        return tr

    def filter_by_channel(self, channel):
        tr = filter(lambda t: t.nslc_id[3]==channel, self.traces)[0]
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

    def arrival(self):
        return self.chopper.arrival(self.source, self.target)

    def get_geometry(self):
        return self.source.distance_to(self.target), self.source.depth

    def label(self):
        return self.config.id

class DataTracer(Tracer):
    def __init__(self, **kwargs):
        self.data_pile = kwargs.pop('data_pile')
        kwargs.update({'no_config': True})
        Tracer.__init__(self, **kwargs)

    def setup_data(self):
        self.process()
        return True

    def process(self, **pp_kwargs):
        tr = self.processed
        if not tr:
            tr = self.chopper.chop_pile(self.data_pile, self.source, self.target)
        self.traces = [tr]
        self.processed = tr
        return self.post_process(tr, **pp_kwargs)

    def label(self):
        return '%s/%s' % (util.time_to_str(self.source.time), ".".join(self.target.codes))

class Chopper():
    def __init__(self, startphasestr=None, endphasestr=None, fixed_length=None,
                 by_magnitude=None, phase_position=0.5, xfade=0.0, phaser=None):
        assert None in [fixed_length, by_magnitude]
        self.by_magnitude = by_magnitude
        self.phase_pie = phaser or PhasePie()
        self.startphasestr = startphasestr
        self.endphasestr = endphasestr
        self.phase_position = phase_position
        self.fixed_length = fixed_length
        self.chopped = DDContainer()
        self.xfade = xfade

    def chop(self, s, t, tr, debug=False):
        dist = s.distance_to(t)
        depth = s.depth
        tstart = self.phase_pie.t(self.startphasestr, (depth, dist))
        if self.endphasestr!=None:
            tend = self.phase_pie.t(self.endphasestr, (depth, dist))
        elif self.fixed_length!=None:
            tend = self.fixed_length+tstart
        elif self.by_magnitude!=None:
            tend = tstart+self.by_magnitude(s.magnitude)
        else:
            raise Exception('Need to define exactly one of endphasestr and fixed length')
        tr_bkp = tr
        tr = tr.copy()
        tr.set_location('cp')
        tstart, tend = self.setup_time_window(tstart, tend)
        tr.chop(tstart, tend)
        self.chopped.add(s, t, tr)
        return tr

    def chop_pile(self, data_pile, source, target):
        tstart = self.phase_pie.t(self.startphasestr, (source, target))
        if not tstart:
            return tstart
        if self.by_magnitude!=None:
            tend = tstart+self.by_magnitude(source.magnitude)
        else:
            tend = self.fixed_length + tstart
        tstart, tend = self.setup_time_window(tstart, tend)
        select = lambda x: x.nslc_id == target.codes

        # A little dangerous:
        try:
            tr = data_pile.chop(tmin=tstart, tmax=tend, trace_selector=select)[0][0]
        except IndexError as e:
            tr = None
        self.chopped.add(source, target, tr) 
        return tr

    def setup_time_window(self, tstart, tend):
        trange = tend-tstart
        tstart -= trange*self.phase_position
        tend -= trange*self.phase_position
        return tstart, tend

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

    def set_tfade(self, tfade):
        '''only needed for method fft'''
        self.tfade = tfade

    def onset(self, *args, **kwargs):
        return self.arrival(*args, **kwargs).t

    def arrival(self, s, t):
        try:
            return self.phase_pie.arrival(self.startphasestr, (s.depth, s.distance_to(t)))
        except AttributeError:
            return self.phase_pie.arrival(self.startphasestr, (s, t))


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

