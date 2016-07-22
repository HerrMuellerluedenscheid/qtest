import numpy as num
import os
import tempfile
import time
import argparse
import logging
import progressbar
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from pyrocko.fomosto import qseis
from pyrocko import trace, guts, io, util, pz, orthodrome
from pyrocko.ahfullgreen import add_seismogram, Impulse
from pyrocko.gf import meta, Target, DCSource, Filter
from pyrocko.gf import RectangularSource, OutOfBounds
from pyrocko.gf.store import Store
from pyrocko.parimap import parimap
from autogain.autogain import PhasePie

pjoin = os.path.join

logger = logging.getLogger()
diff_response = trace.DifferentiationResponse()
tr_meta_init = {'noisified': False}

class ResponseFilter(Filter):
    response = trace.FrequencyResponse.T()


class TTPerturbation():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def perturb(self, t):
        return t + num.random.normal(self.mu, self.sigma)

    def plot(self):
        fig = plt.figure(figsize=(4, 5))
        ax = fig.add_subplot(111)
        ts = [self.perturb(0) for i in xrange(1000)]
        ax.hist(ts, bins=20)
        ax.set_title('Travel Time Perturbation')


class UniformTTPerturbation(TTPerturbation):
    def __init__(self, mu, sigma):
        '''mu: center
        sigma: width'''
        TTPerturbation.__init__(self, mu, sigma)
        self.low = self.mu-0.5*self.sigma
        self.high = self.mu+0.5*self.sigma

    def perturb(self, t):
        return t + num.random.uniform(self.low, self.high)


def associate_responses(fns, targets, time=0, type='evalresp'):
    '''
    :param instant: time for which response to be evaluated
    '''
    if len(fns) == 0:
        logger.warn('no files found')
    for fn in fns:
        if type == 'evalresp':
            codes = tuple(fn.split('/')[-1].split('.')[1:])
            response = trace.Evalresp(
                respfile=fn, nslc_id=codes, target='vel', time=time)

        if type == 'polezero':
            zeros, poles, constant = pz.read_sac_zpk(fn)
            sc = fn.split('/')[-1].split('.')[:2]
            codes = ('CZ', sc[0], '', sc[1])
            #codes = tuple(('CZ', ) + tuple(sc))
            #poles = [p/(2*num.pi) for p in poles]
            response = trace.PoleZeroResponse(poles=poles, zeros=zeros, constant=num.complex(constant))
        for t in targets:
            if t.codes == codes:
                t.filter = ResponseFilter(response=response)
                logger.info('resp-file: %s <---> target: %s' %(codes, t.codes))
                break
        else:
            logger.info('resp-file: %s unassociated' % str(codes))

    for t in targets:
        if not t.filter:
            logger.warn('Target %s does not carry response info' % str(t.codes))



class Noise():
    def __init__(self, files, scale=1.):
        traces = io.load(files)
        self.scale = scale
        self.noise = self.dictify_noise(traces)

    def dictify_noise(self, traces):
        noise = {}
        for tr in traces:
            if not tr.nslc_id in noise:
                noise[tr.nslc_id] = tr.get_ydata()
            else:
                logger.warn('More than one noisy traces for %s' % ('.'.join(tr.nslc_id)))
        return noise

    def noisify(self, tr):
        if tr.nslc_id not in self.noise.keys():
            raise Exception('No Noise for tr %s' % ('.'.join(tr.nslc_id)))
        n = self.extract_noise(tr) * self.scale
        tr.ydata += n
        return tr

    def extract_noise(self, tr):
        n_want = len(tr.ydata)
        i_start = num.random.choice(range(len(self.noise[tr.nslc_id])-n_want))
        return self.noise[tr.nslc_id][i_start:i_start+n_want]



class Noise:
    def __init__(self, level):
        self.level = level

    def noisify(self, tr, inplace=True):
        if inplace is False:
            tr = tr.copy()
        if tr.meta['noisified'] == True:
            raise Exception('Trace was already noisified')
        tr.meta['noisified'] = True
        return self.add_noise(tr, self.level)

    def add_noise(self, *args, **kwargs):
        # to be implemented in subclass
        pass


class RandomNoise(Noise):
    def __init__(self, *args, **kwargs):
        Noise.__init__(self, *args, **kwargs)

    def add_noise(self, t, level):
        ydata = t.get_ydata()
        noise = num.random.random(len(ydata))-0.5
        noise *= ((num.max(num.abs(ydata))) * level)
        ydata += noise
        t.set_ydata(ydata)
        return t


class RandomNoiseConstantLevel(Noise):
    def __init__(self, *args, **kwargs):
        Noise.__init__(self, *args, **kwargs)

    def add_noise(self, t, level):
        ydata = t.get_ydata()
        noise = (num.random.random(len(ydata))-0.5) * level
        ydata += noise
        t.set_ydata(ydata)
        return t


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
            if tr.setup_data(engine=engine):
                ready.append(tr)
            #elif self.cache(tr, load=True):
            #    ready.append(tr)
            #else:
            #    need_work.append(tr)
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
        self.back_azimuth = kwargs.get('back_azimuth', None)
        self.incidence_angle = kwargs.get('incidence_angle', None)
        self.tinc = kwargs.pop('tinc', None)
        self.config = kwargs.pop('config', None)
        self.trace = None
        self.processed = False
        self.chopper = chopper
        self.kwargs = kwargs
        self.channel = channel
        self.want_trace_length = None
        self.perturbation = kwargs.pop('perturbation', 0.)
        self.want = kwargs.pop('want', 'displacement')
        self.fmin = kwargs.pop('fmin', -99999.)
        self.fmax = kwargs.pop('fmax', 99999.)

        self._apply_transfer = self.prepare_quantity(self.want)

    def prepare_quantity(self, want):
        if want == 'velocity':
            return self.differentiate
        elif self.want == 'displacement':
            return _do_nothing
        else:
            raise Exception('unknown wanted quantity: %s' % self.want)

    def read_files(self, dir):
        trs = io.load(dir)
        if len(trs) > 1:
            self.trace = trs
            logger.warn("new rule: one traces-one source-one target-one trace!")
        else:
            self.trace = trs[0]

    def setup_data(self, *args, **kwargs):
        engine = kwargs.get('engine', None)
        self.setup_from_engine(engine)
        if self.trace is not None:
            #self.process()
            return self.trace
        else:
            return False

    def process(self, **pp_kwargs):

        if isinstance(self.processed, trace.Trace):
            # done already:
            return self.processed

        elif isinstance(self.processed, str):
            # something went wrong before
            return self.processed

        else:
            tr = self.trace.copy()
            self.processed = self.chopper.chop(self.source, self.target, tr,
                                              inplace=True)
            if self.processed!="NoData":
                self.processed = self.post_process(self.processed, **pp_kwargs)

            return self.processed

    def differentiate(self, tr):
        t = tr.tmax - tr.tmin
        sr = 0.5 / tr.deltat
        return tr.transfer(transfer_function=diff_response,
                           freqlimits=(1./t, 1.1/t, sr*1.1, sr*1.5),
                           tfade=t*0.15)

    def simulate(self, tr):
        return tr.transfer(transfer_function=self.target.filter.response)

    def setup_from_engine(self, engine):
        try:
            response = engine.process(sources=[self.source], targets=[self.target])
        except OutOfBounds as e:
            self.trace = "OutOfBounds"
            self.processed = self.trace
            return True
        self.trace = response.pyrocko_traces()[0]
        self.trace.meta = tr_meta_init
        if not isinstance(self.source, RectangularSource): #and not isinstance(self.source, CircularSource):
        #if not isinstance(self.source, CircularSource):
            if self.source.brunes:
                self.source.brunes.preset(source=self.source, target=self.target)
                self.trace = self.trace.transfer(transfer_function=self.source.brunes)
        self.trace = self._apply_transfer(self.trace)
        if self.target.filter:
            self.trace = self.simulate(self.trace)
        #self.traces = rotate_rtz(self.traces)
        self.config = engine.get_store_config(self.target.store_id)
        return True

    def post_process(self, tr, normalize=False, response=False, noise=False):
        if normalize:
            tr.set_ydata(tr.ydata/num.max(num.abs(tr.ydata)))
        if noise:
            tr = noise.noisify(tr)
        trace_length = tr.tmax-tr.tmin
        if self.want_trace_length is not None and trace_length!=self.want_trace_length:
            diff = trace_length - self.want_trace_length
            tr.extend(tmin=tr.tmin-diff, tr=tr.tmax)
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
        return self.chopper.onset(self.source, self.target) + self.perturbation

    def arrival(self):
        return self.chopper.arrival(self.source, self.target) + self.perturbation

    def get_geometry(self):
        return self.source.distance_to(self.target), self.source.depth

    def label(self):
        if self.config is not None:
            l = self.config.id
        else:
            l = "%s-%s" % (self.source.time, ".".join(self.target.codes))
        return l


class QSeisTraces(Tracer):
    def __init__(self, *args, **kwargs):
        Tracer.__init__(self, *args, **kwargs)

        if self.target.store_id is not None and not self.kwargs.pop('no_config', False):
            self.config.receiver_distances = [source.distance_to(target)/1000.]
            self.config.receiver_azimuths = [source.azibazi_to(target)[0]]
            self.config.source_depth = source.depth/1000.
            self.config.id+= "_%s" % (self.source.id)
            self.config.regularize()



class AhfullgreenTracer(Tracer):
    def __init__(self, material, deltat, *args, **kwargs):
        Tracer.__init__(self, *args, **kwargs)
        self.material = material
        self.deltat = deltat
        self.f = (0.0, 0.0, 0.0)

    def setup_data(self, *args, **kwargs):
        m = self.material
        m6 = self.source.pyrocko_moment_tensor().m6()
        n, e = orthodrome.latlon_to_ne_numpy(self.source.effective_lat,
                                             self.source.effective_lon,
                                             self.target.effective_lat,
                                             self.target.effective_lon)
        n = n[0]
        e = e[0]
        d = self.source.depth
        xyz = (n, e, d)
        ns = num.sqrt(n**2 + e**2 + d**2) / m.vs / self.deltat * 2.
        ns = int(ns/2)*2
        out_x = num.zeros(ns)
        out_y = num.zeros(ns)
        out_z = num.zeros(ns)
        add_seismogram(m.vp, m.vs, m.rho, m.qp, m.qs, xyz, self.f, m6,
                       self.target.quantity, self.deltat, 0.0,
                       out_x, out_y, out_z, Impulse())

        self.trace = trace.Trace(
            channel='Z', tmin=self.source.time, deltat=self.deltat,
            ydata=out_z, meta=tr_meta_init)

        return self.trace

    def process(self, **kwargs):
        if self.processed is not "NoData" or self.processed is False:
            self.processed = self.chopper.chop(
                self.source, self.target, self.trace)
            if self.processed != "NoData":
                self.processed = self.post_process(self.processed, **kwargs)

        return self.processed

    def __str__(self):
        s = "trace: %s \nprocessed: %s" % (self.trace, self.processed)
        return s

class DataTracer(Tracer):
    def __init__(self, incidence_angle=0., data_pile=None,
                 rotate_channels=False, **kwargs):
        kwargs.update({'no_config': True})
        Tracer.__init__(self, **kwargs)
        self.data_pile = data_pile
        self.rotate_channels = rotate_channels
        self.incidence_angle = incidence_angle
        self.back_azimuth = self.target.azibazi_to(self.source)[1]

    def setup_data(self, *args, **kwargs):
        if self.rotate_channels:
            trs = self.chopper.chop_pile(
                self.data_pile, self.source, self.target, all_channels=True)
            if trs is None:
                tr = False
            else:
                trs = trace.rotate_to_lqt(
                    trs, self.back_azimuth, self.incidence_angle,
                    in_channels=self.rotate_channels['in_channels'],
                    out_channels=self.rotate_channels['out_channels'])
                tr = filter(tr.channel==self.channel, trs)
        else:
            tr = self.chopper.chop_pile(
                self.data_pile, self.source, self.target, all_channels=False)
            if tr is None:
                tr = False
            else:
                tr.meta = tr_meta_init
        self.trace = tr
        self.processed = self.trace
        return self.trace

    def process(self, **kwargs):
        return self.processed

    def label(self):
        return '%s/%s' % (util.time_to_str(self.source.time), ".".join(self.target.codes))

    def drop_data(self):
        self.trace = None
        self.processed = None


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
        self.xfade = xfade

    def chop(self, s, t, tr, offset=0., inplace=False):
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

        if inplace is False:
            tr = tr.copy()
        tstart += (offset + s.time)
        tend += (offset + s.time)
        tstart, tend = self.setup_time_window(tstart, tend)
        try:
            tr.chop(tstart, tend)
        except trace.NoData as e:
            logger.warn("chopping trace: %s failed:  %s" % (tr, "NoData"))
            return "NoData"
        return tr

    def chop_pile(self, data_pile, source, target, all_channels=False):
        tstart = self.phase_pie.t(self.startphasestr, (source, target))
        if not tstart:
            return tstart
        tstart += source.time
        if self.by_magnitude!=None:
            tend = tstart+self.by_magnitude(source.magnitude)
        else:
            tend = self.fixed_length + tstart
        tstart, tend = self.setup_time_window(tstart, tend)
        if all_channels:
            select = lambda x: x.nslc_id == target.codes
        else:
            select = lambda x: x.nslc_id[:3] == target.codes[:3]
        # A little dangerous:
        try:
            tr = data_pile.chop(tmin=tstart, tmax=tend, trace_selector=select)[0][0]
        except IndexError as e:
            tr = None
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

