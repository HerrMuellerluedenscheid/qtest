import numpy as num
import os
import tempfile
from autogain.autogain import PhasePie
from pyrocko.fomosto import qseis
from pyrocko import trace
from pyrocko import guts
from pyrocko import io
from pyrocko.gf import meta, Engine, Target, DCSource
from pyrocko.gf.store import Store
import time
import argparse
import logging
import multiprocessing

pjoin = os.path.join

logger = logging.getLogger()


class Tracer():
    def __init__(self, source, target, *args, **kwargs):
        self.runner = None
        self.source = source
        self.target = target
        self.config = kwargs.pop('config', None)
        self.phases = PhasePie(mod=self.config.earthmodel_1d)
        self.traces = None
        self.processed = None
        self.kwargs = kwargs

        self.config.receiver_distances = [source.distance_to(target)/1000.]
        self.config.receiver_azimuths = [source.azibazi_to(target)[0]]
        self.config.source_depth = source.depth/1000.
        self.config.id+= "_%s" % (self.source.id)

    def run(self, cache_dir=False):
        configfn='qseis-config.yaml'
        fn = 'traces.mseed'
        if cache_dir:
            subdirs = os.listdir(cache_dir)
            for sdir in subdirs:
                config = guts.load(filename=pjoin(cache_dir, sdir, configfn))
                self.config.regularize()
                if str(config)==str(self.config):
                    self.traces = io.load(pjoin(cache_dir, sdir, fn))
                    return

        runner = qseis.QSeisRunner()
        runner.run(config=self.config)
        self.traces = runner.get_traces()


        if not self.traces:
            logger.warn('no traces returned')
        if cache_dir and self.traces:
            tmpdir = tempfile.mkdtemp(dir=cache_dir)
            self.config.dump(filename=pjoin(tmpdir, configfn))
            io.save(self.traces, pjoin(tmpdir, fn))
            logger.info('cached under: %s' % cache_dir)

    def set_geometry(self, distances, azimuths, depths):
        self.config.receiver_distances = distances
        self.config.receiver_azimuths = azimuths
        self.config.source_depths = depths

    def set_config(self, config):
        self.config = config


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
    print 'done'
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

