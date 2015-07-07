from pyrocko.fomosto import qseis
from pyrocko import trace
from pyrocko.gf import meta, Engine
from pyrocko.gf.store import Store
import time
import argparse
import logging
import multiprocessing


class OnDemandStore():
    def __init__(self, *args, **kwargs):
        self.runner = None
        self.store_id = kwargs.pop('store_id', None)
        self.config = kwargs.pop('config', None)

    def run(self):
        runner = qseis.QSeisRunner()
        import pdb 
        pdb.set_trace()
        runner.run(config=self.config)    
    def set_geometry(self, distances, azimuths, depths):
        self.config.receiver_distances = distances
        self.config.receiver_azimuths = azimuths
        self.config.source_depths = depths

    def set_config(self, config):
        self.config = config

    
class OnDemandEngine():
    def __init__(self, *args, **kwargs):
        self.on_demand_stores = kwargs.pop('on_demand_stores')
        #Engine.__init__(self, *args, **kwargs)

    def process(self, *args, **kwargs):
        targets = kwargs.get('targets', [])
        sources = kwargs.get('sources', [])

        distances = {}
        depths = {}
        azimuths = {}
        for s in sources:
            for t in targets:
                if not s.store_id in distances.keys():
                    distances[s.store_id] = []
                if not s.store_id in azimuths.keys():
                    azimuths[s.store_id] = []
                if not s.store_id in depths.keys():
                    depths[s.store_id] = []
                distances[s.store_id].append(s.distance_to(t))
                azimuths[s.store_id].append(s.azibazi_to(t)[0])
                depths[s.store_id].append(s.depth)
        print distances
        import pdb 
        pdb.set_trace()
        for store in self.on_demand_stores:
            store.set_geometry(distances[store.store_id],
                               azimuths[store.store_id],
                               depths[store.store_id])
        
        logging.debug("start QSeisRunner ")

        #p = multiprocessing.Pool()
        #p.map(self.on_demand_stores, lambda x: x.run)
        for s in self.on_demand_stores:
            s.run()
        logging.debug("done QSeisRunner ")

    def get_store(self, *arg, **kwargs):
        return None

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", dest="debug",  help="debug")
    args = parser.parse_args()
    
    config = qseis.QSeisConfigFull.example()
    config.receiver_distances = [10., 20.]
    config.receiver_azimuths = [0., 0.]
    config.time_region = [meta.Timing(-10.), meta.Timing(30.)]
    config.time_window = 40.
    config.nsamples = 40/0.5
    config.earthmodel_1d = config.earthmodel_1d.extract(0, 80000)
    s = OnDemandStore(config=config, store_id='test')
    print s
    s.set_geometry(distances=[10.], azimuths=[0.], depths=[1])
    s.run()

    e = OnDemandEngine(on_demand_stores=[s])
    tstart = time.time()
    e.process()
    print "%s seconds"%(time.time()-tstart)

    if args.debug:
        import pdb 
        pdb.set_trace()
    print 'start'
    print config

    trace.snuffle(e.runner.get_traces())

