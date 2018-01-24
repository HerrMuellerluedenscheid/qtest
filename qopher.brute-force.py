import concurrent.futures
import multiprocessing
import itertools
from qtest.config import QConfig
from qtest.qopher import run_qopher
import copy


base_config = QConfig.load(filename='base_config.yaml')

test_grid = {
    'ntapers': [3, 5],
    'time_bandwidth': [2, 4, 6],
    'position': [0.9, 0.7, 0.5],
    'window_length': [0.1, 0.2, 0.3],
    'traversing_distance_min': [200.],
    'traversing_ratio': [1., 2., 4.],
}


keys, values = zip(*test_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
configs = []
print('total: %s experiments' % len(experiments))
for iexperi, e in enumerate(experiments):
    config = copy.deepcopy(base_config)
    
    for k, v in e.items():
        setattr(config, k, v)

    config.outdir = str(iexperi) + '-' + '_'.join(e[:3] for e in e.keys())
    configs.append(config) 


#with concurrent.futures.ProcessPoolExecutor() as executor:
#    executor.map(run_qopher, configs)

pool = multiprocessing.Pool()
pool.map(run_qopher, configs)
