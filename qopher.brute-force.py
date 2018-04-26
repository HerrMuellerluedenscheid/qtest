import concurrent.futures
import multiprocessing
import itertools
import os
import copy
import sys
import argparse
import numpy as num
from pyrocko.gf.store import remake_dir
from qtest.config import QConfig
from qtest.qopher import run_qopher
from pyrocko.guts import Int, List, Object, Float


class TestGrid(Object):
    ntapers = List.T(Int.T())
    time_bandwidth = List.T(Float.T())
    position = List.T(Float.T())
    fmax_lim = List.T(Float.T())
    fmin_lim = List.T(Float.T())
    min_bandwidth = List.T(Float.T())
    cc_min = List.T(Float.T())
    snr = List.T(Float.T())
    window_length = List.T(Float.T())
    traversing_distance_min = List.T(Float.T())
    traversing_ratio = List.T(Float.T())
    number_of_tests = Int.T(default=2000)
    test_seed = Int.T(default=0, optional=True)

    ignore = [
        'number_of_tests',
        'test_seed'
    ]
    
    def setup(self):
        vals = {}
        num.random.seed(self.test_seed)
        for k, v in self.__dict__.items():
            if k in self.ignore:
                continue
            elif len(v) == 1:
                vals[k] = [v[0]] * self.number_of_tests
            else:
                vals[k] = num.random.uniform(v[0], v[1], self.number_of_tests)

        return vals

    def iter_tests(self):
        vals = self.setup()
        keys = list(vals.keys())
        for i in range(self.number_of_tests):
            v = [vals[k][i] for k in keys]
            yield dict(zip(keys, v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-config', default='base_config.yaml')
    parser.add_argument('--grid', default=None, type=str)
    parser.add_argument('--outdir')
    parser.add_argument('--run-first', action='store_true', default=False)
    parser.add_argument('--show-only', action='store_true', default=False,
                        help='show output names and how config will be configured')
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--continue', dest='cont', action='store_true', default=False)
    parser.add_argument('--nthreads', type=int, default=4)

    args = parser.parse_args()
    base_config = QConfig.load(filename=args.base_config)

    if args.grid:
        test_grid = TestGrid.load(filename=args.grid)
    else:
        test_grid = {
            # 'ntapers': [3, 5],
            # 'time_bandwidth': [4, 6],
            'position': [0.9, 0.7],
            'fmax_lim': [60., 80.],
            'fmin_lim': [30., 50.],
            'window_length': [0.15, 0.25],
            'traversing_distance_min': [1000., 2000.],
            'traversing_ratio': [3., 5],
            # 'number_of_test': 2000,
        }

        test_grid = TestGrid(**test_grid)
        test_grid.dump(filename='test-example.grid')
        print('generated example test grid')
        sys.exit()

    if args.force:
        input("\n Sure, you want to force and overwrite? Enter to continue...")

    outdir = args.outdir
    if not args.show_only:
        if not args.cont:
            remake_dir(outdir, force=args.force)

    keys, values = zip(*test_grid.__dict__.items())

    # experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    configs = []
    for iexperi, test_values in enumerate(test_grid.iter_tests()):
        config = copy.deepcopy(base_config)
        
        info_str = '|'

        for keys, vals in test_values.items():
            setattr(config, keys, vals)
            info_str += ' %1.1f |' % vals

        keys = sorted(test_values.keys())
        config.outdir = os.path.join(outdir, str(iexperi) + '-' + '_'.join(e[:1] for e in
                                                                    keys))
        if args.cont and os.path.exists(config.outdir):
            print('Exists. continue.')
            continue

        print("filename: %s   | %s" % (config.outdir, info_str))
        config.regularize()
        configs.append(config) 

    n_run = len(configs)
    if args.show_only:
        sys.exit(1)

    elif args.run_first:
        run_qopher(configs[0])
        sys.exit(0)
    else:
        pool = multiprocessing.Pool(args.nthreads)
        pool.map(run_qopher, configs)
