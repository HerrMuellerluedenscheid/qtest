import concurrent.futures
import multiprocessing
import itertools
import os
import copy
import sys
import argparse
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
    snr = List.T(Float.T())
    window_length = List.T(Float.T())
    traversing_distance_min = List.T(Float.T())
    traversing_ratio = List.T(Float.T())

    def __getitem__(self, k):
        return getattr(self, k)


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
            'ntapers': [3, 5],
            'time_bandwidth': [4, 6],
            'position': [0.9, 0.7],
            'fmax_lim': [60., 70., 80.],
            'fmin_lim': [30., 40., 50.],
            'window_length': [0.15, 0.2, 0.25],
            'traversing_distance_min': [400., 600.],
            'traversing_ratio': [3., 4., 5],
        }


        test_grid = TestGrid(**test_grid)
        test_grid.dump(filename='test.grid')
        sys.exit()

    if args.force:
        input("\n Sure, you want to force and overwrite? Enter to continue...")

    outdir = args.outdir
    if not args.show_only:
        if not args.cont:
            remake_dir(outdir, force=args.force)

    keys, values = zip(*test_grid.__dict__.items())
    # keys, values = zip(*test_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    configs = []
    print('total: %s experiments' % len(experiments))
    for iexperi, e in enumerate(experiments):
        config = copy.deepcopy(base_config)
        
        info_str = '|'
        for k, v in e.items():
            info_str += ' %1.1f |' % v
            setattr(config, k, v)
        keys = sorted(list(e.keys()))
        config.outdir = os.path.join(outdir, str(iexperi) + '-' + '_'.join(e[:3] for e in
                                                                    keys))
        if args.cont and os.path.exists(config.outdir):
            print('Exists. continue.')
            continue
        print("filename: %s" % config.outdir, info_str)

        configs.append(config) 
    n_run = len(configs)
    if args.show_only:
        sys.exit(1)

    elif args.run_first:
        print('XXX')
        run_qopher(configs[0])
        sys.exit(0)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(run_qopher, configs)
    else:
        pool = multiprocessing.Pool(args.nthreads)
        pool.map(run_qopher, configs)
