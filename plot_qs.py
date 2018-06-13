#! /usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

from qtest.config import QConfig
import os
import numpy as num
import scipy.stats as stats
import sys
import argparse
import operator as opr


operator_dict = {
    '>': 'gt',
    '<': 'lt',
    '>=': 'ge',
    '==': 'eq',
    '<=': 'le',
}


def read_dirs(args):
    slopes = []
    for result_dir in args.directories:
        fn_config = os.path.join(result_dir, 'config.yaml')
        f = os.path.join(result_dir, args.result_filename)
        try:
            d = num.loadtxt(f, ndmin=1)
        except (ValueError, IOError) as e:
            print('failed to read fn: %s' % f)
            continue

        if len(d) == 0:
            print('File is empty %s ' % f)
            continue

        elif args.nmin and len(d) < args.nmin:
            print('Too few datapoints: %s' % f)
            continue

        slopes.append((f, fn_config, d))
    return slopes


def prefilter(slopes, args):
    if args.filter_column:
        col, expr = args.filter_column.split(':')
        compare, val = expr.split()
        val = float(val)

        def config_filter(conf):
            cmp_ = getattr(opr, operator_dict[compare])
            return cmp_(getattr(conf, col), val)
    else:
        def config_filter(conf):
            return True

    _slopes = []
    for islope, (f, fn_config, sl) in enumerate(slopes):
        conf = QConfig.load(filename=fn_config)
        if not config_filter(conf):
            continue
        else:
            _slopes.append((f, fn_config, sl))

    return _slopes


def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError) as e:
        return False


def cleanup_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)


def plothist(slopes, recip=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
    if recip:
        slopes = 1./slopes
        idx = num.where(num.abs(slopes) < 200)
    else:
        idx = num.arange(len(slopes))
    ax.hist(slopes[idx], bins=201, normed=False)
    ax.axvline(num.median(slopes), label='median', color='grey')
    ax.text(num.median(slopes), 0., 'median: %1.4f' % num.median(slopes),
            color='grey')
    ax.set_xlabel('1/Q')
    ax.set_ylabel('N')

    return fig


def init_parameterdict(conf):
    parameters_dict = {}
    for k, v in conf.__dict__.items():
        if not is_number(v):
            continue
        parameters_dict[k] = {}
        parameters_dict[k]['config_value'] = []
        for k_qc, v_qc in qc_dict.items():
            parameters_dict[k][k_qc] = []

    return parameters_dict


def print_sorted(data, filenames, sorted_colun=0):

    indices = num.argsort(data[:, sorted_colun])
    n = len(data.T)
    for index in indices:
        d = ["%1.1f" % val for val in data[index]]
        print("{filename: <60} {data}".format(
            **{'filename': filenames[index],
            'data': " | ".join(d)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", help='directories', nargs='*')
    parser.add_argument("--recip", help='Use the inverse', action='store_true',
                        default=False)
    parser.add_argument("--hists",
                        help='Make histogram plots for each result',
                        action='store_true', default=False)
    parser.add_argument("--xlim", help='set +- xlim', default=False, type=float)
    parser.add_argument("--nmin", help='number of minimum datapoints',
                        default=False, type=int)
    parser.add_argument(
        "--outfn", help='default: qs_histogram{.png (will be appended)}',
        default='qs_histogram.png')
    parser.add_argument("--acorr", help='autocorrelate distribution',
                        default=False, action='store_true')
    parser.add_argument("--result-filename", help='default: qs_inv.txt',
                        default='qs_inv.txt')
    parser.add_argument("--expected", help='Expected value', type=float,
                        default=0.)
    parser.add_argument("--sort-column", help='sort printed values by column N', type=int,
                        default=None)
    parser.add_argument("--filter-column", help='filter column N where x: e.g. --filter-column="5:> 10" (note the whitespace!)',
        type=str, default=False)
    parser.add_argument("--combine", help='combine results into a single plot', action='store_true',
        default=False)
    
    args = parser.parse_args()

    slopes = read_dirs(args)

    slopes = prefilter(slopes, args)

    parameters_dict = None
    nsl = len(slopes)
    ncol = int(num.sqrt(nsl))
    if args.hists:
        if args.combine:
            fig_hists = plt.figure()
            ax = fig_hists.add_subplot(111)
            axs = [ax] * nsl

        else:
            if ncol == 0:
                fig_hists = plt.figure()
                axs = [fig_hists.add_subplot(111)]
            elif ncol != 1:
                ncol += 1
                nrow = int(ncol % nsl) + 1
                fig_hists, ax = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(10, 10))
                axs = [ai for aii in ax for ai in aii]
            else:
                fig_hists = plt.figure()
                axs = [fig_hists.add_subplot(111)]

        axs = iter(axs)

    nbins = 201
    nslopes = len(slopes)
    all_data = []
    filenames = []

    for islope, (f, fn_config, sl) in enumerate(slopes):

        label = f.split('/')[0]
        filenames.append(f)
        if args.recip:
            sl = 1./sl
        # if args.xlim:
        if False:
            idx = num.where(num.abs(sl) < args.xlim)
        else:
            idx = num.arange(len(sl))

        d = sl[idx]
        if args.expected:
            median = args.expected
        else:
            median = num.median(sl)

        if args.hists:
            try:
                ax = axs.__next__()
            except StopIteration as e:
                print(e)
                print('try --combine')
                sys.exit()

            if args.acorr:
                d, bin_edges = num.histogram(d, bins=nbins)
                d = num.correlate(d, d, mode='full')
                ax.plot(d, label=label)
            else:
                ax.hist(d, bins=nbins, label=label, normed=False, alpha=0.5)
                ax.axvline(median, label='Model', color='grey', alpha=0.5)
                # ax.text(median, 0., 'median: %1.4f' % median,
                #         color='grey', size=7)
                ax.set_xlabel('1/Q' if not args.recip else 'Q')
            if args.combine:
                ax.legend()

            if args.xlim:
                ax.set_xlim([-args.xlim, args.xlim])
            cleanup_axis(ax)

        qc_dict = {
            'variance' : num.var(d),
            'kurt' : stats.kurtosis(d),
            'weight' : len(d),
            'median' : median,
        }

        if args.hists and not args.combine:
            txt = ''
            for i in qc_dict.items():
                txt += '%s: %1.4f\n' % i

            ax.text(1., 1., txt, color='grey', size=7, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right')

            cleanup_axis(ax)

        row_info = []
        conf = QConfig.load(filename=fn_config)
        if parameters_dict is None:
            parameters_dict = init_parameterdict(conf)
        for k, v in conf.__dict__.items():
            if not is_number(v):
                continue
            parameters_dict[k]['config_value'].append(v)
            for k_qc, v_qc in qc_dict.items():
                parameters_dict[k][k_qc].append(v_qc)

            row_info.append(v)
        all_data.append(row_info)
#
#    if args.filter_column:
#        col, expr = args.filter_column.split(':')
#        compare, val = expr.split()
#        val = float(val)
#        col = int(col)
#
#        def column_filter(row):
#            cmp_ = getattr(opr, operator_dict[compare])
#            return cmp_(row[col], val)
#    else:
#        def column_filter(row):
#            return True
#
    # print_sorted(num.array(all_data), filenames, args.sort_column, filter=column_filter)
    if args.sort_column:
        print_sorted(num.array(all_data), filenames, args.sort_column) #, filter=column_filter)

    if args.hists:
        fig_hists.savefig(args.outfn)
    
    ncol = len(qc_dict)
    nrow = len(parameters_dict)
    fig_size_scale = 2.
    fig_sigma , axs = plt.subplots(ncol, nrow, figsize=(nrow*fig_size_scale,
                                                        ncol*fig_size_scale))
    for ip, (parameter, k_v_qc) in enumerate(parameters_dict.items()):
        # else:
        #     ax.yaxis.set_major_locator(plt.NullLocator())
        #     # ax.spines['left'].set_visible(False)
        for ik, k in enumerate(qc_dict.keys()):
            ax = axs[ik, ip]
            if ip == 0:
                # is a left outer axis
                ax.set_ylabel(k)
            if ik == 0:
                # is a bottom outer axis
                ax.set_xlabel(parameter)
            # else:
            #     ax.spines['bottom'].set_visible(False)
            #     # remove all ticks, etc
            #     ax.xaxis.set_major_locator(plt.NullLocator())
            v_qc = k_v_qc[k]
            ax.scatter(k_v_qc['config_value'], v_qc, alpha=0.4)
            cleanup_axis(ax)

    

    fig_sigma.savefig(args.outfn + 'x.png')
    plt.show()
