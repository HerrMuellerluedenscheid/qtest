import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as num
import scipy.stats as stats
import sys
import argparse


def plothist(slopes, recip=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
    if recip:
        slopes = 1./slopes
        idx = num.where(num.abs(slopes) < 200)
    else:
        idx = num.arange(len(slopes))
    ax.hist(slopes[idx], bins=201)
    ax.axvline(num.median(slopes), label='median', color='grey')
    ax.text(num.median(slopes), 0., 'median: %1.4f' % num.median(slopes),
            color='grey')
    ax.set_xlabel('1/Q')
    ax.set_ylabel('N')

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", help='filename of two column file', nargs='*')
    parser.add_argument("--recip", help='Use the inverse', action='store_true',
                        default=False)
    parser.add_argument("--xlim", help='set +- xlim', default=False, type=float)
    parser.add_argument("--nmin", help='number of minimum datapoints',
                        default=False, type=int)
    parser.add_argument("--outfn", help='default: qs_histogram.png', default='qs_histogram.png')
    parser.add_argument("--acorr", help='autocorrelate distribution',
                        default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    slopes = []
    for f in args.filenames:
        try:
            d = num.loadtxt(f, ndmin=1)
        except ValueError as e:
            print('failed to read fn: %s' % f)
            continue

        if len(d) == 0:
            print('File is empty %s ' % f)
            continue

        elif args.nmin and len(d) < args.nmin:
            print('Too few datapoints: %s' % f)
            continue

        slopes.append((f, d))

    nsl = len(slopes)
    ncol = int(num.sqrt(nsl))
    nrow = int(ncol % nsl) + 1
    fig, ax = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(10, 10))
    axs = iter([ai for aii in ax for ai in aii])
    nbins = 201
    for f, sl in slopes:
        ax = axs.next()
        if args.recip:
            sl = 1./sl
        if args.xlim:
            idx = num.where(num.abs(sl) < args.xlim)
        else:
            idx = num.arange(len(sl))
        d = sl[idx]
        if args.acorr:
            print(d)
            d, bin_edges = num.histogram(d, bins=nbins)
            d = num.correlate(d, d, mode='full')
            # d = d[d.size/2:]
            print(d)
            ax.plot(d, label=f)
        else:
            ax.hist(d, bins=nbins, label=f)
            ax.axvline(num.median(sl), label='median', color='grey', alpha=0.5)
            ax.text(0., 1., f, color='black', transform=ax.transAxes, size=7,
                    rotation=90)
            ax.text(num.median(sl), 0., 'median: %1.4f' % num.median(sl),
                    color='grey', size=7)
        ax.text(1., 1., 'kurt: %1.4f' % stats.kurtosis(d),
                color='grey', size=7, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right')

    # ax.set_xlabel('1/Q')
    # ax.set_ylabel('N')
    fig.savefig(args.outfn)
    plt.show()
