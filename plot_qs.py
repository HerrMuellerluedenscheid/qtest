import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as num
import sys
import argparse


def plothist(slopes, recip=False):
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
    parser.add_argument("filename", help='filename of two column file')
    parser.add_argument("--recip", help='Use the inverse', action='store_true',
                     default=False)
    args = parser.parse_args()
    print(args)
    slopes = num.loadtxt(args.filename)
    if len(slopes) == 0:
        sys.exit('File is empty!')

    fig = plothist(slopes, recip=args.recip)
    fig.savefig(sys.argv[1] + '.hist.png')
    plt.show()
