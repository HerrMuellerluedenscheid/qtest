import numpy as num
import matplotlib.pyplot as plt


# https://www.geophysik.uni-muenchen.de/MESS/2012/programme/dahm_sudelfeld2012.pdf

def boxcar_amp_spec(f, M0, t_rupture, t_displacement):
    return M0 * num.abs(num.sinc(f * t_rupture)) * num.abs(num.sinc(f * t_displacement))


def corner_frequency(t_rupture, t_displacement):
    return 1./ (t_rupture + 2*t_displacement)


if __name__  == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)

    f = num.linspace(1E-3, 100, 10000)
    M0 = 1.

    t_displacement = 10
    for t_rupture in [1,  10, 100]:
        ax.loglog(f, boxcar_amp_spec(f, M0, t_rupture, t_displacement), label='factor %s' % t_rupture)
    ax.set_xlabel('f [Hz]')
    ax.set_ylabel('Amplitude (normalized)')
    ax.set_xlim(0, 20)
    ax.set_ylim(1E-10, 10.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    fig.savefig('stf_compare.png')
    plt.show()
