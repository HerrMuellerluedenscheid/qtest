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
    M0 = 10.
    
    t_displacement = 10
    for t_rupture in [5,  30, 60]:

        ax.loglog(f, boxcar_amp_spec(f, M0, t_rupture, t_displacement))
        # ax.plot(f, boxcar_amp_spec(f, M0, t_rupture, t_displacement))
        # ax.set_yscale('log')
    plt.show()
