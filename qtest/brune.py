import numpy as num
from .rupture_size import radius as source_radius
from pyrocko.guts import Object, Float


def brune(t, sigma, r, beta, mu, z, a):
    '''
    Principles of Seismology, 18.28

    :param t: time vector
    :param sigma: stress drop
    :param beta: v_s
    :param mu: shear module
    :param r: distance
    :param z: depth
    :param a: source radius
    '''
    b = 2.33*beta / a
    u1 = sigma*beta/mu * (t-r/beta) * num.exp(-b*(t-r/beta)) # principles of seismology
    return u1


def brune_omega(w, sigma, r, beta, mu, a):
    '''
    Principles of Seismology, 18.28

    :param w: frequencies
    :param sigma: stress drop
    :param beta: v_s
    :param mu: shear module
    :param r: distance
    :param a: source radius
    '''
    b = 2.33*beta / a
    U = sigma*beta/mu / (w**2 + b **2)
    return U


class Brune(Object):
    sigma = Float.T()
    mu = Float.T()
    beta = Float.T()

    def preset(self, source, target):
        self.source = source
        self.target = target

    def drop(self):
        self.source = None
        self.target = None

    def evaluate(self, freqs):
        '''
        :param freqs: f in rad/s

        returns U(omega)
        '''
        r = self.source.distance_to(self.target)
        z = self.source.depth
        # eigentlich local magnitude. Conversion siehe Fischer paper.
        a = source_radius([self.source.magnitude])
        #print self.source.magnitude, a
        omega = 2.*num.pi*freqs
        U = brune_omega(w=omega, sigma=self.sigma, r=r, beta=self.beta, mu=self.mu, z=z, a=a)
        #import matplotlib.pyplot as plt

        #fig= plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(omega, U)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #plt.show()
        self.drop()
        return U


def presentation_plot():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    duration = 20
    sampling_rate = 100
    sigma = 2.9E6
    r = 8000.
    beta = 3500.
    mu = 2e10
    z = 8000
    #a = 300.
    t = num.linspace(0.002, duration, duration*sampling_rate)
    f = 1./t
    #omega = 2*num.pi * f
    cmap = cm.copper
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=50, vmax=450))
    sm._A = []
    #aa = num.linspace(50, 500, 100)
    aa = [300, 500]
    fcs = [16, 20]
    for ii, a in enumerate(aa):
        fig = plt.figure(figsize=(3, 2.3))
        ax = fig.add_subplot(111)
        U = brune_omega(w=f, sigma=sigma, r=r, beta=beta, mu=mu, z=z, a=a)
        ax.plot(f, U, color=cmap((a-50)/450), label=a)
        ax.set_title('A$_%s$' % (ii))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axvline(fcs[ii])
        fig.savefig('synthetic_tests/brune_example_%s.svg' %ii, dpi=200)
        ax.set_ylabel('A')
        ax.set_xlabel('f')
        #Ufft = num.fft.rfft(U)
        #omegafft = num.fft.rfftfreq(n=len(U), d=1/sampling_rate)
    #ax = fig.add_subplot(311)
    #ax.plot(t, u2, 'o')
    #ax = fig.add_subplot(313)
    #ax.plot(t, num.abs(u1), 'bo')
    #ax.plot(omegafft, Ufft)
    #ax.plot(Ufft)
    plt.show()

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm
    #presentation_plot()
    #import sys
    #sys.exit(0)
    duration = 20
    sampling_rate = 1000
    sigma = 2.9E6
    r = 10000.
    beta = 3000.
    mu = 2e10
    z = 8000
    #a = 300.
    t = num.linspace(0.001, duration, duration*sampling_rate)
    f = 1./t
    #omega = 2*num.pi * f
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = cm.copper
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=50, vmax=450))
    sm._A = []
    aa = num.linspace(50, 500, 100)
    for a in aa:
        U = brune_omega(w=f, sigma=sigma, r=r, beta=beta, mu=mu, z=z, a=a)
        ax.plot(f, U, color=cmap((a-50)/450), label=a)
        #Ufft = num.fft.rfft(U)
        #omegafft = num.fft.rfftfreq(n=len(U), d=1/sampling_rate)
    plt.colorbar(sm)
    #ax = fig.add_subplot(311)
    #ax.plot(t, u2, 'o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('f[Hz]')
    ax.set_ylabel('A')
    fig.savefig('brune_sources.png', dpi=200)
    #ax = fig.add_subplot(313)
    #ax.plot(t, num.abs(u1), 'bo')
    #ax.plot(omegafft, Ufft)
    #ax.plot(Ufft)
    plt.show()
