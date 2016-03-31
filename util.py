from pyrocko import moment_tensor
from matplotlib import pyplot as plt


def M02tr(Mo, stress, vr):
    #stress=stress drop in MPa
    #vr=rupture velocity in m/s
    #Mo = seismic moment calculated with Hanks and Kanamori,1979    
    #Mo=(10.**((3./2.)*(Mw+10.73))/1E+7 #Mo in Nm
    #Calculate rupture length or source radio (m) with Madariaga(1976), stress drop on a circular fault
    Lr = ((7.*Mo)/(16.*stress*1E+6))**(1./3.) #stress Mpa to Nm2
    #Calculate ruputure time in seconds with the rupture velocity
    tr = Lr/vr
    return tr


def fmin_by_magnitude(magnitude, stress=0.1, vr=2750):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    duration = M02tr(Mo, stress, vr)
    return 1./duration


class Magnitude2fmin():
    def __init__(self, stress, vr, lim):
        self.stress = stress
        self.vr = vr
        self.lim = lim

    def __call__(self, magnitude):
        return max(fmin_by_magnitude(magnitude, self.stress, self.vr), self.lim)

    @classmethod
    def setup(cls, stress=0.1, vr=2750, lim=0.):
        return cls(stress, vr, lim)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('fmin')


class Magnitude2Window():
    def __init__(self, t_static, t_factor):
        self.t_static = t_static
        self.t_factor = t_factor

    def __call__(self, magnitude):
        return self.t_static+self.t_factor/fmin_by_magnitude(magnitude)

    @classmethod
    def setup(cls, t_static=0.1, t_factor=5.):
        return cls(t_static, t_factor)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('time')


