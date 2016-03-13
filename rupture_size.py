import numpy.random as random

def radius(Ml, C1=32.7, C1_err=1.1, C2=0.33, C2_err=0.01):
    # default values by Petr Kolar, 2015
    _C1_err = random.normal(scale=C1_err,size=len(Ml))
    _C2_err = random.normal(scale=C2_err,size=len(Ml))
    return (C1+_C1_err)*10**((C2+_C2_err)*Ml)

def wc(a, beta):
    return 2.33*beta/a

if __name__=='__main__':
    import matplotlib.pylab as pylab
    from math import pi

    mls = random.random(1000)*4.
    a = radius(mls)
    pylab.plot(mls, a, '+')
    pylab.title('Ml vs. source radius')

    pylab.figure()
    beta = 2300.
    pylab.plot(mls, wc(beta, a), '+')
    pylab.title('Ml vs. wc')


    pylab.show()
