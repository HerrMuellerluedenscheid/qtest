import numpy as num

def radius(Ml, C1=32.7, C1_err=1.1, C2=0.33, C2_err=0.01, perturb=False):
    # default values by Petr Kolar, 2015
    if perturb:
        _C1_err = num.random.normal(scale=C1_err,size=len(Ml))
        _C2_err = num.random.normal(scale=C2_err,size=len(Ml))
    else:
        _C1_err = num.zeros(len(Ml))
        _C2_err = num.zeros(len(Ml))
    return (C1+_C1_err)*10**((C2+_C2_err)*Ml)

def wc(a, beta):
    return 2.33*beta/a

if __name__=='__main__':
    import matplotlib.pylab as pylab
    from math import pi

    mls = num.random.random(1000)*4.
    a = radius(mls)
    pylab.plot(mls, a, '+')
    pylab.ylabel('Radius [m]')
    pylab.xlabel('Magnitude')
    pylab.title('Ml vs. source radius')
    pylab.savefig('source_dimensions.png', dpi=200)
    pylab.show()
