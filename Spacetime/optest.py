# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 22:18:18 2014

@author: gleicher
"""

import scipy.optimize

def f(x):
    print "F ",x
    return x[0] + x[1]
    
def fprime(x):
    print "F'",x
    return [1,1]
    
def c(x):
    print "C ",x
    return [ x[0]*x[0] + x[1]*x[1] - 4 ]
    
def cprime(x):
    print "C'",x
    return [ [ 2*x[0], 2*x[1] ] ]
    
r = scipy.optimize.fmin_slsqp(f, [27,25], fprime=fprime, f_eqcons=c, fprime_eqcons=cprime)

print r