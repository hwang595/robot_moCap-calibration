__author__ = 'gleicher'

import numpy as N
import math

from MikesToys.timer import Timer

import scipy.optimize._slsqp

# taken right from slsqp
_epsilon = math.sqrt(N.finfo(float).eps)


slsqp_exit_modes = {-1: "Gradient evaluation required (g & a)",
                    0: "Optimization terminated successfully.",
                    1: "Function evaluation required (f & c)",
                    2: "More equality constraints than independent variables",
                    3: "More than 3*n iterations in LSQ subproblem",
                    4: "Inequality constraints incompatible",
                    5: "Singular matrix E in LSQ subproblem",
                    6: "Singular matrix C in LSQ subproblem",
                    7: "Rank-deficient equality constraint subproblem HFTI",
                    8: "Positive directional derivative for linesearch",
                    9: "Iteration limit exceeded"}

def makeBoundsArray(n, val, default):
    if val is None or val is False:
        return N.array([default] * n)
    else:
        try: # see if its an array - otherwise we assume its a number
            if len(val) == n:
                return N.array(val)
            else:
                raise TypeError("Boundary array is the wrong length")
        except: # assume that its a number
            return N.full(val, n)

def assembleConsts(eqs, ineqs):
    return N.concatenate((eqs if eqs!=None else [], ineqs if ineqs != None else []))

def assembleCjacs(eqs, ineqs):
    if ineqs is None or len(ineqs)==0:
        a = eqs
    elif eqs is None or len(eqs)==0:
        a = ineqs
    else:
        a = N.vstack( (eqs, ineqs) )
    # we need to add an extra column (?) for the fortran workspace
    if len(a)==0 or a is None:
        a = None
    else:
        a = N.concatenate((a,N.zeros([len(eqs)+len(ineqs),1])),1)
    return a

slsqp_last_trace = []
slsqp_last_status = {}

def myslsqp(func, fprime, x0,
            upper = None,
            lower = None,
            iter=100, acc=1.0E-6,
            iprint=1, disp=None, full_output=0, epsilon=_epsilon,
            verbose = False,
            callback=None):
    global slsqp_last_trace
    global slsqp_last_status
    slsqp_last_trace = []

    Ttotal = Timer("total")
    Tslsqp = Timer("_slsqp")
    Teval  = Timer("eval")
    TevalG = Timer("evalG")

    Ttotal.start()

    # do an evaluation to count the number of constraints
    # since we need the gradients on the first evaluation, we'll do that
    # remember - for us, fprime returns the values as well
    # we actually only need the sizes to make the workspaces
    # note: its unclear whether the values are ever used: the first iteration
    # might recompute them. but we need to build the datastructures
    TevalG.start()
    fret, eret, iret, fgret, egret, igret = fprime(x0)
    TevalG.stop()

    # n = number of variables
    n = len(x0)

    # m = total number of constraints
    # meq = number of equality constraints
    meq = len(eret)
    miq = len(iret)
    m = meq + miq

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1       # note: this is NOT MIQ!
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w =  N.zeros(len_w)
    jw = N.zeros(len_jw)

    # setup bounds - whether we need them or not
    xl = makeBoundsArray(n, lower, -1.0E12)
    xu = makeBoundsArray(n, upper,  1.0E12)

    # Initialize the iteration counter and the mode value
    mode =    N.array(0,int)
    acc =     N.array(acc,float)
    majiter = N.array(iter,int)
    majiter_prev = 0

    # build the fortran versions from the returned versions
    fx = fret
    c = assembleConsts(eret, iret)
    a = assembleCjacs(egret, igret)
    g = N.append(fgret,0.0)   # see note in scipy - this is required

    # make a zero size null jacobian, just in case
    # note that its zero rows all have the right number of columns
    a0 = N.zeros( (1,n+1) )

    # get the starting point as our vector
    x = N.array(x0)

    # stats
    neval = 0
    ngeval = 0
    nloop = 0

    slsqp_last_trace.append( (int(majiter), nloop, fx, max(c) if len(c) else 0, N.dot(c,c)) if len(c) else 0)

    # life with fortran - we need an infinite loop with a break...
    while 1:
        # note that we are already all set up and ready to go...
        # print "    %3d %3d %6.2f %6.2f %6.2f %d:%6.2f %6.2f"%(       mode, majiter, fx, min(x), max(x), len(g), N.dot(g,g), N.sum(a))
        Tslsqp.start()
        if a is None or len(a)==0:
            a = a0
        scipy.optimize._slsqp.slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw)
        Tslsqp.stop()
        # print "%3d %3d %3d %6.2f %6.2f %6.2f %d:%6.2f %6.2f"%(nloop, mode, majiter, fx, min(x), max(x), len(g), N.dot(g,g), N.sum(a))

        if callback is not None and majiter > majiter_prev:
            callback(x)

        if majiter > majiter_prev:
            # we don't actually remember the values for the iteration,
            # so we need to do an extra eval per iteration (ick!)
            # so that we can keep track of progress
            lf,le,li = func(x)
            try:
                cmax = max(le)
                cmag = N.dot(le,le)
            except:
                cmax = 0
                cmag = 0
            try:
                # cimax = max(le)
                cimax = max(li) # bug
                cmax = max(cmax,cimax)
                cmag += N.dot(li,li)
            except:
                pass
            slsqp_last_trace.append( (int(majiter), nloop, lf, cmax, cmag) )

        if abs(mode) != 1:
            break

        # do we need an eval?
        if mode == 0 or mode == 1:
            Teval.start()
            fret, eret, iret = func(x)
            Teval.stop()
            c = assembleConsts(eret,iret)
            fx = fret
            neval+=1

        # do we need a gradient eval (note: we're wasting the regular eval)
        if mode==0 or mode == -1:
            TevalG.start()
            fret, eret, iret, fgret, egret, igret = fprime(x)
            TevalG.stop()
            a = assembleCjacs(egret, igret)
            g = N.append(fgret,0.0)   # see note in scipy - this is required
            ngeval+=1

        nloop+=1
        majiter_prev = int(majiter)

    Ttotal.stop()

    if verbose:
        print "Done! - stop criteria(%d = %s)" % (int(mode),slsqp_exit_modes[int(mode)])
        print "loops(%d) evals(%d) grads(%d)" % (nloop,neval,ngeval)
        print "minimum", fx
        print Ttotal
        print Tslsqp
        print Teval
        print TevalG

    slsqp_last_status["Ttotal"] = Ttotal
    slsqp_last_status["Tslsqp"] = Tslsqp
    slsqp_last_status["Teval"] = Teval
    slsqp_last_status["TevalG"] = TevalG
    slsqp_last_status["mode"] = int(mode)
    slsqp_last_status["mode-name"] = slsqp_exit_modes[int(mode)]
    slsqp_last_status["minimum"] = fx
    slsqp_last_status["loops"] = nloop
    slsqp_last_status["evals"] = neval
    slsqp_last_status["ngeval"] = ngeval


    return x
