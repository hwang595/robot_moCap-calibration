__author__ = 'gleicher'

# an "IK problem" takes a robot and a set of constraints and puts them together
# it's not really an IK problem - its more of a per-frame posing problem

# https://mathieularose.com/how-not-to-flatten-a-list-of-lists-in-python/
from itertools import chain

import numpy as N
import adInterface as AD

import scipy.optimize
import myslsqp

class IKFunc:
    def __init__(self, _robot, _consts, objs = []):
        self.robot = _robot
        self.consts = _consts
        self.objs = objs
        self.__name__ = "IKFunc"
        # gross hack - you can't have a zero length list of constraints
        # so if you have that situation, tell us so we might be able to help
        self.noIneq = False
        self.noEq = False
        # this is useful for damping
        # make sure its something whose length can be computed
        self.dampingBase = []
        # this is a tweaked "standard" objective function - built in to make things easier
        # set to negative to not use
        self.damping = -1

    def __call__(self, state):
        """
        given a state, return the 3 parts of the problem
        (eq consts, ineq consts, objective function)

        :param state:
        :return:
        """
        points,frames = self.robot.getFrames(state)
        eqs = []
        ineqs = []

        # CB - pass constraints values similar to objectives
        conTerms = {"state":state, "points":points, "frames":frames}

        for c in self.consts:
            e,i = c.constraint(**conTerms)
            eqs.append(e)
            ineqs.append(i)
        # don't forget the constraints from the robot!
        e,i = self.robot.constraint(**conTerms)
        eqs.append(e)
        ineqs.append(i)

        obj = 0
        for po in self.objs:
            # we assume that its a tuple (obj, weight)
            p = po[0]
            w = po[1]
            obj += p(state=state, points=points) * w
        if self.damping > 0:
            if len(self.dampingBase) == len(state):
                diff = state - self.dampingBase
                obj += N.dot(diff,diff)
            else:
                obj += N.dot(state,state)

        return list(chain.from_iterable(eqs)), list(chain.from_iterable(ineqs)), obj

    # an aid for solving: make a vector of variables for ad
    def makeAdVars(self, vector):
        return [ AD.adnumber(v,self.robot.varnames[c]) for (c,v) in enumerate(vector)]

    # the naive functions so that we can throw something at the built-in solvers
    # see spacetime for a discussion
    def naive_obj(self):
        return lambda(x) : self(x)[2]
    def naive_fe(self):
        return None if self.noEq   else lambda(x) : self(x)[0]
    def naive_fi(self):
        return None if self.noIneq else lambda(x) : self(x)[1]

    def nObjG_internal(self, x):
        v = self.makeAdVars(x)
        o = self(v)[2]
        rv =  o.gradient(v) if isinstance(o,AD.ADF) else N.zeros( (len(x)) )
        return rv
    def naive_objG(self):
       return lambda(x): self.nObjG_internal(x)

    def nFG_internal(self, x, eORi):
        v = self.makeAdVars(x)
        j = self(v)[ eORi ]
        if len(j)==0:
            return None
        return N.vstack([(i.gradient(v) if isinstance(i,AD.ADF) else N.zeros( (len(x)) )) for i in j])
    def naive_fiG(self):
        return None if self.noIneq else lambda (x) : self.nFG_internal(x,1)
    def naive_feG(self):
        return None if self.noEq   else lambda (x) : self.nFG_internal(x,0)

    # while we're at it... do a naive solve!
    def naive_slsqp(self,start):
        self.dampingBase = N.array(start)
        rv = scipy.optimize.fmin_slsqp(self.naive_obj(), start, fprime=self.naive_objG(),
                          f_eqcons=self.naive_fe(), fprime_eqcons=self.naive_feG(),
                          f_ieqcons=self.naive_fi(), fprime_ieqcons=self.naive_fiG())
        self.dampingBase = []
        return rv

    # this is for using myslsqp
    def func(self, state):
        e,i,o = self(state)
        return o,e,i
    def fprime(self, state):
        v = self.makeAdVars(state)
        e,i,o = self(v)
        og = o.gradient(v) if isinstance(o,AD.ADF) else N.zeros( (len(state)) )
        eg = N.vstack([(cv.gradient(v) if isinstance(cv,AD.ADF) else N.zeros( (len(state)) )) for cv in e]) if len(e) else []
        ig = N.vstack([(cv.gradient(v) if isinstance(cv,AD.ADF) else N.zeros( (len(state)) )) for cv in i]) if len(i) else []
        return o,e,i, og,eg,ig

    def myslsqp(self, start, callback=True, useLimits=True, **kwargs):
        self.dampingBase = N.array(start)
        if callback==True:
            callback = self.robot.cleanupCallback
        upper = self.robot.xUBounds if useLimits else None
        lower = self.robot.xLBounds if useLimits else None
        rv = myslsqp.myslsqp(lambda x: self.func(x),
                             lambda x: self.fprime(x),
                             start,
                             upper = upper,
                             lower = lower,
                             callback = callback,
                             **kwargs)
        self.dampingBase = []
        return rv

    # this is for doing least squares (levenberg marquardt)
    def asLSlist(self, state, doInEq=True, objWeight=.01):
        """
        does evaluation, but assembles it as a single list of things that should be zero
        (for using least squares)
        note: we cannot make the weight list inside the loop
        :return: listOfFVals
        """
        e,i,o = self(state)
        vals = [o * objWeight]
        for v in e:
            vals.append(v)
        if doInEq:
            for v in i:
                vals.append(0 if i>0 else i)
        return vals

    # do lm naively - no derivative evaluations
    def lsNaive(self,start, objWeight=.01, ineqWeight=.1):
        r =  scipy.optimize.leastsq(lambda x: self.asLSlist(x, objWeight),
                                    start,
                                    full_output=True
                               )
        if self.robot.cleanupCallback:
            self.robot.cleanupCallback(r[0])
        return r[0]