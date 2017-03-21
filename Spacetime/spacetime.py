__author__ = 'gleicher'

"""
an attempt to define spacetime problems

at one level, all a spacetime problem is is a function that given a vector
(the KeyVariables - see states.py) returns the value of the objective function,
and the vector of constraint values (well, two - one for eqs, one for ineqs)

to do this, it needs to keep a lot of stuff
"""
from itertools import chain
import numpy as N
import adInterface as AD

# parts from this package
import states as ST
import ik
import myslsqp
import scipy.optimize

import arm

try:
    import pyipopt as IPOPT # added by CB for pyipopt
except ImportError:
    print "Cannot Import IPOPT, trying to get by without it..."


from MikesToys.timer import Timer

def returnListOrNone(lst):
    try:
        if len(lst)>0:
            return lst
    except:
        pass
    print "Zero Length Constraint List May Be a Problem!"
    return []

def pointVels(a, b,_noZ):
    if _noZ:
        return [ (a[i][0]-b[i][0], a[i][1]-b[i][1]) for i in range(len(a)) ]
    else:
        return [ (a[i][0]-b[i][0], a[i][1]-b[i][1], a[i][2]-b[i][2]) for i in range(len(a)) ]
    # return [ (a[i][0]-b[i][0], a[i][1]-b[i][1], a[i][2]-b[i][2]) for i in range(len(a)) ]

def pointAccs(a,b,c,_noZ):
    if _noZ:
        return [ (a[i][0]-b[i][0]*2+c[i][0], a[i][1]-b[i][1]*2+c[i][1]) for i in range(len(a))]
    else:
        return [ (a[i][0]-b[i][0]*2+c[i][0], a[i][1]-b[i][1]*2+c[i][1], a[i][2]-b[i][2]*2+c[i][2]) for i in range(len(a))]
    # return [ (a[i][0]-b[i][0]*2+c[i][0], a[i][1]-b[i][1]*2+c[i][1], a[i][2]-b[i][2]*2+c[i][2]) for i in range(len(a))]

class Spacetime:
    def __init__(self, robot, nstates):
        self.excludeKeys = []

        # things that define the spacetime problem
        self.robot = robot
        self.interpolationScheme = None

        # we need a state vector for injecting the keyvariables into
        self.defaultState = ST.KeyVector(nstates,self.robot.nvars)
        for i in range(len(self.defaultState)):
            self.defaultState[i] = robot.default

        # the constraints and objectives that make up the problem
        # note: these should be added by the "add" functions - so that
        # appropriate checking happens - that's why the names have the underscores
        # constraints
        self._pointConstraints = []      # tuples (time,constraint)
        self._allTimesConstraints = []   # point constraints applied to all times
        # objective functions - needs to be a tuple (weight, PointObjTerm)
        self._pointObjectives = []       # applied to ALL times (where the derivatives exist)

        # we need to know whether or not we have different kinds of constraints
        # before we ever gather them up (for example, in setting up the solver)
        self.hasIneq = False
        self.hasEq = False
        # it's also useful to know if we need derivatives in evaluation (so we can save
        # effort)
        self.maxStateDeriv = 0
        self.maxPointDeriv = 0

        # keep track of these kinds of things
        self.evalTimer = Timer("eval")
        self.nobjGTimer = Timer("nObjG")
        self.evalGTimer = Timer("evalG")

    def __len__(self): return len(self.defaultState)

    def makeBlankState(self):
        """
        this makes something like the default state, but since the default state is special,
        the code is redundant
        :return:  a state vector with the correct initial configuration
        """
        newstate = ST.KeyVector(len(self),self.robot.nvars)
        for i in range(len(self)):
            newstate[i] = self.robot.default
        return newstate


    def addConstraint(self, t, cons):
        self._pointConstraints.append( (t,cons) )
        if cons.eqs:   self.hasEq = True
        if cons.ineqs: self.hasIneq = True
        self.maxStateDeriv = max(self.maxStateDeriv,cons.usesStateDerivatives)
        self.maxPointDeriv = max(self.maxPointDeriv,cons.usesPointDerivatives)

    def addAllTimeConstraint(self, cons):
        self._allTimesConstraints.append( cons )
        if cons.eqs:   self.hasEq = True
        if cons.ineqs: self.hasIneq = True
        self.maxStateDeriv = max(self.maxStateDeriv,cons.usesStateDerivatives)
        self.maxPointDeriv = max(self.maxPointDeriv,cons.usesPointDerivatives)

    def addPointObjective(self, tupleOrObjective, weight=1.0):
        try:
            ptObjective = tupleOrObjective[0]
            weight = tupleOrObjective[1]
        except:
            ptObjective = tupleOrObjective
        self._pointObjectives.append( (ptObjective, weight) )
        self.maxStateDeriv = max(self.maxStateDeriv,ptObjective.usesStateDerivatives)
        self.maxPointDeriv = max(self.maxPointDeriv,ptObjective.usesPointDerivatives)

    def changeWeight(self, objective, newWeight):
        changed = None
        # since we cannot change the tuple, we have to go through the list
        # ugly, and non Pythonic
        for i in range(len(self._pointObjectives)):
            if self._pointObjectives[i][0]==objective:
                changed = True
                self._pointObjectives[i] = (objective,newWeight)
        if changed is None:
            raise KeyError("didn't find objective")


    def makeStateVector(self, keyvariables):
        """
        this makes a state vector (an array of state variables) from a key vector
        (an array of variables, with only the active variables)
        :param keyvariables: remember this takes a KEYVARIABLES (see states.py)
        :return:
        """
        # make the state vector
        keyvec = self.defaultState.inject(keyvariables, self.excludeKeys)
        # turn this into a state sequence
        states = keyvec if self.interpolationScheme == None else self.interpolationScheme(keyvec)
        return states

    def getStates(self, keyvariablesOrStateVector):
        """
        if you're passed either key variables or a state vector, make good use of it
        :param keyvariablesOrStateVector:
        :return: a state vector appropriate for this spacetime problem
        """
        # if we're passed a state vector, allow us to evaluate it
        nstates = len(self.defaultState)
        try:
            if keyvariablesOrStateVector.nkeys == nstates:
                states = keyvariablesOrStateVector
            else:
                raise IndexError("Wrong size State Vector to Spacetime Eval")
        except AttributeError:
            states = self.makeStateVector(keyvariablesOrStateVector)

        return states

    def getVarBounds(self):
        nstates = len(self)-len(self.excludeKeys)
        upper = N.empty(nstates * self.robot.nvars)
        lower = N.empty(nstates * self.robot.nvars)
        for i in range(nstates):
            upper[i*self.robot.nvars:(i+1)*self.robot.nvars] = self.robot.xUBounds
            lower[i*self.robot.nvars:(i+1)*self.robot.nvars] = self.robot.xLBounds
        return lower, upper


    def eval(self, keyvariablesOrStateVector):
        """
        evaluate the spacetime problem from a given state vector
        :param keyvariablesOrStateVector:
        :return: three values a scalar (objective) and a lists of the eqs and ineqs
        """
        self.evalTimer.start()
        # keep this around and handy
        nstates = len(self.defaultState)

        states = self.getStates(keyvariablesOrStateVector)

        # compute the velocity and acceleration vectors
        # just in case we need them
        # note: the ends might not be useful - but we compute something anyway
        stvels = None if self.maxStateDeriv<1 else [(states[i]-states[i-1] if i>0 else states[i+1]-states[i]) for i in range(nstates)]
        stacc = None if self.maxStateDeriv<2 else\
                [states[1]*2 - states[0] - states[2]] + \
                [(states[i]*2 - states[i-1] - states[i+1]) for i in range(1,nstates-1)] + \
                [states[nstates-2]*2 - states[nstates-3] - states[nstates-1]]

        # compute the point position for each point for each time frame
        # might be a little wasteful, but can serve as a caching strategy if they are used
        # note: the ends may be bogus, so don't use them if you really care
        # points = [self.robot(state) for state in states]
        points = []
        frames = []
        for state in states:
            p,f = self.robot.getFrames(state)
            points.append(p)
            frames.append(f)

        ptvels = None if self.maxPointDeriv<1 else \
                 [pointVels(points[1], points[0], self.robot.noZ)] + \
                 [pointVels(points[i], points[i-1], self.robot.noZ) for i in range(1,nstates) ]
        ptacc  = None if self.maxPointDeriv<2 else \
                 [pointAccs(points[0], points[1], points[2], self.robot.noZ)] + \
                 [pointAccs(points[i-1], points[i], points[i+1], self.robot.noZ) for i in range(1,nstates-1)] + \
                 [pointAccs(points[nstates-3], points[nstates-2],points[nstates-1], self.robot.noZ)]

        #######################
        # now gather up all of the constraints - point constraints are at specific
        # times
        eqs = []
        ineqs = []

        # CB - pass constraints values similar to objectives
        conTerms = {"states":states, "points":points, "t":0}

        # first let the robot add the constraints it wants to
        for t in range(nstates):
            if t not in self.excludeKeys:
                conTerms["t"] = t
                conTerms["state"] = states[t]
                conTerms["points"] = points[t]
                conTerms["frames"] = frames[t]
                if stvels: conTerms["stvel"] = stvels[t]
                if stacc: conTerms["stacc"] = stacc[t]
                if ptvels: conTerms["ptvel"] = ptvels[t]
                if ptacc: conTerms["ptacc"] = ptacc[t]
                e,i = self.robot.constraint(**conTerms)
                eqs.append(e)
                ineqs.append(i)

        # now add the point constraints
        # check to avoid things on excluded keys
        for t,c in self._pointConstraints:
            if t not in self.excludeKeys:
                conTerms["t"] = t
                conTerms["state"] = states[t]
                conTerms["points"] = points[t]
                conTerms["frames"] = frames[t]
                if stvels: conTerms["stvel"] = stvels[t]
                if stacc: conTerms["stacc"] = stacc[t]
                if ptvels: conTerms["ptvel"] = ptvels[t]
                if ptacc: conTerms["ptacc"] = ptacc[t]
                e,i = c.constraint(**conTerms)
                eqs.append(e)
                ineqs.append(i)

        # now add the all times constraints
        # note that we skip times with excluded keys
        for t in range(nstates):
            if t not in self.excludeKeys:
                for c in self._allTimesConstraints:
                    conTerms["t"] = t
                    conTerms["state"] = states[t]
                    conTerms["points"] = points[t]
                    conTerms["frames"] = frames[t]
                    if stvels: conTerms["stvel"] = stvels[t]
                    if stacc: conTerms["stacc"] = stacc[t]
                    if ptvels: conTerms["ptvel"] = ptvels[t]
                    if ptacc: conTerms["ptacc"] = ptacc[t]
                    e,i = c.constraint(**conTerms)
                    eqs.append(e)
                    ineqs.append(i)

        #######################
        # now make the objective function
        # warning - rather than += (have obj be a number) collect all the terms
        # as a list and use sum - this way automatic differentiation can look at
        # all terms together
        # this did not actually seem to make a difference in terms of performance
        # so maybe it could be switched back
        # the issue is that sum just seems to use radd - so we could provide a better
        # implementation of sum someday
        # because objTerms is build at each time step, we want to loop over time steps
        objlist = []
        objTerms = {"states":states, "points":points, "t":0}
        for t in range(nstates):
            objTerms["t"] = t
            objTerms["state"] = states[t]
            objTerms["points"] = points[t]
            objTerms["frames"] = frames[t]
            if stvels: objTerms["stvel"] = stvels[t]
            if stacc: objTerms["stacc"] = stacc[t]
            if ptvels: objTerms["ptvel"] = ptvels[t]
            if ptacc: objTerms["ptacc"] = ptacc[t]
            for po in self._pointObjectives:
                # we assume that its a tuple (obj, weight)
                try:
                    p = po[0]
                    w = po[1]
                except:
                    p = po
                    w = 1
                # in the event that the end derivatives aren't useful, avoid using them
                # for first derivatives, the 0 time is suspect, for 2nd derivatives, the end is as well
                dmax = max(p.usesPointDerivatives, p.usesStateDerivatives)
                if dmax<1 or t>0:   # if we use derivatives skip the first
                    if dmax<2 or t<nstates-1: # if we do second derivatives, skip the last
                        objlist.append( p(**objTerms) * w )
        # ad does this in a naive way
        obj = AD.fsum(objlist)

        self.lastKeyVariables = keyvariablesOrStateVector
        self.lastStates = states
        self.lastPoints = points
        self.lastFrames = frames

        self.evalTimer.end()

        return obj, list(chain.from_iterable(eqs)), list(chain.from_iterable(ineqs))

    def makeAdVars(self, vector):
        """
        this makes a keyvariables vector - but makes each of the variables an adnumber
        so we can take derivates of it. each is assigned a meaningful name.
        :param vector: the initial state of the variables
        :return: a vector of adnumbers
        """
        stateIds = [i for i in range(self.defaultState.nkeys) if i not in self.excludeKeys]
        adv = []
        c=0
        for t in stateIds:
            for (i,vn) in enumerate(self.robot.varnames):
                adv.append(AD.adnumber(vector[c], "%d:%s" % (t,vn)))
                c += 1
        return N.array(adv)

    # these functions create "naive" evaluation functions, suitable for optimization
    # routines
    # they are "naive" because each one requires a separate call to eval, so in the process
    # of one eval for the optimizer 3 (identical) calls to eval are made.
    # a caching scheme (or a less naive interface) is necessary
    def naive_obj(self):
        """
        :return: a function that takes a state (keyvariables) and returns the objective function value
        """
        return lambda(x) : self.eval(x)[0]
    def naive_fe(self):
        """
        :return: a function that takes a state (keyvariables) and returns the vector value of the equality constraints
        """
        return None if not(self.hasEq)   else lambda(x) : returnListOrNone(self.eval(x)[1])
    def naive_fi(self):
        """
        :return: a function that takes a state (keyvariables) and returns the vector value of the inequality constraints
        """
        return None if not(self.hasIneq) else lambda(x) : returnListOrNone(self.eval(x)[2])

    # evaluation with derivatives
    def evalG(self, x):
        self.evalGTimer.start()
        v = self.makeAdVars(x)
        fv,ev,iv = self.eval(v)

        f = fv.x if isinstance(fv,AD.ADF) else fv
        fg = fv.gradient(v) if isinstance(fv,AD.ADF) else N.zeros( (len(x)))

        if ev != None and len(ev):
            e = [ (c.x if isinstance(c,AD.ADF) else c) for c in ev]
            el = [(c.gradient(v) if isinstance(c,AD.ADF) else N.zeros( (len(x)) )) for c in ev]
            eg = N.vstack(el)
        else:
            e = []
            eg = []

        if iv != None and len(iv):
            i = [ (c.x if isinstance(c,AD.ADF) else c) for c in iv]
            ig = N.vstack([(c.gradient(v) if isinstance(c,AD.ADF) else N.zeros( (len(x)) )) for c in iv])
        else:
            i = []
            ig = []

        self.evalGTimer.stop()
        return f,e,i,fg,eg,ig

    # naive derivative evaluation
    # note that this is REALLY naive - since we need to create a new set of variables
    # for each sub-evaluation, so not only do we compute the gradients/jacobians 3 times,
    # but we make the variable names 3 times as well!
    # compute the gradient of the objective function - has to create a new vector of variables
    # note that there is an internal function so that we can make the necessary lambda expressions
    def nObjG_internal(self, x):
        self.nobjGTimer.start()
        v = self.makeAdVars(x)
        o = self.eval(v)[0]
        rv =  o.gradient(v) if isinstance(o,AD.ADF) else N.zeros( (len(x)) )
        self.nobjGTimer.end()
        return rv
    def naive_objG(self):
        """
        :return: a function that takes a keyvariables and returns a vector that is the gradient of the objective
        """
        return lambda(x): self.nObjG_internal(x)

    def nFG_internal(self, x, eORi):
        v = self.makeAdVars(x)
        j = self.eval(v)[ eORi ]
        if len(j)==0:
            return None
        return N.vstack([(i.gradient(v) if isinstance(i,AD.ADF) else N.zeros( (len(x)) )) for i in j])
    def naive_fiG(self):
        """
        :return: a function that takes a keyvariables and returns a matrix that is the jacobian of the ineq constraints
        """
        return lambda (x) : self.nFG_internal(x,2)
    def naive_feG(self):
        """
        :return: a function that takes a keyvariables and returns a matrix that is the jacobian of the eq constraints
        """
        return lambda (x) : self.nFG_internal(x,1)

    # START CB ADDITIONS


    def nFG_internal_all(self, x):
        v = self.makeAdVars(x)
        first = self.eval(v)[1]
        second = self.eval(v)[2]
        j = first + second
        if len(j)==0:
            return []
        return N.vstack([(i.gradient(v) if isinstance(i,AD.ADF) else N.zeros( (len(x)) )) for i in j])

    def evalWithAll(self, keyvariablesOrStateVector):
        """
        This function wraps eval to get both ineq and eq lists as a single vector making only 1 eval call

        :return:
        """
        obj,eq,ineq = self.eval(keyvariablesOrStateVector)
        return eq + ineq

    #CB -
    def eval_f(self, X, user_data=None):
        result = self.eval(X)[0]
        return result

    #CB -
    def eval_grad_f(self, X, user_data=None):
        result = self.nObjG_internal(X)
        return N.array(result)

    #added by CB to return all constraint values
    def eval_g(self, X, user_data=None):
        """
        :return: a function that takes a state (keyvariables) and returns the vector of all constraints
        """
        if (not(self.hasEq) and not(self.hasIneq)):
            return N.array([], dtype=float)
        else:
            result = returnListOrNone(self.evalWithAll(X))
            return N.array(result)

    #CB - FINISH!!! this needs to return the sparse Jocabian structure if true otherwise the evaluation of the Jacobian
    def eval_jac_g(self, X, flag, user_data = None):
        v = N.array(self.nFG_internal_all(X))
        if flag:
            tmp_result = N.where(N.ones(v.shape))
            result = (N.array(tmp_result[0]),N.array(tmp_result[1]))
        else:
            result = N.reshape(v, v.size)
        return result

    # def naive_fallG(self):
    #     """
    #     Added by CB to compute jacobian for all constraints (ipopt)
    #
    #     :return: a function that takes a keyvariables and returns a matrix that is the jacobian of all constraints
    #     """
    #     return lambda x,flag: self.pyipoptNFGWrapper(x, flag)

    def extractIK(self, time, ikDamping = -1):
        """
        extract an IK problem for a specific time. this gathers up all of the constraints active at
        the desired time (this include the "all times" constraints), as well as the objectives
        note: because we only look at one time, we can only consider objectives that have no derivatives
        the robot contributes its own constraints at the appropriate time
        :param time: the time (state number) to build this for
        :return: a new IKFunc
        """
        # get all the constraints at the appropriate times
        nc = [c[1] for c in self._pointConstraints if c[0]==time]
        # add the all times constraints to this list (since its being made just for this purpose)
        nc.extend(self._allTimesConstraints)
        obj = [p for p in self._pointObjectives if p[0].meaningfulForIK and max(p[0].usesPointDerivatives, p[0].usesStateDerivatives)==0]
        ic = ik.IKFunc(self.robot,nc,objs=obj)
        ic.damping = ikDamping
        return ic

    def constrainedFrames(self):
        """
        gives a list of frames (the number) where there are constraints
        :return: a list of state numbers
        """
        return sorted(set([c[0] for c in self._pointConstraints]))

    def upsample(self):
        """
        creates a new spacetime problem that inserts a new state in between every old state
        note that the new problem has 2*(n-1)+1 states
        :return: a new Spacetime object
        """
        raise NotImplementedError

    def downsample(self):
        """
        creates a new spacetime problem that deletes states (every other one)
        note that this leaves the ends, so it makes things with .5*(n-1)+1 states
        :return: a new Spacetime object
        """
        raise NotImplementedError

    def writeCSV(self,fname, statevec, doPts=[-1]):
        with open(fname,"w") as f:
            # header row
            f.write("time")
            for vn in self.robot.varnames:
                f.write(", %s" % vn)
            for pt in doPts:
                pi = pt if pt>0 else self.robot.npoints+pt
                pname = "end" if pi>=self.robot.npoints-1 else "P%d" % pi
                f.write(", %s.x, %s.y, %s.z" % (pname,pname,pname))
            f.write("\n")
            # data rows
            for t,st in enumerate(statevec):
                f.write("%3d" % t)
                for v in st:
                    f.write(", %f" % v)
                pts = self.robot(st)
                for pt in doPts:
                    pi = pt if pt>0 else self.robot.npoints+pt
                    f.write(", %f, %f, %f" % (pts[pi][0], pts[pi][1], pts[pi][2]))
                f.write("\n")

    def ikSolveAndLerp(self, ikDamping=.01, verbose=False, serial=False, solver="myslsqp"):
        fr = self.constrainedFrames()
        stv = ST.KeyVector(self.defaultState.nkeys, self.defaultState.nvars)
        for i in range(len(stv)):
            stv[i] = self.defaultState[i]
        for idx,t in enumerate(fr):
            i = self.extractIK(t, ikDamping=ikDamping)

            start = stv[fr[idx-1]] if idx and serial else stv[t]
            if solver=="myslsqp":
                ir = i.myslsqp(start,verbose=verbose)
            elif solver=="ls":
                ir = i.lsNaive(start)
            elif solver=="both":
                ir = i.lsNaive(start)
                ir = i.myslsqp(ir,verbose=verbose)
            else:
                print "Unknown IK Solver! %s (using slsqp)" % solver
                ir = i.myslsqp(start,verbose=verbose)
            stv[t]=ir
        # we also want to interpolate the excluded keys
        for e in self.excludeKeys:
            fr.append(e)
        fr.sort()
        stv.lerp(fr)
        return stv

    def myslsqp(self, start=None, iters=200, acc=1E-5, callback=True, verbose=False, doBounds=True):
        if start is None:
            start = self.defaultState.extract(excludeKeys=self.excludeKeys)
        else:
            if hasattr(start,"extract"):
                start = start.extract(excludeKeys=self.excludeKeys)
            if len(start) != self.defaultState.lenExtract(excludeKeys=self.excludeKeys):
                print "WARNING: wrong length start vector to myslsqp"
        if callback==True:
            callback = self.robot.cleanupCallback
        elif callback=="despin":
            def despinI(v):
                r = self.defaultState.inject(v,self.excludeKeys)
                arm.despinSeries(r)
                rv = r.extract(excludeKeys=self.excludeKeys)
                for i,e in enumerate(rv):
                    v[i]= e
            callback = despinI
        if doBounds:
            lower,upper = self.getVarBounds()
        else:
            lower,upper = None,None
        r = myslsqp.myslsqp(lambda(x):self.eval(x), lambda(x):self.evalG(x),start,iter=iters, upper=upper, lower=lower, acc=acc, callback=callback,verbose=verbose)
        return r

    def slsqp_naive(self, start=None, iters=200, acc=1E-5, callback=True, verbose=False):
        if start is None:
            start = self.defaultState.extract(excludeKeys=self.excludeKeys)
        if callback==True:
            callback = self.robot.cleanupCallback
        r = scipy.optimize.fmin_slsqp(self.naive_obj(), start, fprime=self.naive_objG(),
                              f_eqcons=self.naive_fe(), fprime_eqcons=self.naive_feG(),
                              f_ieqcons=self.naive_fi(), fprime_ieqcons=self.naive_fiG(),
                              callback=callback, iter=iters, acc=acc
                              )
        return r

    # CB - added to retrieve the constraint data
    def getConstData(self):
        nstates = len(self.defaultState)
        numC = 0
        uBounds = []
        lBounds = []

        # the robot can add constraints, but this doesn't really work within the constraint framework as far as bounds are concerned.
        # None of the robots actually have constraints, so I'll reevaluate this later.

        # now add the point constraints
        # check to avoid things on excluded keys
        for t,c in self._pointConstraints:
            if t not in self.excludeKeys:
                numC += c.numConstraints
                uBounds += c.cUBounds
                lBounds += c.cLBounds
        # now add the all times constraints
        # note that we skip times with excluded keys
        for t in range(nstates):
            if t not in self.excludeKeys:
                for c in self._allTimesConstraints:
                    numC += c.numConstraints
                    uBounds += c.cUBounds
                    lBounds += c.cLBounds
        return numC, N.array(lBounds, dtype=float), N.array(uBounds, dtype=float)

    # added by CB for pyipopt
    def ipopt_naive(self, start=None, iters=3000, acc=1E-5, callback=True, verbose=False):
        """
        x - # of variables
        xl - lower bounds of variables
        xu - upper bounds of variables
        m - # of constraints
        gl - lower bounds of constraints
        gu - upper bounds of constraints
        nnzj - number of nonzero values in jacobian
        nnzh - number of nonzero values in hessian (set to 0 if eval_h is not used)
        eval_f - objective function
        eval_grad_f - calculates gradient of objective function
        eval_g - calculates constraint values
        eval_jac_g - calculates jacobian
        eval_h - calculates hessian (optional, if not used set nnzh to 0)
        """

        try:
            if not(hasattr(IPOPT,"create")):
                print "WARNING: IPOPT not correctly installed (not start)"
                return start
        except NameError:
            print "WARNING: IPOPT is not installed!"
            return start

        if start is None:
            start = self.defaultState.extract(excludeKeys=self.excludeKeys)
        if callback==True:
            callback = self.robot.cleanupCallback
        #start = N.array(start, subok=True)
        numConst, lCBounds, uCBounds = self.getConstData()
        numVar = start.size
        lVBounds = []
        uVBounds = []
        for i in range(start.size / self.robot.nvars):
            lVBounds += self.robot.xLBounds
            uVBounds += self.robot.xUBounds
        lVBounds = N.array(lVBounds, dtype=float)
        uVBounds = N.array(uVBounds, dtype=float)

        struct = self.eval_jac_g(start,False)
        numNonZero = struct.size

        ipprob = IPOPT.create(numVar,
                              lVBounds,
                              uVBounds,
                              numConst,
                              lCBounds,
                              uCBounds,
                              numNonZero,
                              0,
                              self.eval_f,
                              self.eval_grad_f,
                              self.eval_g,
                              self.eval_jac_g)
        ipprob.num_option('tol', acc)
        ipprob.int_option('max_iter', iters)
        # call solve with initial state (start)
        x, zl, zu, constraint_multipliers, obj, status = ipprob.solve(start)
        ipprob.close()
        return x
    # end pyipopt

    def asLSlist(self,state,doInEq=True,objWeight=.01):
        o,e,i = self.eval(state)
        vals = [.0001 * s for s in state]
        vals.append(o * objWeight)
        for v in e:
            vals.append(v)
        if doInEq:
            for v in i:
                vals.append(0 if i>0 else i)
        return vals

    def asLSlistD(self,state,doInEq=True,objWeight=.01):
        v = self.makeAdVars(state)
        ls = self.asLSlist(v,doInEq,objWeight)
        return N.vstack([(i.gradient(v) if isinstance(i,AD.ADF) else N.zeros( (len(state)) )) for i in ls])


    # do lm naively - no derivative evaluations
    def lsNaive(self,start=None, objWeight=.01, doInEq=True, doDerivs=False,
                iters=None  # for compatibility with other solvers
                ):
        if start is None:
            start = self.defaultState.extract(excludeKeys=self.excludeKeys)

        r =  scipy.optimize.leastsq(lambda x: self.asLSlist(x, objWeight=objWeight,doInEq=doInEq),
                                    start,
                                    Dfun=None if doDerivs==False else lambda x: self.asLSlistD(x, objWeight=objWeight,doInEq=doInEq),
                                    full_output=True
                               )
        if self.robot.cleanupCallback:
            self.robot.cleanupCallback(r[0])

        return r[0]

    def bbox(self, keyvariablesOrStateVector):
        """
        get the XYZ range of the points (and only the points) of the entire motion
        :param keyvariablesOrStateVector:
        :return: X,Y,Z  (min/max) for each
        """
        states = self.getStates(keyvariablesOrStateVector)
        points = [self.robot(state) for state in states]
        # flatten the list
        pl = list(chain.from_iterable(points))

        minX = min([p[0] for p in pl])
        maxX = max([p[0] for p in pl])
        minY = min([p[1] for p in pl])
        maxY = max([p[1] for p in pl])

        minZ = 0.0
        maxZ = 0.0
        if not self.robot.noZ:
            minZ = min([p[2] for p in pl])
            maxZ = max([p[2] for p in pl])

        return (minX, maxX), (minY, maxY), (minZ, maxZ)

import csv

def readArmFile(filename, robot, firstFrame=None):
    with open(filename) as f:
        r = csv.reader(f)
        lines = [l for l in r]

        sf = 1 if not(firstFrame is None) else 0
        st = Spacetime(robot,len(lines) + sf)

        state = st.makeBlankState()

        if sf:
            state[0]=firstFrame
        for i,line in enumerate(lines):
            key = state[i+sf]
            for j,v in enumerate(line):
                key[j] = float(v)

        return st,state

