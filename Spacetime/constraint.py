__author__ = 'gleicher'

"""
Implements constraints for a specific unit of time - so it can be used for IK

for spacetime, you need to associate it with a specific time frame

can have multiple scalar constraints, so it outputs 2 lists (eqs, ineqs)
note that this is the same as Robot.constraint\

note: for a while, there was an idea to make these work as objective terms
as well, however, this got lost (and probably wasn't an important idea)
"""

import adInterface as AD
import numpy as N
import math

class Constraint:
    def __init__(self, eqs, ineqs, noZ):
        self.noZ = noZ
        self.eqs = eqs
        self.ineqs = ineqs
        #CB - define bounds as lists (these set defaults and should be overridden by each subclass of Constraint)
        self.cUBounds = []
        self.cLBounds = []
        self.numConstraints = 0 # a Constraint object can actually have more than 1 constraint

        # do we use point information?
        self.usesPoints = 1
        self.usesPointDerivatives = 0
        # do we use state?
        self.usesState = 1
        self.usesStateDerivatives = 0

        pass

    def getConstInfo(self):
        return self.numConstraints, self.cUBounds, self.cLBounds

    def constraint(self, **kwargs):
        raise NotImplemented

class Nail(Constraint):
    """
    simplest possible constraint - puts a point at a specific position
    """
    def __init__(self, _pointID, _position, _noZ):
        Constraint.__init__(self,True,False,_noZ)
        self.pointID = _pointID
        self.position = _position
        #CB - added code to set bounds and numConstraints (for equality uBound == lBound)
        if self.noZ:
            self.cUBounds = [0,0]
        else:
            self.cUBounds =  [0,0,0]
        self.cLBounds = self.cUBounds
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)

    def constraint(self, points, **kwargs):
        if self.noZ:
            return [points[self.pointID][0]-self.position[0],points[self.pointID][1]-self.position[1]], []
        else:
            return [points[self.pointID][0]-self.position[0],points[self.pointID][1]-self.position[1],points[self.pointID][2]-self.position[2]], []

    def __repr__(self):
        return "<Nail %d @ (%f,%f,%f)>" % (self.pointID, self.position[0], self.position[1], self.position[2])

class alignAxis(Constraint):
    def __init__(self, _pt, _ax, _vec):
        Constraint.__init__(self,1,0,False)
        self.pt = _pt
        self.ax = _ax
        self.vec = _vec
        self.numConstraints = 1

    def constraint(self, frames, **kwargs):
        return [ 1-N.dot(frames[self.pt][:,self.ax],self.vec) ], []

class alignAxisGT(Constraint):
    def __init__(self, _pt, _ax, _vec, _gtv=.99):
        Constraint.__init__(self,0,1,False)
        self.pt = _pt
        self.ax = _ax
        self.vec = _vec
        self.gtv = _gtv
        self.numConstraints = 1
    def constraint(self, frames, **kwargs):
        return [], [ N.dot(frames[self.pt][:,self.ax],self.vec) - self.gtv ]

class Marker(Constraint):
    """
    simplest possible constraint - a nail that doesn't actually connect to anything!
    """
    def __init__(self, _position, _noZ=False):
        Constraint.__init__(self,True,False,_noZ)
        self.position = _position
        #CB - added code to set bounds and numConstraints (for equality uBound == lBound)
        if self.noZ:
            self.cUBounds = [0,0]
        else:
            self.cUBounds =  [0,0,0]
        self.cLBounds = self.cUBounds
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = 0

    def constraint(self, **kwargs):
        if self.noZ:
            return [], []
        else:
            return [], []

    def __repr__(self):
        return "<Marker %d @ (%f,%f,%f)>" % (self.pointID, self.position[0], self.position[1], self.position[2])

class AboveFloor(Constraint):
    """
    simplest inequality - only works in Y
    """
    def __init__(self, _pointID, _noZ, _floorHeight=0):
        Constraint.__init__(self, False, True, _noZ)
        self.pointID = _pointID
        self.floorHeight = _floorHeight
        #CB - added code to set bounds and numConstraints
        self.cUBounds = [float("inf")]
        self.cLBounds = [0]
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)

    def constraint(self, points, **kwargs):
        return [], [ points[self.pointID][1]-self.floorHeight ]

class VariableBetween(Constraint):
    """
    kindof like a joint limit - but implemented as a constraint
    """
    def __init__(self, _varID, _minV, _maxV, _noZ):
        Constraint.__init__(self, False, True, _noZ)
        self.varID = _varID
        self.maxV = _maxV
        self.minV = _minV
        #CB - added code to set bounds and numConstraints
        self.cUBounds = [float("inf"), float("inf")]
        self.cLBounds = [0, 0]
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)

    def constraint(self, state, **kwargs):
        return [], [ state[self.varID]-self.minV, self.maxV-state[self.varID] ]

class StateVelocity(Constraint):
    """
    makes sure a state variable is less than or equal to a max velocity
    _vMax is max velocity per frame (time parameterize frames outside the constraint)
    """
    def __init__(self, _varID, _vMax, noZ=False):
        Constraint.__init__(self, False, True, noZ)
        self.varID = _varID
        self.vMax = _vMax
        self.usesStateDerivatives = 1
        #CB - added code to set bounds and numConstraints
        self.cUBounds = [float("inf")]
        self.cLBounds = [0]
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)

    def constraint(self, stvel, **kwargs):
        return [], [self.vMax - abs(stvel[self.varID])]

class pointDistance(Constraint):
    """
    makes sure a point is at least some distance from a fixed location
    """
    def __init__(self, _pointID, _r, _x, _y, _z, noZ):
        Constraint.__init__(self, False, True, noZ)
        self.pointID = _pointID
        self.r = _r
        self.x = _x
        self.y = _y
        self.z = _z
        #CB - added code to set bounds and numConstraints
        self.cUBounds = [float("inf")]
        self.cLBounds = [0]
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)


    def constraint(self, points, **kwargs):
        dx = self.x - points[self.pointID][0]
        dy = self.y - points[self.pointID][1]
        if self.noZ:
            dst = AD.MATH.sqrt(dx*dx+dy*dy)
        else:
            dz = self.z - points[self.pointID][2]
            dst = AD.MATH.sqrt(dx*dx+dy*dy+dz*dz)

        return [], [dst - self.r]

class allPointsDistance(Constraint):
    """
    makes sure all points of a robot are at least some distance from a fixed location
    ignores the first points if you like
    """
    def __init__(self, _r, _x, _y, _z, _noZ, _firstPoint=1,numPoints=None):
        Constraint.__init__(self, False, True, _noZ)
        self.first = _firstPoint
        self.r = _r
        self.x = _x
        self.y = _y
        self.z = _z
        #CB - added code to set bounds and numConstraints
        self.cUBounds = [float("inf")] * numPoints
        self.cLBounds = [0] * numPoints
        assert len(self.cUBounds) == len(self.cLBounds)
        self.numConstraints = len(self.cUBounds)

    def constraint(self, points, **kwargs):
        lst = []
        for i in range(self.first,len(points)):
            dx = self.x - points[i][0]
            dy = self.y - points[i][1]
            if self.noZ:
                dst = AD.MATH.sqrt(dx*dx+dy*dy)
            else:
                dz = self.z - points[i][2]
                dst = AD.MATH.sqrt(dx*dx+dy*dy+dz*dz)
            lst.append(dst - self.r)
        return [], lst

