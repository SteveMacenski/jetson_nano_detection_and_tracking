import numpy as np

# TODO tune values
# TODO add KF to project
# TODO add constant rate (10hz)

class ConstantVelocityMotionModel():
    def __init__(self, dt):
        self.dt = dt
        self.A = np.array((1, 0, self.dt, 0), (0, 1, 0, self.dt), (0, 0, 1, 0), (0, 0, 0, 1))
        self.H = np.array((1, 0, 0, 0), (0, 1, 0, 0))
        self.I = np.array((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))        
        return


class KalmonFilter(ConstantVelocityMotionModel):
    def __init__(self, fps):
        dt = 1.0/fps
        super(KalmonFilter, self).__init__(dt)
        self.P = np.array((100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100))
        self.Q = np.array((100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100))
        self.R = np.array((100, 0), (0, 100))
        return

    def initializeFilter(self, detection, missedUpdatesThresh = 5):
        self.state = detection
        self.missedUpdatesThreshold = missedUpdatesThresh
        self.missedUpdates = 0
        self.dirtyState = False
        return

    def predict(self):
        self.dirtyState = True
        self.missedUpdates = self.missedUpdates + 1

        self.state = self.A * self.state
        self.P = self.A * self.P * self.A.T() + self.Q
        return

    def correct(self, measurement):
        self.dirtyState = False
        self.missedUpdates = 0
        
        K = self.P * self.H.T() * (self.H * self.P * self.H.T() + self.R).T()
        self.state = self.state + K * (measurement - self.H * self.state)
        self.P = (I - K * self.H) * self.P
        return

    def getState(self):
        return self.state

    def isValid(self):
        return self.missedUpdates < self.missedUpdatesThreshold

