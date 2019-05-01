import numpy as np

# TODO tune values

class ConstantVelocityMotionModel(object):
    def __init__(self, dt):
        self.dt = dt
        self.A = np.matrix(((1, 0, self.dt, 0), (0, 1, 0, self.dt), (0, 0, 1, 0), (0, 0, 0, 1)))
        self.H = np.matrix(((1, 0, 0, 0), (0, 1, 0, 0)))
        self.I = np.matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))        
        return


class KalmanFilter(ConstantVelocityMotionModel):
    def __init__(self, fps):
        dt = 1.0/fps
        super(KalmanFilter, self).__init__(dt)
        self.P = np.matrix(((100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100)))
        self.Q = np.matrix(((100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100)))
        self.R = np.matrix(((100, 0), (0, 100)))
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
        self.P = self.A * self.P * self.A.T + self.Q
        return self.state

    def correct(self, measurement):
        self.dirtyState = False
        self.missedUpdates = 0
        
        K = self.P * self.H.T * np.linalg.pinv(self.H * self.P * self.H.T + self.R)
        self.state = self.state + K * (measurement - self.H * self.state)
        self.P = (self.I - K * self.H) * self.P
        return self.state

    def getState(self):
        return self.state

    def isValid(self):
        return self.missedUpdates < self.missedUpdatesThreshold

