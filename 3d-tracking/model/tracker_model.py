import numpy as np

import torch
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # x, y,s,a, dy, dy, ds
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[
        4:,
        4:] *= 1000.  # give high uncertainty to the unobservable initial
        # velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.R *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.lost = False

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.lost = False

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def predict_no_effect(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        # self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


class KalmanBox3dTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, coord3d):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # X,Y,Z, dX, dY, dZ
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array(
            [[1, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[
        3:,
        3:] *= 1000.  # give high uncertainty to the unobservable initial
        # velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[3:, 3:] *= 0.01
        self.kf.R *= 0.01

        # X, Y, Z, s (area), r
        self.kf.x[:3] = coord3d.reshape(-1, 1)
        self.time_since_update = 0
        self.id = KalmanBox3dTracker.count
        KalmanBox3dTracker.count += 1
        self.history = [coord3d.reshape(-1, 1)]
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.aff_value = 0
        self.occ = False
        self.lost = False

    def update(self, coord3d):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = [coord3d.reshape(-1, 1)]
        self.hits += 1
        self.hit_streak += 1
        self.occ = False
        self.lost = False
        # X, Y, Z, s (area), r
        self.kf.update(coord3d.squeeze())

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:3])
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


class LSTM3dTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, device, lstm, coord3d):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # X,Y,Z,s,a, dX, dY, dZ, ds

        self.x = coord3d
        self.device = device
        self.lstm = lstm
        self.time_since_update = 0
        self.id = LSTM3dTracker.count
        LSTM3dTracker.count += 1
        self.nfr = 5
        self.history = [np.zeros_like(coord3d)]*self.nfr
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.aff_value = 0
        self.hidden_pred = None
        self.hidden_ref = None
        self.occ = False
        self.lost = False

    def update(self, coord3d):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = self.history[1:] + [coord3d-self.x]
        self.hits += 1
        self.hit_streak += 1

        with torch.no_grad():
            updated_loc, self.hidden_ref = self.lstm.refine(
                torch.from_numpy(self.x).view(1, 3).float().to(self.device),
                torch.from_numpy(coord3d).view(1, 3).float().to(self.device),
                self.hidden_ref)

        #print('Update', self.id+1, self.x, coord3d, updated_loc.data.cpu().numpy())
        self.x = updated_loc.data.cpu().view(3).numpy()

        self.occ = False
        self.lost = False

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        with torch.no_grad():
            pred_loc, self.hidden_pred = self.lstm.predict(
                torch.from_numpy(np.array(self.history)).view(self.nfr, -1, 3).float().to(self.device),
                torch.from_numpy(np.array(self.x)).view(-1, 3).float().to(self.device),
                self.hidden_pred)

        #print('Predict', self.id+1, self.x, pred_loc.data.cpu().numpy())
        prev = self.x.copy().flatten()
        self.x = pred_loc.data.cpu().numpy().flatten()
        self.velocity = self.x - prev

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        return self.x

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return np.hstack([self.x, self.velocity])

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


class LSTMKF3dTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, device, lstm, coord3d):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # X,Y,Z,s,a, dX, dY, dZ, ds

        self.x = coord3d
        self.lstm = lstm
        self.device = device
        self.time_since_update = 0
        self.id = LSTMKF3dTracker.count
        LSTMKF3dTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.aff_value = 0
        self.h_pd = None
        self.h_q = None
        self.h_r = None
        self.occ = False
        self.lost = False

    def update(self, coord3d):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        with torch.no_grad():
            updated_loc, self.h_r = self.lstm.refine(
                torch.from_numpy(self.x).view(1, 3).float().to(self.device),
                torch.from_numpy(coord3d).view(1, 3).float().to(self.device),
                self.h_r)

        # print(self.id, self.x, updated_loc.data.cpu().numpy())
        self.x = updated_loc.data.cpu().view(3).numpy()

        self.occ = False
        self.lost = False

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        with torch.no_grad():
            pred_loc, self.h_pd, self.h_q = self.lstm.predict(
                torch.from_numpy(self.x).view(1, 3).float().to(self.device),
                self.h_pd,
                self.h_q)

        # print(self.id, self.x, updated_loc.data.cpu().numpy())
        self.x = pred_loc.data.cpu().view(3).numpy()

        prev = self.x.copy().flatten()
        self.velocity = self.x - prev
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return np.hstack([self.x, self.velocity])

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


'''
Functions used in KalmanFilter 2D
'''

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area
      and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0] + 1
    h = bbox[3] - bbox[1] + 1
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score is None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2.,
                         x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.,
             score]).reshape((1, 5))
