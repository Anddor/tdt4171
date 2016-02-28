from numpy import matrix, transpose
from numpy.ma import multiply
from sklearn.preprocessing import normalize


class Filter(object):
    def __init__(self, sensor_model, transition_model):
        self.sensor_model = sensor_model  # Matrix [[],[]]
        self.transition_model = transition_model  # Matrix with [[],[]]

    def forward(self, belief_state, new_evidence):
        # Transition Step:
        prediction = self.transition_model * belief_state
        # Sensor step:
        not_normal = multiply(self.sensor_model[new_evidence], transpose(prediction))
        return normalize(not_normal, norm='l1')
