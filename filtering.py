from numpy import matrix


class Filter(object):
    def __init__(self, sensor_model, transition_model):
        self.sensor_model = sensor_model
        self.transition_model = transition_model
