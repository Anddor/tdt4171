from numpy import matrix

from filtering import Filter

transition_model = matrix([[0.7, 0.3],
                           [0.3, 0.7]])
sensor_model = matrix([[0.9, 0.2], [0.1, 0.8]])

f = Filter(transition_model=transition_model, sensor_model=sensor_model)

print(f.forward([[0.5], [0.5]], 0))
