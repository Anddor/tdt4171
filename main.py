from numpy import matrix, transpose
from numpy.ma import multiply
from sklearn.preprocessing import normalize

transition_model = matrix([[0.7, 0.3],
                           [0.3, 0.7]])
sensor_model = matrix([[0.9, 0.2], [0.1, 0.8]])


def forward(belief_state, new_evidence):
    # Transition Step:
    prediction = transition_model * belief_state
    # Sensor step:
    not_normal = multiply(sensor_model[new_evidence], transpose(prediction))
    return normalize(not_normal, norm='l1')


def forward_backward(evidence_sequence, prior, f):
    t = len(evidence_sequence)
    forward_messages = []  # forward messages
    backward_messages = []  #
    smoothed_estimates = []  #

    forward_messages.append(prior)
    for i in range(1, t):
        forward_messages.append(f.forward(forward_messages[i - 1], evidence_sequence[i]))


print(forward([[0.5], [0.5]], 0))
