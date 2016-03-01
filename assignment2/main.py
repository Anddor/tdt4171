from numpy import matrix, transpose, zeros, ones, newaxis
from numpy.ma import multiply

transition_model = matrix([[0.7, 0.3],
                           [0.3, 0.7]])

sensor_model = [matrix([[0.9, 0],
                        [0, 0.2]]),
                matrix([[0.1, 0],
                        [0, 0.8]])]


def forward(belief_state, new_evidence):
    """Implements the forward algorithm, through matrix operations
    :param belief_state: Current belief state, before prediction or evidence
    :param new_evidence: New sensor evidence
    """
    f = sensor_model[new_evidence] * transition_model * belief_state
    normal = normalize(transpose(f))
    return transpose(normal)


def backward(b, evidence):
    """Implements the backward algorithm, through matrix operations
    :param evidence: evidence to update the b value
    :param b: The old b value
    """
    new_b = transition_model * sensor_model[evidence]
    new_b = new_b * b
    return new_b


def normalize(old_matrix):
    """Normalizes, row sums to one, a single matrix row
    :param old_matrix: single matrix row, not normalized
    """
    row_sum = old_matrix.sum()
    n = 1 / row_sum
    new_matrix = old_matrix * n
    return new_matrix


def forward_backward(evidence_sequence, prior):
    """Does the forwards-backwards-algorithm with the given evidence sequence and prior"""
    t = len(evidence_sequence)
    forward_messages = []  # forward messages
    b = matrix([1, 1])
    b = transpose(b)  # B needs to be as columns for matrix operations.
    smoothed_estimates = [0] * (t + 1)  #
    forward_messages.append(prior)
    for i in range(1, t + 1):
        forward_messages.append(forward(forward_messages[i - 1], evidence_sequence[i - 1]))

    print("forwarded= %s" % forward_messages)
    for j in range(t, 1, -1):
        normal = multiply(forward_messages[j], b)
        normal = normalize(transpose(normal))  # Normalizing requires rows.
        smoothed_estimates[j] = normal
        b = backward(b, evidence_sequence[j - 1])
        print("Backward message: %s" % b)

    return smoothed_estimates


def forward_only():
    evi = [0, 0, 1, 0, 0]
    t = len(evi)
    belief = matrix([[0.5], [0.5]])
    for i in range(t):
        belief = forward(belief_state=belief, new_evidence=evi.pop())
        print("Forward message: %s" % belief)


def forward_and_backward():
    evi = [0, 0, 1, 0, 0]
    pri = matrix([[0.5], [0.5]])
    ans = forward_backward(evi, pri)
    print(ans)


forward_only()
forward_and_backward()
