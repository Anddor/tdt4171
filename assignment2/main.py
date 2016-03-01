from numpy import matrix, transpose, zeros, ones
from numpy.ma import multiply
from sklearn.preprocessing import normalize

transition_model = matrix([[0.7, 0.3],
                           [0.3, 0.7]])

sensor_model = [matrix([[0.9, 0],
                        [0, 0.2]]), \
                matrix([[0.1, 0],
                        [0, 0.8]])]


def forward(belief_state, new_evidence):
    # Transition Step:
    # prediction = transition_model * belief_state
    # Sensor step:
    # not_normal = multiply(sensor_model[new_evidence], transpose(prediction))

    f = sensor_model[new_evidence] * transition_model * belief_state
    normal = normalize(transpose(f), norm='l1')
    # print("evidence= %s \n f=%s \n normal=%s " % (sensor_model[new_evidence], f, normal))

    return transpose(normal)


def backward(b, evidence):
    #    transition_model *
    new_b = transition_model * sensor_model[evidence]
    new_b = new_b * b
    return new_b


def forward_backward(evidence_sequence, prior):
    t = len(evidence_sequence)
    forward_messages = []  # forward messages
    b = matrix([1, 1])
    b = transpose(b)
    smoothed_estimates = [0] * (t + 1)  #

    forward_messages.append(prior)
    for i in range(1, t+1):
        print("i=%s" % i)
        forward_messages.append(forward(forward_messages[i - 1], evidence_sequence[i-1]))

    for j in range(t, 1, -1):
        normal = multiply(forward_messages[j], transpose(b))
        print("j=%s \n forward=%s \n b=%s \n normal=%s" % (j, forward_messages[j], b, normal))
        smoothed_estimates[j] = normal
        b = backward(b, evidence_sequence[j-1])

    return smoothed_estimates


# r1 = forward([[0.5], [0.5]], 0)
# r2 = forward(r1, 0)

# print("rain on day1, given umbrella: %s \n rain on day2 given umbrella: %s" % (r1, r2))

evi = [0, 0]
pri = [[0.5], [0.5]]

res = forward_backward(evi, pri)
print("result=%s" % res)

evi5 = [0, 0, 1, 0, 0]

ans = forward_backward(evi5, pri)
print(ans)