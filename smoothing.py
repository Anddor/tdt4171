def forward_backward(evidence_sequence, prior, f):
    t = len(evidence_sequence)
    forward_messages = []  # forward messages
    backward_messages = []  #
    smoothed_estimates = []  #

    forward_messages.append(prior)
    for i in range(1, t):
        forward_messages.append(f.forward(forward_messages[i-1], evidence_sequence[i]))

