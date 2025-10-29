import numpy as np

states = ["R", "P", "S"]
win_states = ["P", "S", "R"]

two_step_states = [a+b for a in states for b in states]

def get_td_2(opponent_history):
    s = ''.join(opponent_history)
    eps = 0.1

    A = np.zeros((9,3), dtype=np.float64)

    for i, pair in enumerate(two_step_states):
        count_R = 0
        count_P = 0
        count_S = 0
        for j in range(len(s)-2):
            if s[j:j+2] == pair:
                next_move = s[j+2]
                if next_move == "R":
                    count_R += 1
                elif next_move == "P":
                    count_P += 1
                elif next_move == "S":
                    count_S += 1
        total = count_R + count_P + count_S + 3*eps
        A[i,0] = (count_R + eps)/total
        A[i,1] = (count_P + eps)/total
        A[i,2] = (count_S + eps)/total

    return A

def player(prev_play, opponent_history=[]):
    if prev_play != "":
        opponent_history.append(prev_play)

    if len(opponent_history) < 2:
        return np.random.choice(states)

    opponent_history = opponent_history[-50:]

    last_pair = ''.join(opponent_history[-2:])
    pi = np.zeros(len(two_step_states))
    pi[two_step_states.index(last_pair)] = 1

    A = get_td_2(opponent_history)

    pi_next = pi @ A 
    pi_next /= pi_next.sum()

    threshold = 0.34
    if np.max(pi_next) > threshold:
        guess = states[np.argmax(pi_next)]
    else:
        guess = np.random.choice(states, p=pi_next)

    next_state = win_states[states.index(guess)]
    return next_state