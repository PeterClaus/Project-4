import numpy as np

states = ['zero', 'aware', 'considering', 'experiencing', 'ready', 'lost', 'satisfied']
observations = ['demo', 'video', 'testimonial', 'pricing', 'blog', 'payment']
start_probability = {'zero': 1, 'aware': 0, 'considering': 0, 'experiencing': 0, 'ready': 0, 'lost': 0, 'satisfied': 0}
transition_probability = {'zero': {'aware': 0.4}, \
                          'aware': {'considering': 0.3, 'ready': 0.01, 'lost':0.2}, \
                          'considering': {'experiencing': 0.2, 'ready': 0.02, 'lost': 0.3}, \
                          'experiencing': {'ready': 0.3, 'lost': 0.3},\
                          'ready': {'lost': 0.2}}
emission_probability = {'zero': {'demo': 0.1, 'video': 0.01, 'testimonial': 0.05, 'pricing': 0.3, 'blog': 0.5, 'payment': 0}, \
                        'aware': {'demo': 0.1, 'video': 0.01, 'testimonial': 0.15, 'pricing': 0.3, 'blog': 0.4, 'payment': 0}, \
                        'considering': {'demo': 0.2, 'video': 0.3, 'testimonial': 0.05, 'pricing': 0.4, 'blog': 0.4, 'payment': 0}, \
                        'experiencing': {'demo': 0.4, 'video': 0.6, 'testimonial': 0.05, 'pricing': 0.3, 'blog': 0.4, 'payment': 0}, \
                        'ready': {'demo': 0.05, 'video': 0.75, 'testimonial': 0.35, 'pricing': 0.2, 'blog': 0.4, 'payment': 0}, \
                        'lost': {'demo': 0.01, 'video': 0.01, 'testimonial': 0.03, 'pricing': 0.05, 'blog': 0.2, 'payment': 0}, \
                        'satisfied': {'demo': 0.4, 'video': 0.4, 'testimonial': 0.01, 'pricing': 0.05, 'blog': 0.5, 'payment': 1}}




def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id

states_id2label, states_label2id = generate_index_map(states)
observations_id2label, observations_label2id = generate_index_map(observations)

def file_open(filename):
    f = open(filename, 'r')
    observation = []
    stateV = []
    change = 0
    m = ''
    for line in f:
        if line.startswith('# States'):
            change = 1
            continue
        if line.startswith('#'):
            continue
        line = line.strip().split(',')
        if change == 0:
            if line[0] == '':
                observation.append([])
            else:
                observation.append([observations_label2id[n.lower()] for n in line])
        if change == 1: 
            if line[0] == '':
                continue
            for n in line:
                stateV.append(states_label2id[n.lower()])
    return observation, stateV

def convert_map_to_vector(map_, label2id):
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v

def convert_map_to_matrix(map_, label2id1, label2id2):
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for line in map_:
        for col in map_[line]:
            m[label2id1[line]][label2id2[col]] = map_[line][col]
    return m


def viterbi(obs, A, B, pi):
    sl = A.shape[0]
    obsl = len(obs)
    B2 = 1 - B

    def observation_prob(obs2):
        B3 = B2.copy()
        B3[:, obs2] = B[:, obs2]
        state = np.prod(B3, axis=1)
        return state

    PROBABILITY = np.zeros((obsl, sl), dtype = float)
    PROBABILITY[0, :] = pi * observation_prob(obs[0])
    EXPLANATION = np.zeros((obsl, sl), dtype=np.int)
    for i in range(1, obsl):
        tmp = np.repeat(PROBABILITY[i - 1, :].reshape(-1, 1), sl, axis=1)
        tmp *= A
        EXPLANATION[i, :] = np.argmax(tmp, axis=0)
        PROBABILITY[i, :] = np.max(tmp, axis=0) * observation_prob(obs[i])
    path = [0] * (obsl + 1)
    path[-1] = np.argmax(PROBABILITY[-1, :])
    for j in range(2, obsl + 1):
        path[-j] = EXPLANATION[-j, path[-j + 1]]
    return PROBABILITY, EXPLANATION, path

def accuracy(x, y):
    if not x or not y:
        return None
    count = 0
    l = min(len(x), len(y))
    for i in range(l):
        if x[-i] == y[-i]:
            count = count + 1
    return round(count / l, 2)




A = convert_map_to_matrix(transition_probability, states_label2id, states_label2id)

l = len(A)
for i in range(l):
    A[i][i] = 1 - sum(A[i])

B = convert_map_to_matrix(emission_probability, states_label2id, observations_label2id)

pi = convert_map_to_vector(start_probability, states_label2id)

observations_data, states_data = file_open('customer\hmm_customer_1586733275338.txt')

PROBABILITY, EXPLANATION, path = viterbi(observations_data, A, B, pi)

State = [states[n] for n in path]

if len(states_data) > 0:
    accuracy = accuracy(path, states_data)

print('States: ')
print(states)
print()
print('Transition probability: ')
print(A)
print()
print('Emission probability: ')
print(B)
print()
print('The most likely explanation of the state: ')
print(State)
print()
if len(states_data) > 0:
    print('Accuracy: ')
    print(accuracy)




