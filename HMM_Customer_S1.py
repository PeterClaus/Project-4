import numpy as np
import os
import matplotlib.pyplot as plt

def model_open(file):
    model = open(file, 'r')
    change = 0
    transition_probability = {}
    emission_probability = {}
    start_probability = {}
    for line in model:
        if line.startswith('# States'):
            change = 1
            continue
        if change == 1:
            states = line.strip().split(',')
            change = 0
        if line.startswith('# Observations'):
            change = 2
            continue
        if change == 2:
            observations = line.strip().split(',')
            change = 0
        if line.startswith('# Start Probability'):
            change = 5
            continue
        if change == 5:
            if line.strip() == '':
                change = 0
                continue
            s = line.strip().split('-')
            start_probability[s[0]] = int(s[1])
        if line.startswith('# Transition Probability'):
            change = 3
            continue
        if change == 3:
            if line.strip() == '':
                change = 0
                continue
            t = line.strip().split('-')
            if t[0] not in transition_probability:
                transition_probability[t[0]] = {}
            transition_probability[t[0]][t[1]] = float(t[2])
        if line.startswith('# Emission Probability'):
            change = 4
            continue
        if change == 4:
            if line.strip() == '':
                break
            e = line.strip().split('-')
            if e[0] not in emission_probability:
                emission_probability[e[0]] = {}
            emission_probability[e[0]][e[1]] = float(e[2])
    return states, observations, start_probability, transition_probability, emission_probability


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


def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id

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

def accuracy_cal(x, y):
    if not x or not y:
        return None
    count = 0
    l = min(len(x), len(y))
    for i in range(l):
        if x[-i] == y[-i]:
            count = count + 1
    return round(count / l, 2)

states, observations, start_probability, transition_probability, emission_probability = model_open('HMM_S1.txt')


# Set all the emission_probability be the same
data = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
graph = []
for d in data:
    for i in emission_probability:
        for j in emission_probability[i]:
            emission_probability[i][j] = d


    states_id2label, states_label2id = generate_index_map(states)

    observations_id2label, observations_label2id = generate_index_map(observations)

    A = convert_map_to_matrix(transition_probability, states_label2id, states_label2id)

    l = len(A)
    for i in range(l):
        A[i][i] = 1 - sum(A[i])

    B = convert_map_to_matrix(emission_probability, states_label2id, observations_label2id)

    pi = convert_map_to_vector(start_probability, states_label2id)
    '''
    print('States: ')
    print(states)
    print()
    print('Transition probability: ')
    print(A)
    print()
    '''
    print('Emission probability: ')
    print(B)
    print()
    
    dir_path = 'customer'

    files = os.listdir(dir_path)

    files_count = len(files)
    '''
    print('There are ' + str(files_count) + ' files')
    print()
    '''
    Average_accuracy = []

    for current_file in range(files_count):
        #print('Current file is: '+ files[current_file])
        #print()
        
        observations_data, states_data = file_open(dir_path + '/' + files[current_file])

        PROBABILITY, EXPLANATION, path = viterbi(observations_data, A, B, pi)

        State = [states[n] for n in path]

        #print('The most likely explanation of the state: ')
        #print(State)
        #print()
        
        if len(states_data) > 0:
            accuracy = accuracy_cal(path, states_data)
            Average_accuracy.append(accuracy)
            #print('Accuracy: ')
            #print(accuracy)
            #print()
    value = round(sum(Average_accuracy)/len(Average_accuracy), 2)
    graph.append(value)
    #print('The average accuracy is: ' + str(value))

plt.plot(data, graph, color = 'r',label='Average Accuracy')
plt.show()






