import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

# with open("./sonar/sonar.names") as f:
#     print(f.read())

df = pd.read_csv("./sonar/sonar.all-data", header=None)

X = df.drop(60, axis=1)
y = df[60]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

max_iter = 50
particles_num = 20
c1 = 2
c2 = 2
V_max = 4
V_min = -4
alpha = 0.65

seed = 454
features = X.columns
features_num = len(features)
salient_features = None
similar_features = []
dissimilar_features = []


def sf_determiner():
    # seed
    np.random.seed(seed)
    random.seed(seed)

    # the probability value of determining salient features as an initial number of features : l_sf
    l_sf = []

    for feature in range(1, features_num):
        # nominator
        l = features_num - feature

        # denominator
        sums = 0
        for i in range(l):                                       ##### CHANGED 11111111111111111111
            sums += features_num - i

        l_sf.append(round(l / sums, 3))

    # define M and interval of choosing sf
    # based on article epsilon is in range of [0.15, 0.7]
    epsilon = (np.random.randint(15, 70)) / 100
    M = int(epsilon * features_num)
    # if M is less than x in [x,M]
    M = 4 if M <= 3 else M

    salient_features = random.choices([i for i in range(3, M)], weights=l_sf[3:M], k=1)[0]
    return salient_features


salient_features = sf_determiner()


def dividing_features():
    correlations = []
    # find each column correlation
    for i in range(1, features_num):
        # Pearson Correlation Coefficient
        corr_i = X.corrwith(X[i])
        # if i = j
        corr_i[i] = 0

        correlations.append((round((sum(abs(i) for i in corr_i) / (features_num - 1)), 3), i))
        correlations.sort()

    # dividing into Similar and Dissimilar
    # first half => Dissimilar
    dissimilar_features = correlations[:len(correlations) // 2]
    # second half => Similar
    similar_features = correlations[len(correlations) // 2:]

    return dissimilar_features, similar_features


dissimilar_features, similar_features = dividing_features()


def initializing_particles():
    # seed
    np.random.seed(seed)

    # random initial Velocities
    velocities = [
        [round(np.random.rand(), 3) for i in range(features_num)] for i in range(particles_num)
    ]

    # random initial particles
    particles = []
    # salient_features = sf_determiner()

    sample_particle = [0] * (features_num - salient_features) + [1] * salient_features

    for particle in range(particles_num):
        np.random.shuffle(sample_particle)
        random_particle = sample_particle.copy()

        particles.append(random_particle)

    return particles, velocities


Particles, Velocities = initializing_particles()


def update_particle_positions(particles, velocities, best_particles, best_global):
    # seed
    np.random.seed(seed)

    new_velocities = np.zeros_like(velocities)
    new_positions = np.zeros_like(particles)

    # Particle changes
    for particle in range(particles_num):
        for d in range(features_num):
            new_velocities[particle][d] = velocities[particle][d] + c1 * np.random.rand() * (best_particles[particle][d] - particles[particle][d]) * velocities[particle][d] + c2 * np.random.rand() * (best_global[d] - particles[particle][d])

            # velocities limitation
            if new_velocities[particle][d] > V_max:
                new_velocities[particle][d] = 4
            if new_velocities[particle][d] < V_min:
                new_velocities[particle][d] = -4

                # Changing Particle Position
            # sigmoid func
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            new_positions[particle][d] = 1 if np.random.rand() < sigmoid(new_velocities[particle][d]) else 0

    return new_positions, new_velocities


def local_search(particles):
    # based on article
    N_s = int(alpha * salient_features)
    N_d = int((1 - alpha) * salient_features)

    # storing t+1 particle positions
    moved_particles = []
    for particle in range(particles_num):
        # Defining X_d and X_s
        features_index = [index for index, f in enumerate(particles[particle]) if f == 1]
        X_s = []
        X_d = []

        # Adding features to X_s and X_d
        for i in similar_features:
            if i[1] in features_index: X_s.append(i)
        for i in dissimilar_features:
            if i[1] in features_index: X_d.append(i)

        # ADD and DELETE Operators
        if 0 < (N_s - len(X_s)):
            for i in range(N_s - len(X_s)):
                for j in similar_features:
                    if j not in X_s:
                        X_s.append(j)
                        break
        elif (N_s - len(X_s)) < 0:
            for i in range(abs(N_s - len(X_s))):
                for j in X_s[::-1]:
                    X_s.pop()
                    break
        if 0 < (N_d - len(X_d)):
            for i in range(N_d - len(X_d)):
                for j in dissimilar_features:
                    if j not in X_d:
                        X_d.append(j)
                        break
        elif (N_d - len(X_d)) < 0:
            for i in range(abs(N_d - len(X_d))):
                for j in X_d[::-1]:
                    X_d.pop()
                    break

        # Only save features
        X_s = [i[1] for i in X_s]
        X_d = [i[1] for i in X_d]

        # Features
        X = X_d + X_s

        # Add features to particle
        particle_sample = [0] * features_num
        for feature_index in X:
            particle_sample[feature_index] = 1
        new_partcle = particle_sample.copy()
        moved_particles.append(new_partcle)

    return moved_particles
