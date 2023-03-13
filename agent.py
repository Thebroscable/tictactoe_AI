import tensorflow as tf
import numpy as np
import random
import collections


class Agent:
    def __init__(self, state_space, action_space, model_file):
        self.model_file = model_file
        self.state_space = state_space
        self.action_space = action_space

        self.max_len = 2000
        self.expirience_replay = collections.deque(maxlen=self.max_len)

        self.gamma = 0.6
        self.epsilon = 0.1
        self.lr = 0.001
        self.epochs = 5

        self.q_network = self.build_model()

    def store_data(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(202,), name='input'))
        model.add(tf.keras.layers.Dense(128, activation='relu', name='dense1'))
        model.add(tf.keras.layers.Dense(64, activation='relu', name='dense2'))
        model.add(tf.keras.layers.Dense(self.action_space, activation='linear', name='output'))
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      metrics=['accuracy'])
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.q_network.predict(state)
            action = np.argmax(action[0])
        return action

    def train(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        licznik = 1
        print('trening', end=' ')
        for state, action, reward, next_state, terminated in minibatch:
            print('{} / {}, '.format(licznik, batch_size), end=' ')
            licznik += 1

            target = self.q_network.predict(state)

            if terminated:
                target[0][action] = reward
            else:
                t = self.q_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=self.epochs, verbose=0)
        print('')

    def save_model(self):
        self.q_network.save(self.model_file)

    def load_model(self):
        try:
            self.q_network = tf.keras.models.load_model(self.model_file)
        except:
            print('no file yet')

    def reduce_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon -= 0.001
        print('epsilon {}'.format(self.epsilon))
