from pettingzoo.classic import tictactoe_v3
from q_learning import Agent as QL
from sarsa_lambda import Agent as SL
from q_lambda import Agent as QLL
from sarsa import Agent as S
from DQN import Agent as DQN
from minmax import findBestMove
import numpy as np
import random


def observation2state(observation):
    vector = observation.flatten()
    state = 0

    for i in range(0, 18, 2):
        if vector[i] ^ vector[i+1]:
            state += (vector[i]*2 + vector[i+1]) * 3**(i//2)

    return state


def count_data(done, reward, won, draw, lost):
    if done:
        if reward == 1:
            won += 1
        elif reward == 0:
            draw += 1
        elif reward == -1:
            lost += 1
    return won, draw, lost


env = tictactoe_v3.env()

agent1 = SL(n_states=3 ** 9, n_actions=9, alpha=0.01, gamma=0.99, lambda_=0.99, epsilon=0.0)
# agent1 = DQN(input_dim=(27,), n_actions=9, lr=0.01, gamma=0.99, epsilon_end=0.0, epsilon=0.0)
agent2 = DQN(input_dim=(27,), n_actions=9, lr=0.01, gamma=0.99, epsilon_end=0.0, epsilon=0.0)
# agent2 = SL(n_states=3 ** 9, n_actions=9, alpha=0.01, gamma=0.99, lambda_=0.99, epsilon=0.0)

agent1.load_data(2)
# agent2.load_data(2)
# agent1.load_model(model_file='model/dqn_weights.h5')
agent2.load_model(model_file='model/dqn_weights.h5')

episodes = 100
user = 1
won1 = 0
draw1 = 0
lost1 = 0
won2 = 0
draw2 = 0
lost2 = 0

for episode in range(episodes):
    env.reset()
    state = None
    print('------------*episode: {}*------------'.format(episode+1))

    for _ in env.agent_iter():
        observation, reward, done, truncated, info = env.last()
        state = observation2state(observation['observation'])
        locked_positions = observation['action_mask']
        state_ = np.concatenate((
            observation['observation'].flatten(),
            observation['action_mask'].flatten()
        ))

        if user == 0:

            won1, draw1, lost1 = count_data(done, reward, won1, draw1, lost1)
            action = agent1.make_action(state) if not done else None
            env.step(action)

        elif user == 1:

            won2, draw2, lost2 = count_data(done, reward, won2, draw2, lost2)
            action = agent2.make_action(state_) if not done else None
            env.step(action)

        user = 0 if user == 1 else 1

print(f'player 1')
print(f'won: {(won1/episodes) * 100}')
print(f'draw: {(draw1/episodes) * 100}')
print(f'lost: {(lost1/episodes) * 100}')
print(f'player 2')
print(f'won: {(won2/episodes) * 100}')
print(f'draw: {(draw2/episodes) * 100}')
print(f'lost: {(lost2/episodes) * 100}')
