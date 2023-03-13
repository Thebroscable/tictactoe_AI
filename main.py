from pettingzoo.classic import tictactoe_v3
from q_learning import Agent as QL
from sarsa_lambda import Agent as SL
from q_lambda import Agent as QLL
from sarsa import Agent as S
from DQN import Agent as DQN
from minmax import findBestMove
import numpy as np
import random


def policy(observation):
    action = random.choice(np.flatnonzero(observation['action_mask']))
    return action


def observation2state(observation):
    vector = observation.flatten()
    state = 0

    for i in range(0, 18, 2):
        if vector[i] ^ vector[i+1]:
            state += (vector[i]*2 + vector[i+1]) * 3**(i//2)

    return state

def minmax_move(observation):
    board = [['_' for i in range(3)] for i in range(3)]

    for i, x in enumerate(observation):
        for j, y in enumerate(x):
            value = y[0]*2 + y[1]
            if value == 2:
                board[j][i] = 'x'
            elif value == 1:
                board[j][i] = 'o'

    best_move = findBestMove(board)
    action = best_move[1]*3 + best_move[0]
    return action


env = tictactoe_v3.env()
# agent = S(n_states=3**9, n_actions=9, alpha=0.01, gamma=0.99, epsilon=0.01)
agent = DQN(input_dim=(27,), n_actions=9, lr=0.01, gamma=0.99, epsilon_end=0.01, epsilon=0.0)
agent.load_model('model/dqn_weights.h5')
episodes = 100
user = 1
won = 0
draw = 0
lost = 0

for episode in range(episodes):
    env.reset()
    state = None
    print('--------*episode: {}*--------'.format(episode+1))

    for _ in env.agent_iter():
        observation, reward, done, truncated, info = env.last()
        state_ = np.concatenate((
            observation['observation'].flatten(),
            observation['action_mask'].flatten()
        ))
        # state_ = observation2state(observation['observation'])
        # locked_positions = observation['action_mask']
        if user == 0:
            _action = agent.make_action(state_) if not done else None
            if state is not None:
                agent.store_data(state, action, reward, state_, done)
            if done:
                if reward == 1:
                    won += 1
                elif reward == 0:
                    draw += 1
                elif reward == -1:
                    lost += 1
            state = state_
            action = _action
            # agent.train(64)
            env.step(action)
        elif user == 1:
            # action_p = policy(observation) if not done else None
            action_p = minmax_move(observation['observation']) if not done else None
            env.step(action_p)
        user = 0 if user == 1 else 1

print(f'won: {(won/episodes) * 100}')
print(f'draw: {(draw/episodes) * 100}')
print(f'lost: {(lost/episodes) * 100}')
# agent.save_model(model_file='model/dqn_weights2.h5')
