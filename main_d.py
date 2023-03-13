from pettingzoo.classic import tictactoe_v3
from DQN import Agent
from minmax import findBestMove
import numpy as np
import random


def policy(observation):
    # action = policy(observation) if not done else None
    action = random.choice(np.flatnonzero(observation['action_mask']))
    return action


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


def fl(state):
    state = np.array(state)
    state = state.flatten()
    return state


env = tictactoe_v3.env()
agent = Agent(input_dim=18, n_actions=9, lr=0.0005, gamma=0.99)
episodes = 2000
user = 1
scores = []
d_a = d_r = d_d = None

for episode in range(episodes):
    env.reset()

    for _ in env.agent_iter():
        observation, reward_, done_, truncated, info = env.last()

        if user == 0:
            state_ = fl(observation['observation'])
            locked_positions = observation['action_mask']

            if (locked_positions == 0).sum() > 1:
                agent.store_data(state, action, reward_, state_, done_)
            agent.train(32)

            action = agent.make_action(state_) if not done_ else None
            env.step(action)

            state = state_

            if done_:
                scores.append(reward_)

        elif user == 1:
            # action_ = int(input()) if not done else None
            action_ = policy(observation) if not done_ else None
            # action_ = minmax_move(observation['observation']) if not done else None
            env.step(action_)

        user = 0 if user == 1 else 1
        # env.render()

    print(f'episode: {episode+1}, won: {scores.count(1)}, draw: {scores.count(0)}')

agent.save_model()
