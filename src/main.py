import numpy as np
import gym
from agent import Agent


def main():
    env = gym.make("LunarLander-v2")
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=0.0003,
        seed=1,
    )
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        observation = np.array(observation[0])
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )


if __name__ == "__main__":
    main()
