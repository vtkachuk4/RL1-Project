import numpy as np
import gym
from agent import Agent
from FTA import FTA
import csv


def main():
    # create gym environment
    env = gym.make("LunarLander-v2")

    # create FTA module and its parameters (# value from paper):
    _fta_lower_limit = -20 # -20
    _fta_upper_limit = 20 # 20
    _fta_delta = 2 # 2
    _fta_eta = 2 # 2
    fta = FTA(_fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)

    # agent parameters (# original value)
    _gamma=0.99
    _epsilon=1.0
    _batch_size=256
    _n_actions=4
    _eps_end=0.01
    _input_dims=[8]
    _lr=0.0003
    _seed=0

    # game parameters and score keeping
    scores, eps_history = [], []
    n_games = 500

    # data collection and initialization
    output_name = "output/test.csv" # run from root folder!
    with open(output_name, 'w+', newline = '') as csvfile:
        logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        logger.writerow(["fta params:","lower","upper","delta","eta"])
        logger.writerow(["", str(_fta_lower_limit) , str(_fta_upper_limit) , str(_fta_delta) , str(_fta_eta)])
        logger.writerow(["agent params:","gamma","epsilon","batch_size","n_actions","eps_end","input_dims","lr","seed"])
        logger.writerow(["" , str(_gamma) , str(_epsilon) , str(_n_actions) , str(_eps_end) , str(*_input_dims) , str(_lr) , str(_seed)])
        logger.writerow(["Performance logging begins:"])
        logger.writerow(["Episode:,Score:,Average Score:,Epsilon:"])

    agent = Agent(
        gamma=_gamma,
        epsilon=_epsilon,
        batch_size=_batch_size,
        n_actions=_n_actions,
        activation = fta,
        eps_end=_eps_end,
        input_dims=_input_dims,
        lr=_lr,
        seed=_seed,
    )

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
        with open(output_name, 'a', newline = '') as csvfile:
            logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            logger.writerow([str(i) , str(round(score, 2)) , str(round(avg_score, 2)) , str(round(agent.epsilon, 2))])
        
        print(
            "episode ",
            i,
            "score %.2f" % score,
            "average score %.2f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )


if __name__ == "__main__":
    main()
