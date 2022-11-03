import numpy as np
import gym
from agent import Agent
from FTA import FTA
import csv
from tqdm import tqdm

def main(run_i, _fta_lower_limit = -20, _fta_upper_limit = 20, _fta_delta = 2, _fta_eta = 2):
    # create gym environment
    env = gym.make("LunarLander-v2")

    # create FTA module and its parameters (# value from paper):
    
    fta = FTA(_fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)

    # agent parameters (# original value)
    _gamma=0.99
    _epsilon=1.0
    _batch_size=64 # 256
    _n_actions=4
    _eps_end=0.01
    _input_dims=[8]
    _lr=0.0003
    _seed= run_i*10 # 0

    # game parameters and score keeping
    scores, eps_history = [], []
    n_games = 500
    max_steps_per_episode = 500

    # data collection and initialization
    output_name = f"output/fta_u{_fta_upper_limit}_d{_fta_delta}_{run_i}.csv" # run from root folder!
    with open(output_name, 'w+', newline = '') as csvfile:
        logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        logger.writerow(["fta params:","lower","upper","delta","eta"])
        logger.writerow(["", str(_fta_lower_limit) , str(_fta_upper_limit) , str(_fta_delta) , str(_fta_eta)])
        logger.writerow(["agent params:","gamma","epsilon","batch_size","n_actions","eps_end","input_dims","lr","seed"])
        logger.writerow(["" , str(_gamma) , str(_epsilon) , str(_n_actions) , str(_eps_end) , str(*_input_dims) , str(_lr) , str(_seed)])
        logger.writerow(["Performance logging begins:"])
        logger.writerow(["Episode:","Score:","Average Score:","Epsilon:"])

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

        steps = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            steps = steps + 1
            if steps >= max_steps_per_episode:
                break

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        with open(output_name, 'a', newline = '') as csvfile:
            logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            logger.writerow([str(i) , str(round(score, 2)) , str(round(avg_score, 2)) , str(round(agent.epsilon, 2))])
        
        # print(
        #     "episode ",
        #     i,
        #     "score %.2f" % score,
        #     "average score %.2f" % avg_score,
        #     "epsilon %.2f" % agent.epsilon,
        #     "steps %.2f" %steps
        # )


if __name__ == "__main__":
    fta_params_dict = {
        "fta_upper_limit": [0.05, 0.1, 0.25, 0.5, 1, 2],
        "num_tiles": [10, 11, 12, 13, 14, 15]
    }
    num_runs = 5

    for _fta_upper_limit in fta_params_dict["fta_upper_limit"]:
        _fta_lower_limit = -1*_fta_upper_limit
        for num_tiles in fta_params_dict["num_tiles"]:
            _fta_delta = (2*_fta_upper_limit)/num_tiles
            _fta_eta = _fta_delta
            
            for run_i in range(num_runs):
                print(
                "fta_upper_limit: %.2f  " % _fta_upper_limit,
                "fta_lower_limit: %.2f  " % _fta_lower_limit,
                "fta_delta: %.4f  " % _fta_delta,
                "fta_eta: %.4f  " % _fta_eta,
                "run: ", run_i
                )
                main(run_i, _fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)
     
