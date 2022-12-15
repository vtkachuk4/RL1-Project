import numpy as np
import csv
import gym
from agent import Agent
from itertools import count
import torch as T

from FTA import FTA
from torch.nn.functional import relu

SEED = np.random.randint(0, 100)
RANDOM_NUMBER = np.random.randint(1, 100)


def create_checkpoint():
    pass

def evaluation(agent, seed=1):
    env = gym.make("LunarLander-v2")
    observation = env.reset()
    state = T.tensor(np.asarray([observation[0]]), device=agent.device)
    score = 0

    for step in count():
        action = agent.choose_action(state, greedy=True)
        next_state, reward, done, truncated, _ = env.step(action.item())
        score += reward
        reward = T.tensor(np.asarray([reward]), device=agent.device, dtype=T.float32)
        next_state = T.tensor(np.asarray([next_state]), device=agent.device)
        state = next_state


        if done or truncated:
            return score

def main(run_i=0, _use_target = False, activation = "fta", _fta_lower_limit = -1, _fta_upper_limit = 1, 
        _fta_delta = 0.1, _fta_eta = 0.1, _normalizer = "tanh", _device="cuda:0"):
    env = gym.make("Cartpole-v1") #gym.make("LunarLander-v2")
    
    for run_c in range(20):
        # Activation params
        if activation == "fta":
            _activation = FTA(_fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)
        elif activation == "relu":
            _activation = relu
        else: 
            print("Please specify a valid activation: fta or relu")
            return

        # Activation params
        if activation == "fta":
            _activation = FTA(_fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)
        elif activation == "relu":
            _activation = relu
        else: 
            print("Please specify a valid activation: fta or relu")
            return

        # Multi-test run
        run_i = run_c + 10

        # Agent init
        _gamma = 0.99
        _epsilon = 0.1
        _batch_size = 64
        _n_actions = 2 #4
        _eps_end = 0.1
        _input_dims = 4 #8
        _lr = 0.0001
        _seed = run_i*10
        _large_expansion_factor = 0 #THIS IS FOR -20,20 initial comparison between FTA and DQN-LARGE, k = expansion factor = 20
        _scaling = abs(_fta_upper_limit - _fta_lower_limit) # abs(u-l)
        agent = Agent(
            gamma=_gamma,
            epsilon=_epsilon,
            batch_size=_batch_size,
            n_actions=_n_actions,
            large_expansion_factor = _large_expansion_factor,
            normalizer = _normalizer,
            scaling = _scaling,
            activation=_activation,
            device=_device,
            use_target = _use_target,
            eps_end=_eps_end,
            input_dims=_input_dims,
            lr=_lr,
            seed=_seed #SEED
        )

        # data collection and initialization
        # run from root folder!
        if activation == "fta":
            output_name = f"data/{activation}_t{_use_target}_n{_normalizer}_u{_fta_upper_limit}_d{_fta_delta}_l{_lr}_{run_i}.csv" 
        else:
            output_name = f"data/{activation}_t{_use_target}_l{_lr}_{run_i}.csv"
        with open(output_name, 'w+', newline = '') as csvfile:
            logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            logger.writerow(["fta params:","lower","upper","delta","eta"])
            logger.writerow(["", str(_fta_lower_limit), str(_fta_upper_limit), str(_fta_delta), str(_fta_eta)])
            logger.writerow(["agent params:","gamma","epsilon","batch_size","n_actions","eps_end","use_target","input_dims","lr","seed"])
            logger.writerow(["", str(_gamma), str(_epsilon), str(_batch_size), str(_n_actions), str(_eps_end), str(_use_target), str(_input_dims), str(_lr), str(_seed)])
            logger.writerow(["Performance logging begins:"])
            logger.writerow(["Episode","Step","Score"])

        # Loop    
        scores = []
        policy_scores = []
        max_steps = 500000
        total_timesteps = 0
        for episode in count():

            score = 0
            observation = env.reset()
            state = T.tensor(np.asarray([observation[0]]), device=agent.device)
            for step in count():
                total_timesteps += 1

                action = agent.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action.item())
                score += reward
                reward = T.tensor(np.asarray([reward]), device=agent.device, dtype=T.float32)

                if not done:
                    next_state = T.tensor(np.asarray([next_state]), device=agent.device)
                else:
                    next_state = None
                
                agent.memory.push(state, action, next_state, reward)
                state = next_state

                if total_timesteps >= 5000:
                    agent.learn()

                    if total_timesteps % 1000 == 0:
                        if agent.use_target:
                            agent.targetNetwork.load_state_dict(agent.QNetwork.state_dict())
                        policy_score = evaluation(agent)
                        # if policy_score > 200:
                        #     T.save(agent.QNetwork.state_dict(), f"model{policy_score:.2f}.pth")
                        # print(f"greedy evaluation: {policy_score:.2f}")
                        policy_scores.append((policy_score, total_timesteps))

                        with open(output_name, 'a', newline = '') as csvfile:
                            logger = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                            logger.writerow([str(episode) , str(total_timesteps), str(policy_score)])
                        
                if done or truncated:
                    break

            # scores.append(score)
            # avg_score = np.mean(scores[-100:])
            # print(f"{episode} steps {total_timesteps}, score {score:.2f}, average score {avg_score:.2f}, epsilon {agent.epsilon:.2f}")
            if total_timesteps > max_steps:
            #     fname = f"target-{str(SEED)}-{str(RANDOM_NUMBER)}.npy"
            #     np.save(fname, policy_scores)
                break

if __name__ == "__main__":
    main()
