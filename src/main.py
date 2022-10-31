import numpy as np
import gym
from agent import Agent
from FTA import FTA
from torch.nn.functional import relu
import csv


def main():
    # create gym environment
    env = gym.make("LunarLander-v2")

    # create FTA module and its parameters (# value from paper):
    _fta_lower_limit = -20  # -20
    _fta_upper_limit = 20  # 20
    _fta_delta = 2  # 2
    _fta_eta = 2  # 2
    fta = FTA(_fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta)
    print(type(fta).__name__)

    # agent parameters (# original value)
    _gamma = 0.99
    _epsilon = 1.0
    _batch_size = 256
    _n_actions = 4
    _eps_end = 0.01
    _input_dims = [8]
    _lr = 0.0003
    _seed = 0
    _activation = relu

    # game parameters and score keeping
    n_games = 1000
    horizon = 200

    # data collection and initialization
    # output_name = "output/test.csv"  # run from root folder!

    data_dir = "data/"
    run_settings = [
        {"activation": fta, "outfile_name": "fta_scores"},
        {"activation": relu, "outfile_name": "relu_scores"},
    ]

    for run_setting in run_settings:
        scores, eps_history = [], []
        print(f'==========Running ativation: {run_setting["activation"]}==========')

        with open(
            data_dir + run_setting["outfile_name"] + ".csv", "w+", newline=""
        ) as csvfile:
            logger = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            # logger.writerow(["fta params:", "lower", "upper", "delta", "eta"])
            # logger.writerow(
            #     [
            #         "",
            #         str(_fta_lower_limit),
            #         str(_fta_upper_limit),
            #         str(_fta_delta),
            #         str(_fta_eta),
            #     ]
            # )
            logger.writerow(
                [
                    "activation:",
                    "agent params:",
                    "gamma",
                    "epsilon",
                    "batch_size",
                    "n_actions",
                    "eps_end",
                    "input_dims",
                    "lr",
                    "seed",
                ]
            )
            logger.writerow(
                [
                    "",
                    str(run_setting["activation"]),
                    str(_gamma),
                    str(_epsilon),
                    str(_n_actions),
                    str(_eps_end),
                    str(*_input_dims),
                    str(_lr),
                    str(_seed),
                ]
            )
            logger.writerow(["Performance logging begins:"])
            logger.writerow(["Episode:,Score:,Average Score:,Epsilon:"])

        agent = Agent(
            gamma=_gamma,
            epsilon=_epsilon,
            batch_size=_batch_size,
            n_actions=_n_actions,
            activation=run_setting["activation"],
            eps_end=_eps_end,
            input_dims=_input_dims,
            lr=_lr,
            seed=_seed,
        )

        for i in range(n_games):
            score = 0
            done = False
            truncated = False
            observation = env.reset()
            observation = np.array(observation[0])
            step = 0
            while not done and not truncated:
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
                step += 1
            scores.append(score)
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])
            with open(
                data_dir + run_setting["outfile_name"] + ".csv", "a", newline=""
            ) as csvfile:
                logger = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
                logger.writerow(
                    [
                        str(i),
                        str(round(score, 2)),
                        str(round(avg_score, 2)),
                        str(round(agent.epsilon, 2)),
                    ]
                )

            print(
                "episode ",
                i,
                "score %.2f" % score,
                "average score %.2f" % avg_score,
                "epsilon %.2f" % agent.epsilon,
            )

        # outfile_scores = "output/relu_scores.npy"
        np.save(data_dir + run_setting["outfile_name"] + ".npy", np.array(scores))


if __name__ == "__main__":
    main()
