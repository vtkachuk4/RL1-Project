import numpy as np
import gym
from agent import Agent
from itertools import count
import torch as T

SEED = np.random.randint(0, 100)
RANDOM_NUMBER = np.random.randint(1, 100)


def create_checkpoint():
    pass

def evaluation(agent, seed=1):
    env = gym.make("LunarLander-v2")
    observation = env.reset(seed=seed)
    state = T.tensor([observation[0]], device=agent.device)
    score = 0

    for step in count():
        action = agent.choose_action(state, greedy=True)
        next_state, reward, done, truncated, _ = env.step(action.item())
        score += reward
        reward = T.tensor([reward], device=agent.device, dtype=T.float32)
        next_state = T.tensor([next_state], device=agent.device)
        state = next_state


        if done or truncated:
            return score

def main():
    env = gym.make("LunarLander-v2")
    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=8,
        lr=0.0003,
        seed=SEED,
    )

    scores = []
    policy_scores = []
    max_steps = 500000
    total_timesteps = 0
    for episode in count():

        score = 0
        observation = env.reset(seed=SEED)
        state = T.tensor([observation[0]], device=agent.device)
        for step in count():
            total_timesteps += 1

            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action.item())
            score += reward
            reward = T.tensor([reward], device=agent.device, dtype=T.float32)

            if not done:
                next_state = T.tensor([next_state], device=agent.device)
            else:
                next_state = None
            
            agent.memory.push(state, action, next_state, reward)
            state = next_state

            if total_timesteps > 5000:
                agent.learn()

                if total_timesteps % 1000 == 0:
                    agent.targetNetwork.load_state_dict(agent.QNetwork.state_dict())
                    policy_score = evaluation(agent, seed=SEED)
                    if policy_score > 200:
                        T.save(agent.QNetwork.state_dict(), f"model{policy_score:.2f}.pth")
                    print(f"greedy evaluation: {policy_score:.2f}")
                    policy_scores.append((policy_score, total_timesteps))
                    
            if done or truncated:
                break

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"{episode} steps {total_timesteps}, score {score:.2f}, average score {avg_score:.2f}, epsilon {agent.epsilon:.2f}")

        if total_timesteps > max_steps:
            fname = f"target-{str(SEED)}-{str(RANDOM_NUMBER)}.npy"
            np.save(fname, policy_scores)
            break

if __name__ == "__main__":
    main()
