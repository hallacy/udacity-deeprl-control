import json
import os
from collections import deque, namedtuple

import fire
import numpy as np
import torch
from tqdm import tqdm
from unityagents import UnityEnvironment

from ddpg_agent import Agent
from model import Actor, Critic


def train_personal(
    agent,
    env,
    brain_name,
    num_agents,
    n_episodes=1000,
    max_t=1000,
    print_every=10,
    output_folder=".",
):
    scores_deque = deque(maxlen=print_every)
    project_complete_queue = deque(maxlen=100)
    episode_scores = []
    for i_episode in tqdm(list(range(1, n_episodes + 1))):
        env_info = env.reset(train_mode=True)[brain_name]
        scores = np.zeros(num_agents)

        states = env_info.vector_observations
        agent.reset()

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for i_agent in range(num_agents):
                agent.step(
                    states[i_agent],
                    actions[i_agent],
                    rewards[i_agent],
                    next_states[i_agent],
                    dones[i_agent],
                    t,
                )  # update the system

            states = next_states
            scores += rewards
            if np.any(dones):
                break
        episode_scores.append(scores.tolist())
        scores_deque.append(scores)
        project_complete_queue.append(scores)
        print(
            "\rEpisode {}\tAverage Score this ep: {:.2f}".format(
                i_episode, np.mean(scores)
            )
        )
        with open(f"{output_folder}/scores.json", "w") as fp:
            fp.write(json.dumps(episode_scores))

        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )
        if np.mean(project_complete_queue) > 30.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(project_complete_queue)
                )
            )
            torch.save(
                agent.actor_local.state_dict(), f"{output_folder}/checkpoint_actor.pth"
            )
            torch.save(
                agent.critic_local.state_dict(),
                f"{output_folder}/checkpoint_critic.pth",
            )

            break

    return episode_scores


def main(env_path, output_folder=".", n_episodes=1000, max_t=1000, print_every=10):

    env = UnityEnvironment(file_name=env_path)
    os.makedirs(output_folder, exist_ok=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print("Number of agents:", num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print("Size of each action:", action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print(
        "There are {} agents. Each observes a state with length: {}".format(
            states.shape[0], state_size
        )
    )
    print("The state for the first agent looks like:", states[0])

    actor = Actor
    critic = Critic

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        random_seed=2,
        actor=actor,
        critic=critic,
    )

    # train
    train_personal(
        agent,
        env,
        brain_name,
        num_agents,
        output_folder=output_folder,
        n_episodes=n_episodes,
        max_t=max_t,
        print_every=print_every,
    )


if __name__ == "__main__":
    fire.Fire(main)
