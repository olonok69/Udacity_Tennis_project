from unityagents import UnityEnvironment
from src.magent import MAD4PG , train_mad4pg, play_agent
from src.utils import *
from src.hyper import hp_tuning
import argparse
import pickle

def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("CC")
    parser.add_argument("--mode", type=str, help="training , play")

    args = parser.parse_args()
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":
        if args.mode == "training":
            file_name = "./Tennis_Windows_x86_64/Tennis.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  True, True)
        elif args.mode == "play":
            file_name = "./Tennis_Windows_x86_64/Tennis.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  False, False)


    # Hyper-parameters

    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    REWARD_STEPS = 1  # steps for rewards of consecutive state action pairs
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    N_ATOMS = 51
    Vmax = 1
    Vmin = -1
    TAU = 1e-3  # for soft update of target parameters
    LEARN_EVERY_STEP = 10
    LEARN_REPEAT = 1
    STATE_SIZE = 24
    ACTION_SIZE = 2
    SEED =2

    agent = MAD4PG(STATE_SIZE,
                    ACTION_SIZE,
                    SEED,
                    LR_ACTOR,
                    LR_CRITIC,
                    GAMMA,
                    REWARD_STEPS,
                    BUFFER_SIZE,
                    BATCH_SIZE,
                    N_ATOMS,
                    Vmax,
                    Vmin,
                    TAU,
                    LEARN_EVERY_STEP,
                    LEARN_REPEAT)

    if args.mode == 'training':

        epi_scores = train_mad4pg(agent, n_agents=2, n_episodes=50000, env=env, brain_name=brain_name)

        epi_scores = np.array(epi_scores)
        plot_scores(epi_scores)
        np.save('./outputs/epi_scores.npy', epi_scores)
        env.close()

    elif args.mode == 'play':

        play_agent(agent)
        for i in range(3):  # play game for 5 episodes
            env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
            states = env_info.vector_observations  # get the current state (for each agent)
            num_agents = len(env_info.agents)
            scores = np.zeros(num_agents)  # initialize the score (for each agent)
            while True:
                actions = agent.acts(states, mode='test')
                env_info = env.step(actions)[brain_name]  # send all actions to tne environment
                next_states = env_info.vector_observations  # get next state (for each agent)
                rewards = env_info.rewards  # get reward (for each agent)
                dones = env_info.local_done  # see if episode finished
                scores += env_info.rewards  # update the score (for each agent)
                states = next_states  # roll over states to next time step
                if np.any(dones):  # exit loop if episode finished
                    break
            print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

        env.close()
    elif args.mode == "hp_tuning":  # hyper parameter tuning
        # hyper parameter tuning DDPG agent
        file_name = "./Tennis_Windows_x86_64/Tennis.exe"
        best_params, trials = hp_tuning(file_name)
        print(best_params)
        with open("outputs/trials.pickle", 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    main()
