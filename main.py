import os
from src.magent import multi_agent_d4pg , train_mad4pg, play_agent
from src.utils import *
from src.hyper import hp_tuning
import argparse
import pickle
import time

def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("CC")
    parser.add_argument("--mode", type=str, help="training , play, hp_tunning",  required=True)
    parser.add_argument("--nagents", type=str, help="number of agents , multi-agent",
                        required=True)
    parser.add_argument("--af", type=str, help="Activation Function: 1--> Leaky Relu or 2--> Relu",
                        required=True)

    args = parser.parse_args()
    if int(args.nagents) < 2:
        print("Number of Agents must be higher than 1")
        return
    if not(int(args.af) in [1,2]):
        print("Activation Function must be 1--> Leaky Relu or 2--> Relu")
        return
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":
        file_name = "./Tennis_Windows_x86_64/Tennis.exe"
        worker_id = 1
        base_port = 5005
        if args.mode == "training":

            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  True, True)
        elif args.mode == "play":

            env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  False, False)

    # dictionary to serialize metrics Algos for reporting and plotting
    fname = "outputs/outcomes.pkl"
    if os.path.isfile(fname):
        with open(fname, 'rb') as handle:
            outputs = pickle.load(handle)
    else:
        outputs = {}
    # Hyper-parameters

    LR_ACTOR = 1e-4 # actor learning rate
    LR_CRITIC = 1e-3 # critic learning rate
    GAMMA = 0.99 # gamma  discount factor
    REWARD_STEPS = 1  # steps for rewards of consecutive state action pairs
    BUFFER_SIZE = 100000 # size of buffer
    BATCH_SIZE = 64 # default batch size
    N_ATOMS = 51 # slices for categorical distributions , same than in paper
    Vmax = 1 # max value accumulated discounted reward use in categorical distribution approximation
    Vmin = -1 # min value accumulated discounted reward  use in categorical distribution approximation
    TAU = 1e-3  # for soft update of target parameters
    LEARN_EVERY_STEP = 10
    LEARN_REPEAT = 1
    STATE_SIZE = 24 # observation space
    ACTION_SIZE = 2 # action Space
    SEED =2 # seed

    nagents = int(args.nagents)  # Number of Agents
    activation_function = int(args.af) # --mode


    agent = multi_agent_d4pg(STATE_SIZE,
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
                             LEARN_REPEAT,
                             nagents,
                             activation_function)

    if args.mode == 'training':
        time1 = time.time()
        # train agent
        epi_scores, actor_loss, critic_loss = train_mad4pg(agent, n_agents=2, af=activation_function, magents= nagents,
                                                           n_episodes=50000, env=env, brain_name=brain_name)

        epi_scores = np.array(epi_scores)
        # plot scores
        plot_scores(epi_scores, nagents,
                             activation_function)
        # plot critic loss
        plot_critic_loss(critic_loss, nagents,
                             activation_function)
        # plot actor loss
        plot_actor_loss(actor_loss, nagents,
                             activation_function)
        # save time total took this algo
        time2 = time.time()
        times = time2 - time1
        # loss, score, time to dictionary and serialize
        mode = f"agent_{nagents}_{activation_function}"
        if not (mode in outputs.keys()):
            outputs[str(mode)] = {}
        outputs[str(mode)]['scores'] = epi_scores
        outputs[str(mode)]['time'] = times
        outputs[str(mode)]['actor_loss'] = actor_loss
        outputs[str(mode)]['critic_loss'] = critic_loss
        with open(fname, 'wb') as handle:
            pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        env.close()
    elif args.mode == 'play':

        play_agent(agent, nagents,
                             activation_function)
        mode = f"agent_{nagents}_{activation_function}"
        # list for save scores
        score_play = []
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
            score_play.append(np.mean(scores))
        if len(score_play) > 0:
            outputs[str(mode)]['score_play'] = np.mean(score_play)

            with open(fname, 'wb') as handle:
                pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        env.close()
    elif args.mode == "hp_tuning":  # hyper parameter tuning
        # hyper parameter tuning DDPG agent
        file_name = "./Tennis_Windows_x86_64/Tennis.exe"
        best_params, trials = hp_tuning(file_name)
        print(best_params)
        with open("outputs/trials.pickle", 'wb') as handle:
            pickle.dump(trials, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif args.mode == 'plot': # plot comparision all algos
        labels = plot_scores_training_all()
        plot_time_all(labels)
        plot_number_episodes(labels)
        plot_play_scores(labels)

if __name__ == '__main__':
    main()
