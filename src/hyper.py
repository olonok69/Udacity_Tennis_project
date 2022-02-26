from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from functools import partial
from src.utils import *
from src.magent import MAD4PG



num_episodes =15000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHECKPOINT_FOLDER = './models/'
algo="1"
def evaluate_model(hyperopt_params, env):

    LR_ACTOR = hyperopt_params['lr_actor']
    LR_CRITIC = hyperopt_params['lr_critic']
    GAMMA = hyperopt_params['gamma']
    REWARD_STEPS = hyperopt_params['rewards_steos']
    BUFFER_SIZE = hyperopt_params['buffer_size']
    BATCH_SIZE = hyperopt_params['batch_size']
    N_ATOMS = hyperopt_params['natoms']
    Vmax = hyperopt_params['vmax']
    Vmin = hyperopt_params['vmin']
    TAU = hyperopt_params['tau']  # for soft update of target parameters
    LEARN_EVERY_STEP = hyperopt_params['learn_every_step']
    LEARN_REPEAT = hyperopt_params['learn_repeat']
    STATE_SIZE = hyperopt_params['state_size']
    ACTION_SIZE = hyperopt_params['action_size']
    SEED = 2
    brain_name = hyperopt_params['brain_name']


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


    n_episodes = num_episodes



    epi_scores = []
    scores_window = deque(maxlen=100)
    i_episode = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents)
        while True:
            actions = agent.acts(states, mode='train')  # (n_agents, action_size)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards  # (n_agents,)
            dones = env_info.local_done  # (n_agents,)

            agent.step(states, actions, rewards, next_states, dones)

            scores += env_info.rewards  # (n_agents,)
            states = next_states  # (n_agents, state_size)
            if np.any(dones):
                break
        scores_window.append(np.max(scores))
        epi_scores.append(np.max(scores))
        # print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}'.format(
        #     i_episode, np.mean(scores_window), len(agent.memory)), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {:>4}\tAverage Score:{:>6.3f}\tMemory Size:{:>5}'.format(
                i_episode, np.mean(scores_window), len(agent.memory)))
        if np.mean(scores_window) > 0.5:
            break

    reward = np.mean(scores_window)

    return {'loss': -reward, 'status': STATUS_OK, 'nepisodes': i_episode}

def objective(params, env):
    output = evaluate_model(params, env)
    return {'loss': output['loss'] ,  'status': output['status']}

def hp_tuning(file):

    file_name="./Tennis_Windows_x86_64/Tennis.exe"
    worker_id = 1
    base_port = 5005
    env, brain_name, brain, action_size, env_info, state, state_size = load_env(worker_id,
                                                                                          base_port, file, True,
                                                                                          True)

    # define search Space
    search_space = { 'gamma': hp.loguniform('gamma' ,np.log(0.9), np.log(0.99)),
                    'batch_size' : hp.choice('batch_size', [32,64, 128]),
                     'lr_actor': hp.loguniform('lr_actor',np.log(3e-4), np.log(15e-3)),
                     'lr_critic': hp.loguniform('lr_critic', np.log(1e-4), np.log(15e-3)),
                     'brain_name' : brain_name,
                     'state_size' : state_size,
                     'action_size' : action_size,
                     'rewards_steos' : 1,
                     'buffer_size' : 100000,
                     'natoms' : 51,
                     'vmax': 1,
                     'vmin' : -1,
                     'tau': 1e-3,
                     'learn_every_step' : 10,
                     'learn_repeat' : 1,

                               }



    # send the env with partial as additional env
    fmin_objective = partial(objective, env=env)
    trials = Trials()
    argmin = fmin(
        fn=fmin_objective,
        space=search_space,
        algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
        max_evals=40,
        trials=trials,
        verbose=True
        )#
    # return the best parameters
    best_parms = space_eval(search_space, argmin)
    return best_parms, trials