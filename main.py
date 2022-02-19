from unityagents import UnityEnvironment
from mad4pg_agent import MAD4PG   , train_mad4pg
from utils import *

env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = MAD4PG(state_size=24, action_size=2, seed=199)

epi_scores = train_mad4pg(agent, n_agents=2, n_episodes=10000, env=env, brain_name=brain_name)

epi_scores = np.array(epi_scores)
np.save('./outputs/epi_scores.npy', epi_scores)

epi_scores = np.load('./outputs/epi_scores.npy')
plot_scores(epi_scores)