import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_scores(scores, rolling_window=100):
    '''Plot score and its moving average on the same chart.'''

    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(scores)), scores, '-c', label='episode score')
    plt.title('Episodic Score')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(np.arange(len(scores)), rolling_mean, '-y', label='rolling_mean')
    plt.ylabel('score')
    plt.xlabel('episode #')
    plt.legend()