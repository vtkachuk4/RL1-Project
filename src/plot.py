import numpy as np
import matplotlib.pyplot as plt


def smooth_data(data, smoothness=20):
    smooth_data = np.zeros((len(data) - smoothness))
    for i in np.arange(len(smooth_data)):
        smooth_data[i] = np.mean(data[i : i + smoothness])
    return smooth_data


def plot(all_scores, plot_settings, save_file, smoothness=20):
    x = np.arange(len(all_scores[0]) - smoothness)
    for i, scores in enumerate(all_scores):
        smooth_scores = smooth_data(scores, smoothness=smoothness)
        plt.plot(x, smooth_scores, label=plot_settings[i]["label"])
    plt.legend(loc="upper left")
    plt.savefig(save_file)


if __name__ == "__main__":
    data_dir = "data/"
    plot_file = data_dir + "DQN_plot.png"
    plot_settings = [
        {"datafile_name": "fta_scores.npy", "label": "fta"},
        {"datafile_name": "relu_scores.npy", "label": "relu"},
    ]

    all_scores = []
    for plot_setting in plot_settings:
        scores = np.load(data_dir + plot_setting["datafile_name"])
        all_scores.append(scores)
    plot(all_scores, plot_settings, save_file=plot_file, smoothness=40)
