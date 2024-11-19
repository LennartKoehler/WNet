import matplotlib.pyplot as plt
import numpy as np


def plot_loss(run_name):
    data_rec_path = f"results/{run_name}/rec_losses.npy"
    data_cut_path = f"results/{run_name}/n_cut_losses.npy"

    data_rec = np.load(data_rec_path)
    data_cut = np.load(data_cut_path)

    plt.plot(data_cut)
    plt.savefig(data_cut_path[:-4])
    plt.close()
    plt.plot(data_rec)
    plt.savefig(data_rec_path[:-4])

if __name__ == "__main__":
    run_name = "conv_rna_004_8_epochs_01_dropout"
    plot_loss(run_name)
