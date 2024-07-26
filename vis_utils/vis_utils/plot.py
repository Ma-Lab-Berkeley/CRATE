import matplotlib.pyplot as plt
import numpy as np

def cal_sparsity(matrix, is_sparse=False):
    absmatrix = np.abs(matrix)
    #matrix have shape [batch_size, num_patches, dim]
    if is_sparse==True:
        sparsity_list = [np.count_nonzero(absmatrix[i,:,:]==0)/(matrix.shape[1]*matrix.shape[2]) for i in range(matrix.shape[0])]
        sparsity = np.mean(sparsity_list)
        stdev = np.std(sparsity_list)
    else:
        sparsity = None
        stdev = None
    
    return sparsity, stdev


def plot_sparsity(sparsities, std_sparsities):
    fontsize=20
    plt.rcParams["figure.figsize"] = (10, 6)
    cmap = plt.get_cmap('plasma')
    i=0
    cmap_val = [0.3, 0.5]
    epochs = ["val"]
    for mean_sparsity in sparsities:
        std_sparsity = std_sparsities[i]
    
        mean_sparsity = [1 - i for i in mean_sparsity]
        x_labels = np.arange(len(mean_sparsity)) + 1
        if i == 0:
            plt.plot(x_labels, mean_sparsity, marker='s', alpha=0.9, markersize=8, linewidth=2.5,
                    markeredgecolor='black', markeredgewidth=1.0, color='C1', label=f"{epochs[i]}"
                    )
            plt.errorbar(x=x_labels, y=mean_sparsity, yerr=std_sparsity, fmt='none',
                    ecolor='C1', alpha=0.7,
                    capsize=5, capthick=2.0, elinewidth=2.5, zorder=0)
        else:
            plt.plot(x_labels, mean_sparsity, marker='s', alpha=0.6, markersize=8, linewidth=2.5,
                markeredgecolor='black', markeredgewidth=1.0, color='C1', label=f"{epochs[i]}", linestyle='--'
                )
            plt.errorbar(x=x_labels, y=mean_sparsity, yerr=std_sparsity, fmt='none',
                    ecolor='C1', alpha=0.4,
                    capsize=5, capthick=2.0, elinewidth=2.5, zorder=0, linestyle='--')
        i+=1
    plt.title('Measure output sparsity across layers', fontdict={'fontsize': fontsize})
    plt.ylabel(r"Sparsity [ISTA block]", fontdict={'fontsize': fontsize})
    plt.xlabel(r"Layer index - $\ell$", fontdict={'fontsize': fontsize})
    # plt.xticks(x_labels, [f"{i + 1}" for i in range(len(mean))])
    plt.grid(linestyle='--', color='gray')
    plt.legend(fontsize=fontsize, loc='lower left')
    plt.savefig(f"sparsity.pdf", format='pdf', dpi=600)
    plt.close()

def plot_coding_rate(means, std_devs):
    fontsize=20
    plt.rcParams["figure.figsize"] = (10, 6)
    x_labels = np.arange(len(means[0])) + 1
    cmap = plt.get_cmap('viridis')
    i=0
    cmap_val = [0.3, 0.5, 0.9]
    epochs = ["val"]
    for mean_mcr2 in means:
        #set color
        std_mcr2 = std_devs[i]
        if i == 0:
            plt.plot(x_labels, mean_mcr2, marker='o', alpha=0.9, markersize=10, linewidth=2.5,
                        markeredgecolor='black', markeredgewidth=1.0, color='C0', label=f"{epochs[i]}"
                        )
            plt.errorbar(x=x_labels, y=mean_mcr2, yerr=std_mcr2, fmt='none',
                    ecolor='C0', alpha=0.7,
                    capsize=5, capthick=2.0, elinewidth=2.5, zorder=0)
        if i==1:
            plt.plot(x_labels, mean_mcr2, marker='o', alpha=0.6, markersize=10, linewidth=2.5,
                markeredgecolor='black', markeredgewidth=1.0, color='C0', label=f"{epochs[i]}", linestyle='--'
                )
            plt.errorbar(x=x_labels, y=mean_mcr2, yerr=std_mcr2, fmt='none',
                    ecolor='C0', alpha=0.4,
                    capsize=5, capthick=2.0, elinewidth=2.5, zorder=0, linestyle='--')
        i+=1
    plt.legend(fontsize=fontsize, loc='lower left')
    plt.title('Measure coding rate across layers', fontdict={'fontsize': fontsize})
    plt.ylabel(r"$R^c(Z^{\ell})$ [SSA block]", fontdict={'fontsize': fontsize})
    plt.xlabel(r"Layer index - $\ell$", fontdict={'fontsize': fontsize})
    plt.grid(linestyle='--', color='gray')
    plt.savefig(f"mcr2.pdf", format='pdf', dpi=600)
    plt.close()