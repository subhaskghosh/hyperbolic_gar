import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.integrate import solve_ivp

dpi = 300
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (8, 6)


def plot_accuracy(dataframe, dataset, plot_type='accuracy', x_axis=r'$ m $', y_axis='accuracy'):
    """
    Plots accuracy curves using seaborn's FacetGrid.
    """
    # Reset index to avoid duplicate index errors
    dataframe = dataframe.reset_index(drop=True)
    sns.set_style("white")
    g = sns.FacetGrid(
        dataframe[dataframe['dataset'] == dataset],
        col=r'$ \beta $',
        hue="Aggregation Method",
        legend_out=True,
        col_wrap=4, height=3.5,
        palette=sns.color_palette(["#38BDF2", "#F29544", "#44803F", "#323673", "#F2055C", "#0583F2"])
    )
    g.map(sns.lineplot, x_axis, y_axis)
    g.set_axis_labels(x_axis, y_axis)
    sns.set_theme(rc={'figure.figsize': (12, 12)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(size=20)

    os.makedirs("plots_pdf", exist_ok=True)
    plt.savefig(f"plots_pdf/{plot_type}_plot_{dataset}.pdf", dpi=dpi)
    plt.close()


def plot(dataframe, dataset, plot_type='loss', x_axis='Epoch', y_axis='Training Loss', row_param=r'$ \sigma^2 $'):
    """
    Plots loss curves using seaborn's FacetGrid.
    """
    # Reset index to avoid duplicate index issues
    dataframe = dataframe.reset_index(drop=True)
    sns.set_style("white")
    g = sns.FacetGrid(
        dataframe[dataframe['dataset'] == dataset],
        col=row_param if row_param else None,
        row=r'$ \beta $',
        hue="Aggregation Method",
        legend_out=True,
        palette='Spectral'
    )
    g.map(sns.lineplot, x_axis, y_axis)
    g.set_axis_labels(x_axis, y_axis)
    sns.set_theme(rc={'figure.figsize': (12, 8)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(size=20)

    os.makedirs("plots_pdf", exist_ok=True)
    plt.savefig(f"plots_pdf/{plot_type}_plot_{dataset}.pdf", dpi=dpi)
    plt.close()


# Read and preprocess results
results_df = pd.read_csv("results_1.csv")
# Remove rows corresponding to ipm_attack
results_df = results_df[results_df['attack_mode'] != 'ipm_attack']
# Rename columns for consistency
results_df = results_df.rename(columns={'samples_per_node': r'$ m $',
                                        'byzantine_fraction': r'$ \beta $',
                                        'noise_variance': r'$ \sigma^2 $'})
# Melt the dataframe to create a long-form dataframe for plotting accuracy curves
melted_df = results_df.melt(
    id_vars=['dataset', r'$ \beta $', r'$ \sigma^2 $', 'attack_mode', r'$ m $'],
    value_vars=[col for col in results_df.columns if 'accuracy' in col],
    var_name='accuracy_type',
    value_name='accuracy'
)
# Extract aggregation method from the accuracy_type column and rename for clarity.
melted_df['Aggregation Method'] = (
    melted_df['accuracy_type'].str.split('_').str[0]
    .replace({'noisy': 'mean', 'noiseless': 'mean without noise',
              'hyperbolic': 'hyperbolic median', 'median': 'co-ordinate wise median'})
)

# Plot accuracy curves for each dataset
datasets = ["spambase", "mnist", "fashion_mnist", "cifar10"]
for dataset in datasets:
     plot_accuracy(melted_df, dataset)
     plot_accuracy(melted_df, dataset, plot_type='accuracy_var', x_axis=r'$ \sigma^2 $', y_axis='accuracy')

# Plot loss curves:
for dataset in datasets:
    merged_main_loss_df = pd.DataFrame()
    for byz_frac in [0.1, 0.2, 0.3, 0.4]:
        for noise_var in [1.0, 10.0, 100.0, 200.0]:
            for samples_per_node in [16, 32, 64, 128]:
                loss_file = f"loss_curves_1/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                loss_file_hk = f"loss_curves_hk/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                if os.path.exists(loss_file):
                    loss_df = pd.read_csv(loss_file)
                    loss_df = loss_df.rename(columns={'epoch': 'Epoch','loss': 'Training Loss'})
                    if os.path.exists(loss_file_hk):
                        loss_hk_df = pd.read_csv(loss_file_hk)
                        loss_hk_df = loss_hk_df.rename(
                            columns={'epoch': 'Epoch', 'hyperbolic_krum_loss' : 'hk_loss'})
                        
                        loss_df = loss_df.merge(loss_hk_df, on=['Epoch'], how='outer')
                        


                    loss_melted_df = loss_df.melt(
                        id_vars=['Epoch'],
                        value_vars=[col for col in loss_df.columns if 'loss' in col],
                        var_name='loss_type',
                        value_name='Training Loss'
                    )
                    loss_melted_df['Aggregation Method'] = (
                        loss_melted_df['loss_type'].str.split('_').str[0]
                        .replace({'noisy': 'mean', 'noiseless': 'mean without noise',
                                  'hyperbolic': 'hyperbolic median', 'median': 'co-ordinate wise median', 'hk' : 'hyperbolic krum'})
                    )
                    loss_melted_df[r'$ \sigma^2 $'] = noise_var
                    loss_melted_df[r'$ \beta $'] = byz_frac
                    loss_melted_df[r'$ m $'] = samples_per_node
                    loss_melted_df['dataset'] = dataset
                    merged_main_loss_df = pd.concat([merged_main_loss_df, loss_melted_df], ignore_index=True)
    merged_main_loss_df = merged_main_loss_df.reset_index(drop=True)
    plot(merged_main_loss_df, dataset)
    plot(merged_main_loss_df, dataset, plot_type='loss_sample', x_axis='Epoch', y_axis='Training Loss', row_param=r'$ m $')
