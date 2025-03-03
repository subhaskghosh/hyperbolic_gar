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
    plt.savefig(f"src/plots_pdf/{plot_type}_plot_{dataset}.pdf", dpi=dpi)
    plt.close()

# Plot accuracy curves for each dataset
datasets = ["cifar10"]

# Plot loss curves:
for dataset in datasets:
    merged_main_loss_df = pd.DataFrame()
    for byz_frac in [0.1, 0.2, 0.3]:
        for noise_var in [1.0, 10.0, 100.0, 200.0]:
            for samples_per_node in [16, 32, 64]:
                loss_file = f"src/loss_curves_1/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                loss_file_hk = f"src/loss_curves_hk/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                if os.path.exists(loss_file):
                    loss_df = pd.read_csv(loss_file)
                    loss_df = loss_df.rename(columns={'epoch': 'Epoch','loss': 'Training Loss'})
                    if os.path.exists(loss_file_hk):
                        loss_hk_df = pd.read_csv(loss_file_hk)
                        loss_hk_df = loss_hk_df.rename(
                            columns={'epoch': 'Epoch', 'hyperbolic_krum_loss' : 'hk_loss'})
                        # Merge on Epoch using an outer join to handle missing values
                        loss_df = loss_df.merge(loss_hk_df, on=['Epoch'], how='outer')
                        # If original loss_df had an aggregation method, fill it for consistency


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
    methods = ['mean']
    merged_main_loss_df = merged_main_loss_df.loc[~merged_main_loss_df['Aggregation Method'].isin(methods)]
    plot(merged_main_loss_df, dataset)
    plot(merged_main_loss_df, dataset, plot_type='loss_sample', x_axis='Epoch', y_axis='Training Loss', row_param=r'$ m $')
