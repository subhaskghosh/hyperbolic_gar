import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import seaborn as sns

dpi = 300
rc('text', usetex=True)
plt.style.use('seaborn-v0_8-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-v0_8-ticks")
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
        margin_titles=True,
        col_wrap=3, height=3.5,
        palette='bright'
    )
    g.map(sns.lineplot, x_axis, y_axis, errorbar='sd')
    g.set_axis_labels(x_axis, y_axis, size=18)
    sns.set_theme(rc={'figure.figsize': (12, 12)})
    g.set_titles(size=20)
    g.add_legend(adjust_subtitles=True)
    os.makedirs("src/plots_pdf", exist_ok=True)
    plt.savefig(f"src/plots_pdf/{plot_type}_plot_{dataset}.pdf", dpi=dpi)
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
        row=row_param if row_param else None,
        col=r'$ \beta $',
        hue="Aggregation Method",
        legend_out=True,
        palette='bright'
    )
    g.map(sns.lineplot, x_axis, y_axis, errorbar='sd')
    g.set_axis_labels(x_axis, y_axis)
    sns.set_theme(rc={'figure.figsize': (12, 8)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(size=20)

    os.makedirs("src/plots_pdf", exist_ok=True)
    plt.savefig(f"src/plots_pdf/{plot_type}_plot_{dataset}.pdf", dpi=dpi)
    plt.close()


# Plot accuracy curves for each dataset
datasets = ["cifar10"]
# Read and preprocess results
results_df = pd.read_csv("src/results_1.csv")
results_hk_df = pd.read_csv("src/results_hk.csv")
results_new_df = pd.read_csv("src/results_new.csv")
results_hk_df = results_hk_df.rename(columns={'hyperbolic_krum_train_accuracy': 'hk_train_accuracy','hyperbolic_krum_test_accuracy': 'hk_test_accuracy'})

# Remove rows corresponding to ipm_attack
results_df = results_df[results_df['attack_mode'] != 'ipm_attack']
results_hk_df = results_hk_df[results_hk_df['attack_mode'] != 'ipm_attack']

results_hk_df_selected = results_hk_df[['dataset', 'byzantine_fraction','noise_variance','samples_per_node', 'attack_mode', 'hk_train_accuracy', 'hk_test_accuracy']]
results_new_df_selected = results_new_df[['dataset', 'byzantine_fraction','noise_variance','samples_per_node', 'attack_mode',
                                          'euclidean_median_train_accuracy','euclidean_median_test_accuracy',
                                          'trimmed_mean_train_accuracy','trimmed_mean_test_accuracy',
                                          'centered_clipping_train_accuracy','centered_clipping_test_accuracy']]

# Merge on common columns, handling missing values
merged_results_df = results_df.merge(results_hk_df_selected, on=['dataset', 'byzantine_fraction','noise_variance','samples_per_node', 'attack_mode'], how='outer')
merged_results_df = merged_results_df.merge(results_new_df_selected, on=['dataset', 'byzantine_fraction','noise_variance','samples_per_node', 'attack_mode'], how='outer')

# Rename columns for consistency
results_df = merged_results_df.rename(columns={'samples_per_node': r'$ m $',
                                        'byzantine_fraction': r'$ \beta $',
                                        'noise_variance': r'$ \sigma^2 $'})
# Melt the dataframe to create a long-form dataframe for plotting accuracy curves
melted_df = results_df.melt(
    id_vars=['dataset', r'$ \beta $', r'$ \sigma^2 $', 'attack_mode', r'$ m $'],
    value_vars=[col for col in results_df.columns if 'test_accuracy' in col],
    var_name='accuracy_type',
    value_name='accuracy'
)
# Extract aggregation method from the accuracy_type column and rename for clarity.
# Extract aggregation method from the accuracy_type column and rename for clarity.
melted_df['Aggregation Method'] = (
    melted_df['accuracy_type'].str.split('_').str[0]
    .replace({'noisy': 'Mean',
              'noiseless': 'Mean Without Noise',
              'hyperbolic': 'Hyperbolic Median',
              'median': 'Co-ordinate Wise Median',
              'krum': 'Krum',
              'hk':'Hyperbolic Krum',
              'euclidean': 'Geometric Median (Euclidean)',
              'trimmed': 'Trimmed Mean',
              'centered': 'Centered Clipping'
              })
)

for dataset in datasets:
    plot_accuracy(melted_df, dataset)
    plot_accuracy(melted_df, dataset, plot_type='accuracy_var', x_axis=r'$ \sigma^2 $', y_axis='accuracy')

# Plot loss curves:
for dataset in datasets:
    merged_main_loss_df = pd.DataFrame()
    for byz_frac in [0.1, 0.2, 0.3]:
        for noise_var in [100.0, 200.0]:
            for samples_per_node in [32, 64]:
                loss_file = f"src/loss_curves_1/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                loss_file_hk = f"src/loss_curves_hk/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                loss_file_new = f"src/loss_curves_new/{dataset}_frac_{byz_frac}_noise_{noise_var}_attack_add_noise_samples_{samples_per_node}.csv"
                if os.path.exists(loss_file):
                    loss_df = pd.read_csv(loss_file)
                    loss_df = loss_df.rename(columns={'epoch': 'Epoch'})
                    if os.path.exists(loss_file_hk):
                        loss_hk_df = pd.read_csv(loss_file_hk)
                        loss_hk_df = loss_hk_df.rename(
                            columns={'epoch': 'Epoch', 'hyperbolic_krum_loss' : 'hk_loss'})
                        # Merge on Epoch using an outer join to handle missing values
                        loss_df = loss_df.merge(loss_hk_df, on=['Epoch'], how='outer')
                        if os.path.exists(loss_file_new):
                            loss_new_df = pd.read_csv(loss_file_new)
                            loss_new_df = loss_new_df.rename(
                                columns={'epoch': 'Epoch',
                                         'euclidean_median_loss': 'euclidean_loss',
                                         'trimmed_mean_loss': 'trimmed_loss',
                                         'centered_clipping_loss': 'centered_loss'})
                            # Merge on Epoch using an outer join to handle missing values
                            loss_df = loss_df.merge(loss_new_df, on=['Epoch'], how='outer')
                        else:
                            print(f'Cound not find: {loss_file_new}')
                        loss_melted_df = loss_df.melt(
                            id_vars=['Epoch'],
                            value_vars=[col for col in loss_df.columns if 'loss' in col],
                            var_name='loss_type',
                            value_name='Training Loss'
                        )
                        loss_melted_df['Aggregation Method'] = (
                            loss_melted_df['loss_type'].str.split('_').str[0]
                            .replace({'noisy': 'Mean',
                                      'noiseless': 'Mean Without Noise',
                                      'hyperbolic': 'Hyperbolic Median',
                                      'median': 'Co-ordinate Wise Median',
                                      'krum': 'Krum',
                                      'hk':'Hyperbolic Krum',
                                      'euclidean': 'Geometric Median (Euclidean)',
                                      'trimmed': 'Trimmed Mean',
                                      'centered': 'Centered Clipping'
                                      })
                        )
                        loss_melted_df[r'$ \sigma^2 $'] = noise_var
                        loss_melted_df[r'$ \beta $'] = byz_frac
                        loss_melted_df[r'$ m $'] = samples_per_node
                        loss_melted_df['dataset'] = dataset
                        merged_main_loss_df = pd.concat([merged_main_loss_df, loss_melted_df], ignore_index=True)
                    else:
                        print(f'Cound not find: {loss_file_hk}')
                else:
                    print(f'Cound not find: {loss_file}')
    merged_main_loss_df = merged_main_loss_df.reset_index(drop=True)
    methods = ['Mean']
    merged_main_loss_df_without = merged_main_loss_df.loc[~merged_main_loss_df['Aggregation Method'].isin(methods)]
    plot(merged_main_loss_df, dataset)
    plot(merged_main_loss_df, dataset, plot_type='loss_sample', x_axis='Epoch', y_axis='Training Loss', row_param=r'$ m $')
    plot(merged_main_loss_df_without, dataset, plot_type='loss_wo')
    plot(merged_main_loss_df_without, dataset, plot_type='loss_sample_wo', x_axis='Epoch', y_axis='Training Loss',
         row_param=r'$ m $')
