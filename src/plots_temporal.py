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


def plot_combined(dataframe, plot_type='loss', x_axis='Epoch', y_axis='train_loss', suffix='', palatte=None):
    """
    Plots loss curves using seaborn's FacetGrid arranged in a single row,
    where each facet corresponds to a unique combination of Byzantine fraction
    and noise variance.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the loss curves.
        plot_type (str): Type of plot (e.g., 'loss').
        x_axis (str): Column name for the x-axis.
        y_axis (str): Column name for the y-axis.
    """
    # Create a new column combining Byzantine fraction and noise variance
    dataframe['BetaSigma'] = dataframe.apply(
        lambda row: f"$\\beta = {row['byzantine_fraction']},\\,\\sigma^2 = {row['noise_variance_param']}$", axis=1
    )

    # Reset index to avoid duplicate index issues
    dataframe = dataframe.reset_index(drop=True)

    sns.set_style("white")
    g = sns.FacetGrid(
        dataframe,
        col='BetaSigma',
        hue="Aggregation Method",
        legend_out=True,
        height=2.5,
        palette=palatte,
        col_wrap=2
    )
    g.map(sns.lineplot, x_axis, y_axis, errorbar='sd')
    g.set_axis_labels(x_axis, y_axis, size=16)
    # Set overall figure size wide and short
    sns.set_theme(rc={'figure.figsize': (8, 4)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(col_template="{col_name}", size=16)

    os.makedirs("plots_pdf", exist_ok=True)
    plt.savefig(f"plots_pdf/temporal_{plot_type.replace(' ', '_').lower()}_plot{suffix}.pdf", dpi=300)
    plt.close()


# Read and preprocess results
results_df = pd.read_csv("result_temporal_new.csv")

cols_to_drop = [col for col in results_df.columns if
                col.startswith('train_') and col not in ['train_loss', 'avg_batch_train_loss']]
results_df = results_df.drop(columns=cols_to_drop, errors='ignore')

# 2. Rename columns for plotting clarity
column_rename_map = {
    'aggregation_method': 'Aggregation Method Raw',  # Keep original for now
    "avg_batch_train_loss": "Loss",  # For y-axis if plotting train loss
    "test_MSE": "Test MSE",
    "test_MAE": "Test MAE",
    "test_MAPE": "Test MAPE", # You decided to potentially drop MAPE
    "test_RMSE": "Test RMSE",
    "epoch": "Epoch"
}
results_df = results_df.rename(columns=column_rename_map)


# 3. Create the new 'Aggregation Method' display name based on 'Aggregation Method Raw' and 'is_baseline_run'
def get_display_agg_method(row):
    raw_method = row['Aggregation Method Raw']
    is_baseline = row['is_baseline_run']

    # Define your mapping here
    method_display_names = {
        'hyperbolic': 'Hyperbolic Median',
        'median': 'Co-ordinate Wise Median',
        'krum': 'Krum',
        'euclidean_median': 'Geometric Median (Euclidean)',
        'average': 'Mean Without Noise',
        'trimmed_mean': 'Trimmed Mean',
        'centered_clipping': 'Centered Clipping'
    }

    display_name = method_display_names.get(raw_method, raw_method)  # Default to raw_method if not in map

    if raw_method == 'average' and is_baseline:
        return 'Mean Without Noise'  # Specific name for baseline average
    elif raw_method == 'average' and not is_baseline:
        return 'Mean'  # Specific name for attacked average
    return display_name


# Apply the function to create the new column
results_df['Aggregation Method'] = results_df.apply(get_display_agg_method, axis=1)
metrics_to_plot = ["Loss", "Test MSE", "Test MAE", "Test RMSE", "Test MAPE"]
selected = ['Hyperbolic Median', 'Geometric Median (Euclidean)', 'Centered Clipping', 'Mean Without Noise', 'Krum', 'Co-ordinate Wise Median']
palatte = sns.color_palette(['#023eff', '#ff7c00', '#1ac938', '#e8000b', '#8b2be2', '#9f4800', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff'])
other_palatte = sns.color_palette(['#023eff', '#1ac938', '#e8000b', '#8b2be2', '#f14cc1', '#a3a3a3', '#ffc400', '#00d7ff'])
other_results_df = results_df[results_df['Aggregation Method'].isin(selected)]
for loss in metrics_to_plot:
    plot_combined(results_df, plot_type=loss, y_axis=loss, suffix='', palatte=palatte)
    plot_combined(other_results_df, plot_type=loss, y_axis=loss, suffix='_selected', palatte=other_palatte)
