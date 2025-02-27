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


def plot_combined(dataframe, plot_type='loss', x_axis='epoch', y_axis='train_loss'):
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
        lambda row: f"$\\beta = {row['byzantine_fraction']},\\,\\sigma^2 = {row['noise_variance']}$", axis=1
    )

    # Reset index to avoid duplicate index issues
    dataframe = dataframe.reset_index(drop=True)
    sns.set_style("white")
    g = sns.FacetGrid(
        dataframe,
        col='BetaSigma',
        hue="Aggregation Method",
        legend_out=True,
        height=3.5,
        palette='Spectral',
        col_wrap=4  # Arrange facets in one row if there are exactly 4 unique combinations
    )
    g.map(sns.lineplot, x_axis, y_axis)
    g.set_axis_labels(x_axis, y_axis, size=18)
    # Set overall figure size wide and short
    sns.set_theme(rc={'figure.figsize': (16, 4)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(col_template="{col_name}", size=20)

    os.makedirs("plots_pdf", exist_ok=True)
    plt.savefig(f"plots_pdf/temporal_{plot_type}_plot.pdf", dpi=300)
    plt.close()


# Read and preprocess results
results_df = pd.read_csv("results_temporal.csv")

# Identify columns that start with 'train' but are not 'train_loss'
cols_to_drop = [col for col in results_df.columns if  col.startswith('train') and col != 'train_loss']

# Drop the identified columns
results_df = results_df.drop(columns=cols_to_drop)
results_df = results_df.rename(columns={
                                        'aggregation_method': 'Aggregation Method',
                                         "train_loss": "loss",
                                         "test_MSE":"MSE",
                                         "test_MAE":"MAE","test_MAPE":"MAPE","test_RMSE":"RMSE"})
# Display the resulting DataFrame
print(results_df)
loss_type = ["loss","MSE","MAE","MAPE","RMSE"]
for loss in loss_type:
    plot_combined(results_df, plot_type=loss, y_axis=loss)
