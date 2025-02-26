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

def plot(dataframe, plot_type='loss', x_axis='epoch', y_axis='train_loss', row_param=r'$ \sigma^2 $'):
    """
    Plots loss curves using seaborn's FacetGrid.
    """
    # Reset index to avoid duplicate index issues
    dataframe = dataframe.reset_index(drop=True)
    sns.set_style("white")
    g = sns.FacetGrid(
        dataframe,
        col=row_param if row_param else None,
        row=r'$ \beta $',
        hue="aggregation_method",
        legend_out=True,
        palette='Spectral'
    )
    g.map(sns.lineplot, x_axis, y_axis)
    g.set_axis_labels(x_axis, y_axis)
    sns.set_theme(rc={'figure.figsize': (12, 8)})
    g.add_legend(adjust_subtitles=True)
    g.set_titles(size=20)

    os.makedirs("plots_pdf", exist_ok=True)
    plt.savefig(f"plots_pdf/{plot_type}_plot.pdf", dpi=dpi)
    plt.close()


# Read and preprocess results
results_df = pd.read_csv("results_temporal.csv")

# Identify columns that start with 'train' but are not 'train_loss'
cols_to_drop = [col for col in results_df.columns if  col.startswith('train') and col != 'train_loss']

# Drop the identified columns
results_df = results_df.drop(columns=cols_to_drop)
results_df = results_df.rename(columns={
                                        'byzantine_fraction': r'$ \beta $',
                                        'noise_variance': r'$ \sigma^2 $',
                                         "train_loss":"loss", "test_MSE":"MSE","test_MAE":"MAE"
                                            ,"test_MAPE":"MAPE","test_RMSE":"RMSE"})
# Display the resulting DataFrame
print(results_df)
loss_type = ["loss","MSE","MAE","MAPE","RMSE"]
for loss in loss_type:
    plot(results_df, plot_type=loss, y_axis=loss)
