import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Read and preprocess results
results_df = pd.read_csv("src/results_1.csv")
results_hk_df = pd.read_csv("src/results_hk.csv")
results_hk_df = results_hk_df.rename(columns={'hyperbolic_krum_train_accuracy': 'hk_train_accuracy','hyperbolic_krum_test_accuracy': 'hk_test_accuracy'})
# Remove rows corresponding to ipm_attack
results_df = results_df[results_df['attack_mode'] != 'ipm_attack']
results_hk_df = results_hk_df[results_hk_df['attack_mode'] != 'ipm_attack']

results_hk_df_selected = results_hk_df[['dataset', 'attack_mode', 'hk_train_accuracy', 'hk_test_accuracy']]
# Merge on common columns, handling missing values
merged_results_df = results_df.merge(results_hk_df_selected, on=['dataset', 'attack_mode'], how='outer')

# Melt the dataframe to create a long-form dataframe for plotting accuracy curves
melted_df = results_df.melt(
    id_vars=['dataset', 'byzantine_fraction', 'noise_variance', 'attack_mode', 'samples_per_node'],
    value_vars=[col for col in results_df.columns if 'accuracy' in col],
    var_name='accuracy_type',
    value_name='accuracy'
)
# Extract aggregation method from the accuracy_type column and rename for clarity.
melted_df['Aggregation Method'] = (
    melted_df['accuracy_type'].str.split('_').str[0]
    .replace({'noisy': 'mean with noise', 'noiseless': 'mean without noise',
              'hyperbolic': 'hyperbolic median', 'median': 'co-ordinate wise median', 'hk':'hyperbolic krum'})
)
melted_df.reset_index(drop=True)

melted_df.to_csv('src/combined_result.csv', sep='\t', encoding='utf-8', index=False, header=True)

import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the CSV file
df = melted_df

# Inspect the first few rows to verify column names
print(df.head())

# Suppose the CSV contains columns: 'aggregation_method' and 'accuracy'.
# Group the data by aggregation method.
agg_methods = df['Aggregation Method'].unique()
print("Aggregation methods found:", agg_methods)

# Create a list of accuracy values for each aggregation method
accuracy_groups = [df[df['Aggregation Method'] == method]['accuracy'].dropna().values
                   for method in agg_methods]

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*accuracy_groups)
print("ANOVA F-statistic:", f_stat)
print("ANOVA p-value:", p_value)

# If the ANOVA test is significant, perform Tukey's HSD for pairwise comparisons.
tukey = pairwise_tukeyhsd(endog=df['accuracy'], groups=df['Aggregation Method'], alpha=0.05)
print(tukey)

# Define the column for the aggregation method and get its unique values
unique_methods = df['Aggregation Method'].unique()
# Define a palette for these methods (using the Spectral palette)
palette = sns.color_palette("Spectral", len(unique_methods))
# Create a dictionary mapping each method to its corresponding color
palette_dict = dict(zip(unique_methods, palette))

# Plot a histogram for accuracy
plt.figure(figsize=(12, 8))
ax = sns.histplot(
    data=df,
    x="accuracy",
    hue="Aggregation Method",
    bins=20,
    palette=palette,
    multiple="dodge"
)

# Try to get the legend handles and labels from the current Axes
handles, labels = ax.get_legend_handles_labels()

# If no handles are found, explicitly create them using the custom palette
if not handles or len(handles) == 0:
    handles = [Patch(facecolor=palette_dict[method], label=method) for method in unique_methods]
    labels = list(unique_methods)

plt.legend(handles, labels, title="Aggregation Method")
plt.title("Distribution of Classification Accuracy by Aggregation Method")
plt.xlabel("Accuracy (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("src/accuracy_histogram.pdf", dpi=300)
plt.show()


