import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_ablation_table(csv_filepath="combined_result.csv"):
    """
    Generates a table comparing Euclidean vs. Hyperbolic versions of Krum and Geometric Median.
    Focuses on test accuracy.

    Args:
        csv_filepath (str): Path to the CSV file containing experiment results.

    Returns:
        pandas.DataFrame: A DataFrame summarizing the comparisons.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results CSV file not found at {csv_filepath}")
        return pd.DataFrame()

    # Define the method mappings to CSV column suffixes for test accuracy
    # Ensure these suffixes match your CSV columns exactly
    method_to_col_suffix = {
        'Krum': 'krum_test_accuracy',
        'Hyperbolic Krum': 'hk_test_accuracy',
        'Geometric Median (Euclidean)': 'euclidean_median_test_accuracy',
        'Hyperbolic Median': 'hyperbolic_test_accuracy',
        'Trimmed Mean': 'trimmed_mean_test_accuracy',
        'Centered Clipping': 'centered_clipping_test_accuracy'
    }

    # Check if all necessary columns are present
    required_suffixes = list(method_to_col_suffix.values())
    missing_cols = [col for col in required_suffixes if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in CSV: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # --- Create a list to store rows for the new table ---
    comparison_data = []
    df_filtered = df

    # We need to handle cases where not all methods might have been run for all configs.
    # Let's iterate over groups and then extract specific method results.

    # Define pairs for comparison
    comparison_pairs = [
        ("Krum", "Hyperbolic Krum"),
        ("Geometric Median (Euclidean)", "Hyperbolic Median"),
        ("Trimmed Mean", "Hyperbolic Median"),
        ("Centered Clipping", "Hyperbolic Median")
    ]

    # Define the key columns that define an "experiment setting"
    # Assuming 'samples_per_node' is also part of the configuration
    grouping_cols = ['dataset', 'byzantine_fraction', 'noise_variance', 'samples_per_node', 'attack_mode']

    # Ensure all grouping columns exist
    missing_group_cols = [col for col in grouping_cols if col not in df.columns]
    if missing_group_cols:
        print(f"Error: Missing grouping columns in CSV: {missing_group_cols}")
        return pd.DataFrame()

    for group_keys, group_df in df_filtered.groupby(grouping_cols):
        dataset, byzantine_frac, noise_var, samples_node, attack_m = group_keys

        # For each row in the group (should ideally be one row per unique config)
        # If multiple runs exist for the same config, this will take the first.
        # Consider averaging if you have multiple identical runs.
        if group_df.empty:
            continue
        row = group_df.iloc[0]  # Take the first row of the group

        for euclidean_method_name, hyperbolic_method_name in comparison_pairs:
            euclidean_col = method_to_col_suffix.get(euclidean_method_name)
            hyperbolic_col = method_to_col_suffix.get(hyperbolic_method_name)

            if euclidean_col and hyperbolic_col and euclidean_col in row and hyperbolic_col in row:
                euclidean_acc = row[euclidean_col]
                hyperbolic_acc = row[hyperbolic_col]

                improvement_abs = hyperbolic_acc - euclidean_acc
                improvement_rel = ((hyperbolic_acc - euclidean_acc) / (
                            euclidean_acc + 1e-9)) * 100  # Relative, avoid div by zero

                comparison_data.append({
                    'Dataset': dataset,
                    'Byzantine Fraction': byzantine_frac,
                    'Noise Variance': noise_var,
                    'Samples/Node': samples_node,
                    'Attack Mode': attack_m,
                    'Comparison': f"{hyperbolic_method_name} vs. {euclidean_method_name}",
                    f"{euclidean_method_name} Test Acc. (%)": f"{euclidean_acc:.2f}",
                    f"{hyperbolic_method_name} Test Acc. (%)": f"{hyperbolic_acc:.2f}",
                    'Abs. Improvement (%)': f"{improvement_abs:+.2f}",  # Show sign
                    'Rel. Improvement (%)': f"{improvement_rel:+.2f}"  # Show sign
                })
            # else:
            #     print(f"Warning: Missing data for comparison pair ({euclidean_method_name}, {hyperbolic_method_name}) in config: {group_keys}")

    if not comparison_data:
        print("No data available for comparison after processing.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(comparison_data)

    # Reorder columns for better readability
    ordered_columns = [
        'Dataset', 'Byzantine Fraction', 'Noise Variance', 'Samples/Node', 'Attack Mode',
        'Comparison'
    ]
    # Dynamically add the accuracy and improvement columns based on what's generated
    # Find unique method names from the 'Comparison' column to construct specific acc columns
    unique_comparisons = summary_df['Comparison'].unique()
    for comp_str in unique_comparisons:
        h_method, e_method = comp_str.split(' vs. ')
        e_acc_col = f"{e_method} Test Acc. (%)"
        h_acc_col = f"{h_method} Test Acc. (%)"
        if e_acc_col not in ordered_columns: ordered_columns.append(e_acc_col)
        if h_acc_col not in ordered_columns: ordered_columns.append(h_acc_col)

    ordered_columns.extend(['Abs. Improvement (%)', 'Rel. Improvement (%)'])

    # Ensure all columns in ordered_columns actually exist in summary_df before trying to reindex
    final_columns = [col for col in ordered_columns if col in summary_df.columns]

    return summary_df[final_columns]


if __name__ == '__main__':
    ablation_summary_table = generate_ablation_table(csv_filepath="combined_result.csv")

    if not ablation_summary_table.empty:
        s = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'cifar10') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Centered Clipping') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.3])) &
            (ablation_summary_table['Noise Variance'].isin([1.0, 10.0, 100.0, 200.0]))
        ]
        s['Dataset'] = s['Dataset'].replace({'spambase':'Spambase',
                   'mnist':'MNIST',
                   'fashion_mnist':'Fashion-MNIST',
                   'cifar10':'CIFAR-10'})
        print("\n--- Example LaTeX Output (Centered Clipping vs Hyperbolic Median) ---")
        print(s[['Noise Variance', 'Samples/Node',
                                     'Centered Clipping Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                                     'Abs. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        # --- Output to LaTeX (example) ---
        spambase1 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'spambase') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Geometric Median (Euclidean)') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.4])) &
            (ablation_summary_table['Noise Variance'].isin([200.0]))
        ]
        print("\n--- Example LaTeX Output (Spambase Geometric Median (Euclidean) vs Hyperbolic Median) ---")
        print(spambase1[['Samples/Node',
                                     'Geometric Median (Euclidean) Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                                     'Abs. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        spambase2 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'spambase') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Centered Clipping') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.4])) &
            (ablation_summary_table['Noise Variance'].isin([200.0]))
        ]
        print("\n--- Example LaTeX Output (Spambase Centered Clipping vs Hyperbolic Median) ---")
        print(spambase2[['Samples/Node',
                                     'Centered Clipping Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                                     'Abs. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        mnist1 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'mnist') &
            (ablation_summary_table['Comparison']== 'Hyperbolic Median vs. Geometric Median (Euclidean)') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.1])) &
            (ablation_summary_table['Samples/Node'].isin([128]))&
            (ablation_summary_table['Noise Variance'].isin([1.0, 10.0, 100.0, 200.0]))
            ]
        print("\n--- Example LaTeX Output (MNIST Geometric Median (Euclidean) vs Hyperbolic Median) ---")
        print(mnist1[['Noise Variance',
                                     'Geometric Median (Euclidean) Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                                     'Rel. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        mnist2 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'mnist') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Trimmed Mean') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.2])) &
            (ablation_summary_table['Noise Variance'].isin([100.0]))
            ]
        print("\n--- Example LaTeX Output (MNIST Trimmed Mean vs Hyperbolic Median) ---")
        print(mnist2[['Samples/Node',
                         'Trimmed Mean Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                         'Abs. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        fashion_mnist1 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'fashion_mnist') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Trimmed Mean') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.4])) &
            (ablation_summary_table['Samples/Node'].isin([32]))&
            (ablation_summary_table['Noise Variance'].isin([1.0, 10.0, 100.0, 200.0]))
            ]
        print("\n--- Example LaTeX Output (Fashion-MNIST Trimmed Mean vs Hyperbolic Median) ---")
        print(fashion_mnist1[['Noise Variance',
                      'Trimmed Mean Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                      'Abs. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        cifar10 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'cifar10') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Krum vs. Krum') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.2])) &
            (ablation_summary_table['Samples/Node'].isin([16])) &
            (ablation_summary_table['Noise Variance'].isin([1.0, 10.0, 100.0, 200.0]))
            ]
        print("\n--- Example LaTeX Output (CIFAR-10 Hyperbolic Krum vs. Krum) ---")
        print(cifar10[['Noise Variance',
                      'Krum Test Acc. (%)', 'Hyperbolic Krum Test Acc. (%)',
                      'Rel. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))

        cifar102 = ablation_summary_table[
            (ablation_summary_table['Dataset'] == 'cifar10') &
            (ablation_summary_table['Comparison'] == 'Hyperbolic Median vs. Trimmed Mean') &
            (ablation_summary_table['Byzantine Fraction'].isin([0.3])) &
            (ablation_summary_table['Noise Variance'].isin([1.0, 10.0, 100.0, 200.0]))
            ]
        print("\n--- Example LaTeX Output (CIFAR-10 Hyperbolic Median vs. Trimmed Mean) ---")
        print(cifar102[['Noise Variance', 'Samples/Node',
                       'Trimmed Mean Test Acc. (%)', 'Hyperbolic Median Test Acc. (%)',
                       'Rel. Improvement (%)']].to_latex(index=False, escape=False, float_format="%.2f"))


    else:
        print("Could not generate ablation table. Check for errors or empty data.")