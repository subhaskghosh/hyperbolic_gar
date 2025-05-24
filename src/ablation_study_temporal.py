import pandas as pd
import numpy as np
import os  # Added for dummy file check


def create_ablation_tables(csv_filepath="results_temporal_sota_final_v3.csv",
                           target_epoch=50,
                           metrics_to_compare=None):
    """
    Generates ablation study tables comparing Euclidean Geometric Median vs. Hyperbolic Geometric Median.

    Args:
        csv_filepath (str): Path to the CSV results file.
        target_epoch (int): The epoch number for which to extract metrics.
        metrics_to_compare (list, optional): List of metric column names (e.g., ['Test RMSE', 'Test MAE']).
                                            Defaults to ['Test RMSE', 'Test MAE', 'Test MSE'].
    Returns:
        dict: A dictionary where the key is "GM_vs_HGM"
              and value is a pandas DataFrame with the comparison.
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['Test RMSE', 'Test MAE', 'Test MSE', 'Test MAPE']  # Default metrics

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results CSV file not found at {csv_filepath}")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: {csv_filepath} is empty.")
        return {}

    # Filter for the target epoch and non-baseline runs (attacked scenarios)
    df_epoch = df[(df['epoch'] == target_epoch) & (df['is_baseline_run'] == False)].copy()

    if df_epoch.empty:
        print(f"No data found for epoch {target_epoch} and attacked scenarios.")
        return {}

    # Define method name mappings from CSV to a simpler key for dictionary
    # Ensure these match your 'aggregation_method' column values.
    method_map = {
        'euclidean_median': 'GM_Euclidean',
        'hyperbolic': 'GM_Hyperbolic'  # 'hyperbolic' in your CSV is Hyperbolic Geometric Median
    }

    df_epoch['MethodKey'] = df_epoch['aggregation_method'].map(method_map)
    df_epoch.dropna(subset=['MethodKey'], inplace=True)  # Keep only relevant methods
    column_rename_map = {
        "test_MSE": "Test MSE",
        "test_MAE": "Test MAE",
        "test_MAPE": "Test MAPE",  # You decided to potentially drop MAPE
        "test_RMSE": "Test RMSE"
    }
    df_epoch = df_epoch.rename(columns=column_rename_map)

    if df_epoch.empty:
        print(f"No data found for GM/HGM comparison methods in epoch {target_epoch}.")
        return {}

    ablation_tables = {}

    # --- Comparison: Euclidean Geometric Median vs. Hyperbolic Geometric Median ---
    gm_comparison_df = None
    # df_gm_family already contains only GM_Euclidean and GM_Hyperbolic due to dropna
    df_gm_family = df_epoch  # Use the already filtered df_epoch

    if not df_gm_family.empty and \
            'GM_Euclidean' in df_gm_family['MethodKey'].unique() and \
            'GM_Hyperbolic' in df_gm_family['MethodKey'].unique():
        try:
            pivot_gm = df_gm_family.pivot_table(
                index=['byzantine_fraction', 'noise_variance_param'],
                columns='MethodKey',
                values=metrics_to_compare
            )

            if pivot_gm.empty:
                print("Pivot table for GM comparison is empty. Check data and grouping keys.")
                ablation_tables["GM_vs_HGM"] = pd.DataFrame()
                return ablation_tables

            # Flatten MultiIndex columns if pivot_table created them
            if isinstance(pivot_gm.columns, pd.MultiIndex):
                pivot_gm.columns = [f'{col[0]}_{col[1]}' for col in pivot_gm.columns]

            pivot_gm.reset_index(inplace=True)

            for metric in metrics_to_compare:
                col_euclidean = f'{metric}_GM_Euclidean'
                col_hyperbolic = f'{metric}_GM_Hyperbolic'

                if col_euclidean in pivot_gm.columns and col_hyperbolic in pivot_gm.columns:
                    # For error metrics, improvement is reduction: (Old - New)
                    # Absolute reduction: Euclidean - Hyperbolic. Positive means Hyperbolic is better.
                    pivot_gm[f'{metric}_Abs_Reduction'] = pivot_gm[col_euclidean] - pivot_gm[col_hyperbolic]
                    pivot_gm[f'{metric}_Rel_Reduction (%)'] = \
                        ((pivot_gm[col_euclidean] - pivot_gm[col_hyperbolic]) / (pivot_gm[col_euclidean] + 1e-9)) * 100
                # else:
                # print(f"Warning: Missing columns for GM metric comparison: {col_euclidean} or {col_hyperbolic}")

            gm_comparison_df = pivot_gm
        except Exception as e:
            print(f"Error creating GM comparison table: {e}")
            gm_comparison_df = pd.DataFrame()

    ablation_tables["GM_vs_HGM"] = gm_comparison_df

    return ablation_tables


if __name__ == '__main__':
    final_epoch = 50
    metrics = ['Test RMSE', 'Test MAE', 'Test MSE',  'Test MAPE']

    csv_file_to_use = "result_temporal_new.csv"  # Ensure this is your latest CSV

    ablation_results = create_ablation_tables(
        csv_filepath=csv_file_to_use,
        target_epoch=final_epoch,
        metrics_to_compare=metrics
    )

    for comparison_name, df_comp in ablation_results.items():
        print(f"\n\n--- Ablation Table: {comparison_name} (Epoch {final_epoch}) ---")
        if df_comp is not None and not df_comp.empty:
            print(df_comp.to_string(float_format="%.4f"))  # More precision for review

            metric_for_latex = 'Test RMSE'
            cols_for_latex = ['byzantine_fraction', 'noise_variance_param']

            rename_dict_latex = {}
            if 'GM' in comparison_name:  # Only GM comparison is left
                keys = ['GM_Euclidean', 'GM_Hyperbolic']
                rename_dict_latex = {
                    f'{metric_for_latex}_{keys[0]}': 'Geometric Median (Euclidean)',
                    f'{metric_for_latex}_{keys[1]}': 'Hyperbolic Median'
                }
                cols_for_latex.append(f'{metric_for_latex}_{keys[0]}')
                cols_for_latex.append(f'{metric_for_latex}_{keys[1]}')
                cols_for_latex.append(f'{metric_for_latex}_Abs_Reduction')

            existing_cols_for_latex = [col for col in cols_for_latex if col in df_comp.columns]

            if existing_cols_for_latex and keys:  # Ensure keys were set
                df_latex = df_comp[existing_cols_for_latex].copy()
                df_latex.rename(columns=rename_dict_latex, inplace=True)
                df_latex.rename(columns={'byzantine_fraction': '$\\beta$',
                                         'noise_variance_param': '$\\sigma^2$',
                                         f'{metric_for_latex}_Abs_Reduction': 'Abs. Improv.'},
                                inplace=True)

                print(f"\n--- LaTeX for {comparison_name} - {metric_for_latex} ---")
                print(df_latex.to_latex(index=False, escape=False, float_format="%.2f"))
        else:
            print(f"Table for {comparison_name} is empty or could not be generated.")