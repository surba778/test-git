"""
D7054E Lab 3 – Part B: DBScan Clustering.

Two tasks are performed:

Task (i)  – Spend Score vs Annual Income clustering on the Mall Customer
             dataset.  Because the original Kaggle file requires a manual
             download the script generates a realistic synthetic version of
             the dataset so the analysis can run stand-alone.  If you place
             the original 'Mall_Customers.csv' in the same directory the
             real data will be loaded automatically.

Task (ii) – DBScan-based anomaly detection on AWS CloudWatch CPU-utilisation
             time-series data.  A synthetic signal with injected anomalies is
             produced when the real Numenta/AWS CSV files are not present.
             Place any 'ec2_cpu_utilization_*.csv' files (with columns
             'timestamp' and 'value') in the same directory to use real data.

References
----------
[16] scikit-learn DBScan example:
     https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
[17] DBScan Spend Score vs Annual Income (Kaggle):
     https://www.kaggle.com/bagavathypriya/dbscan-clustering
[18] Mall Customer Dataset:
     https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python
[19] DBScan anomaly detection for AWS CloudWatch:
     https://www.kaggle.com/d4v1d3/dbscan/notebook?select=README.md
[20] Numenta Anomaly Benchmark paper:
     https://arxiv.org/pdf/1510.03336.pdf
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def _synthetic_mall_customers(n_samples=200):
    """Return a DataFrame mimicking the Mall Customers CSV structure."""
    annual_income = np.concatenate([
        rng.normal(20, 5, 40),
        rng.normal(50, 7, 60),
        rng.normal(85, 6, 60),
        rng.normal(55, 8, 40),
    ])
    spend_score = np.concatenate([
        rng.normal(20, 8, 40),
        rng.normal(55, 10, 60),
        rng.normal(80, 7, 60),
        rng.normal(45, 12, 40),
    ])
    annual_income = np.clip(annual_income, 1, 137)
    spend_score = np.clip(spend_score, 1, 99)
    return pd.DataFrame({
        "CustomerID": range(1, n_samples + 1),
        "Genre": rng.choice(["Male", "Female"], n_samples),
        "Age": rng.integers(18, 70, n_samples),
        "Annual Income (k$)": annual_income.astype(int),
        "Spending Score (1-100)": spend_score.astype(int),
    })


def _synthetic_ec2_series(n_points=1440, anomaly_frac=0.02):
    """Return a DataFrame with synthetic CPU utilisation including anomalies."""
    timestamps = pd.date_range("2014-02-14", periods=n_points, freq="5min")
    # Normal diurnal pattern (max ~45 %)
    t = np.linspace(0, 4 * np.pi, n_points)
    base = 30 + 15 * np.sin(t) + rng.normal(0, 3, n_points)
    base = np.clip(base, 0, 100)
    # Inject spikes well outside the normal range (80–100 %) to simulate load bursts
    n_anomalies = int(n_points * anomaly_frac)
    anomaly_idx = rng.choice(n_points, n_anomalies, replace=False)
    base[anomaly_idx] = rng.uniform(80, 100, n_anomalies)
    return pd.DataFrame({"timestamp": timestamps, "value": base})


# ---------------------------------------------------------------------------
# Task (i) – Mall Customers (Spend Score vs Annual Income)
# ---------------------------------------------------------------------------

def run_task_i_mall_customers():
    """DBScan clustering: Spend Score vs Annual Income."""
    print("\n" + "=" * 60)
    print("Task (i) – Mall Customer Clustering (DBScan)")
    print("=" * 60)

    # Load dataset (real file if present, otherwise synthetic)
    csv_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded real dataset: {csv_path}")
    else:
        df = _synthetic_mall_customers()
        print("Mall_Customers.csv not found – using synthetic dataset.")

    income_col = "Annual Income (k$)"
    score_col = "Spending Score (1-100)"
    features = df[[income_col, score_col]].values

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # ----- Run DBScan for several (epsilon, min_samples) combinations -------
    param_grid = [
        {"eps": 0.3, "min_samples": 5},
        {"eps": 0.5, "min_samples": 5},   # default-ish, typically best here
        {"eps": 0.7, "min_samples": 5},
        {"eps": 0.5, "min_samples": 10},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    for ax, params in zip(axes, param_grid):
        eps = params["eps"]
        min_samp = params["min_samples"]
        db = DBSCAN(eps=eps, min_samples=min_samp)
        labels = db.fit_predict(features_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        unique_labels = sorted(set(labels))
        palette = sns.color_palette("tab10", n_colors=max(n_clusters, 1))
        color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels) if lbl != -1}
        color_map[-1] = (0.5, 0.5, 0.5)  # grey for noise

        for lbl in unique_labels:
            mask = labels == lbl
            color = color_map[lbl]
            label_text = "Noise" if lbl == -1 else f"Cluster {lbl}"
            marker = "x" if lbl == -1 else "o"
            ax.scatter(
                features[mask, 0], features[mask, 1],
                c=[color], s=30, marker=marker, alpha=0.7, label=label_text,
            )

        ax.set_xlabel(income_col)
        ax.set_ylabel(score_col)
        ax.set_title(
            f"eps={eps}, min_samples={min_samp}\n"
            f"Clusters={n_clusters}, Noise points={n_noise}"
        )
        ax.legend(fontsize=7, markerscale=1.2)

    fig.suptitle(
        "DBScan – Spend Score vs Annual Income (Mall Customers)\n"
        "Effect of epsilon and min_samples on clustering",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("dbscan_mall_customers.png", dpi=150)
    plt.show()
    print("Plot saved to dbscan_mall_customers.png")

    # Observation summary
    print("\nObservations:")
    print("  • Smaller epsilon (0.3) produces many noise points / fragmented clusters.")
    print("  • Larger epsilon (0.7) merges natural groupings into fewer, larger clusters.")
    print("  • Increasing min_samples requires a denser core to form a cluster,")
    print("    which reduces spurious small clusters but may increase noise labelling.")
    print("  • An epsilon around 0.5 with min_samples=5 typically reveals 4–5 natural")
    print("    spending segments (low-income/low-spend, high-income/high-spend, etc.).")


# ---------------------------------------------------------------------------
# Task (ii) – AWS CloudWatch CPU Anomaly Detection
# ---------------------------------------------------------------------------

def _load_ec2_files():
    """
    Return a list of (filename, DataFrame) tuples.

    Looks for 'ec2_cpu_utilization_*.csv' in the script directory first;
    falls back to a single synthetic series when none are found.
    """
    pattern = os.path.join(os.path.dirname(__file__), "ec2_cpu_utilization_*.csv")
    paths = glob.glob(pattern)

    datasets = []
    if paths:
        for path in sorted(paths):
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df = df[["timestamp", "value"]].dropna().sort_values("timestamp").reset_index(drop=True)
            datasets.append((os.path.basename(path), df))
        print(f"Loaded {len(datasets)} real EC2 CSV file(s).")
    else:
        print("No ec2_cpu_utilization_*.csv files found – using synthetic data.")
        datasets.append(("synthetic_ec2.csv", _synthetic_ec2_series()))

    return datasets


def _detect_anomalies_dbscan(series_values, eps=0.3, min_samples=5):
    """
    Detect anomalies in a 1-D time series using DBScan.

    Feature space: each point is represented by its global z-score (measuring
    how extreme the value is relative to the series distribution) paired with
    its normalised time index (ensuring temporally isolated spikes cannot form
    dense neighbourhoods with other anomalies far away in time).  Points
    labelled −1 (noise) by DBScan are returned as anomalies.

    Parameters
    ----------
    series_values : np.ndarray  – 1-D CPU utilisation values
    eps           : float        – DBScan neighbourhood radius (default 0.3)
    min_samples   : int          – DBScan minimum neighbours (default 5)

    Returns
    -------
    anomaly_mask : boolean array of length len(series_values)
    """
    n = len(series_values)
    global_mean = series_values.mean()
    global_std = series_values.std() + 1e-9
    z_scores = (series_values - global_mean) / global_std
    time_index = np.arange(n) / n  # normalised to [0, 1]

    features = np.column_stack([z_scores, time_index])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features_scaled)

    return labels == -1


def run_task_ii_aws_cloudwatch():
    """DBScan anomaly detection on AWS CloudWatch EC2 CPU-utilisation data."""
    print("\n" + "=" * 60)
    print("Task (ii) – AWS CloudWatch Anomaly Detection (DBScan)")
    print("=" * 60)

    datasets = _load_ec2_files()

    n_files = len(datasets)
    fig, axes = plt.subplots(n_files, 1, figsize=(14, 4 * n_files), squeeze=False)

    for row_idx, (fname, df) in enumerate(datasets):
        ax = axes[row_idx, 0]
        values = df["value"].values
        timestamps = df["timestamp"]

        anomaly_mask = _detect_anomalies_dbscan(values)
        n_anomalies = int(anomaly_mask.sum())

        ax.plot(timestamps, values, color="steelblue", lw=0.8, label="CPU utilisation")
        ax.scatter(
            timestamps[anomaly_mask], values[anomaly_mask],
            color="tomato", zorder=5, s=20, label=f"Anomalies ({n_anomalies})",
        )
        ax.set_title(f"{fname}  –  Anomalies detected: {n_anomalies}", fontsize=11)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("CPU Utilisation (%)")
        ax.legend(fontsize=9)

        print(f"  {fname}: {n_anomalies} anomaly points out of {len(values)} total.")

    fig.suptitle("DBScan Anomaly Detection – EC2 CPU Utilisation (AWS CloudWatch)", fontsize=13)
    plt.tight_layout()
    plt.savefig("dbscan_aws_cloudwatch.png", dpi=150)
    plt.show()
    print("Plot saved to dbscan_aws_cloudwatch.png")

    print("\nExplanation:")
    print("  Each data point is represented by two features:")
    print("    (1) its global z-score – how extreme the value is relative to the")
    print("        overall series distribution;")
    print("    (2) its normalised time index – so temporally isolated spikes cannot")
    print("        share a dense neighbourhood with distant anomalies.")
    print("  DBScan labels low-density points (label = -1 / noise) as anomalies.")
    print("  Sudden CPU spikes that are far above the normal diurnal baseline are")
    print("  isolated in the (z-score, time) feature space and detected as noise.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run both DBScan clustering tasks."""
    print("=" * 60)
    print("D7054E Lab 3 – Part B: DBScan Clustering")
    print("=" * 60)

    run_task_i_mall_customers()
    run_task_ii_aws_cloudwatch()

    print("\nPart B complete.")


if __name__ == "__main__":
    main()
