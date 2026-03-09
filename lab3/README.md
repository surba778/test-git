# D7054E Lab 3

Python implementation for D7054E Lab 3, covering Part A (Naive Bayes) and Part B (DBScan Clustering).

---

## Environment Setup

Create and activate a virtual environment, then install all dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux / macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Part A – Naive Bayes on the UCI Wine Dataset

**Script:** `part_a_naive_bayes.py`

Implements Gaussian Naive Bayes classification on the [UCI Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine) using scikit-learn.

### What it does
1. Loads the wine dataset (178 samples, 13 numeric features, 3 cultivar classes).
2. Splits data 70 / 30 into training and test sets with stratification.
3. Standardises features with `StandardScaler`.
4. Trains a `GaussianNB` classifier.
5. Reports accuracy and a full classification report.
6. Produces three plots:
   - **Confusion matrix heatmap** (`confusion_matrix_naive_bayes.png`)
   - **Feature distributions** by class (`feature_distributions_naive_bayes.png`)
   - **Class prior probabilities** bar chart (`prior_probabilities_naive_bayes.png`)

### Run

```bash
python part_a_naive_bayes.py
```

---

## Part B – DBScan Clustering

**Script:** `part_b_clustering.py`

Implements DBScan-based clustering and anomaly detection using scikit-learn.

### Task (i) – Spend Score vs Annual Income (Mall Customers)

Clusters mall customers by their annual income and spending score.

- If `Mall_Customers.csv` is present in the `lab3/` directory, it is loaded automatically.
  Download from [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).
- Otherwise, a realistic synthetic dataset is generated automatically.

Four DBScan parameter combinations are tried (`eps` ∈ {0.3, 0.5, 0.7}, `min_samples` ∈ {5, 10})
and the effect on cluster count and noise points is documented in the console output.

**Output plot:** `dbscan_mall_customers.png`

### Task (ii) – AWS CloudWatch CPU Anomaly Detection

Detects anomalies in EC2 CPU utilisation time-series data.

- If one or more `ec2_cpu_utilization_*.csv` files (columns: `timestamp`, `value`) are
  present in the `lab3/` directory they are loaded automatically.
  Files are part of the [Numenta Anomaly Benchmark](https://arxiv.org/pdf/1510.03336.pdf).
- Otherwise, a synthetic diurnal signal with injected anomalies is used.

Each point is represented by its global z-score and normalised time index, forming a 2-D
feature space. DBScan treats low-density points (label = −1) as anomalies.

**Output plot:** `dbscan_aws_cloudwatch.png`

### Run

```bash
python part_b_clustering.py
```

---

## References

| # | Description | URL |
|---|-------------|-----|
| 3 | DataCamp Naive Bayes tutorial | https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn |
| 4 | UCI Wine dataset | https://archive.ics.uci.edu/ml/datasets/wine |
| 16 | scikit-learn DBScan example | https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html |
| 17 | DBScan Spend Score vs Annual Income (Kaggle) | https://www.kaggle.com/bagavathypriya/dbscan-clustering |
| 18 | Mall Customer Dataset | https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python |
| 19 | DBScan anomaly detection – AWS CloudWatch | https://www.kaggle.com/d4v1d3/dbscan/notebook?select=README.md |
| 20 | Numenta Anomaly Benchmark paper | https://arxiv.org/pdf/1510.03336.pdf |
