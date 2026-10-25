#!/usr/bin/env python3
"""Provenance helper: export the Wisconsin Diagnostic Breast Cancer data to CSV.

The analysis does NOT need this script (or scikit-learn) to run; the CSV it
writes is already committed in data/wdbc.csv. It is included to show exactly
where the data came from.

Source: UCI Machine Learning Repository, "Breast Cancer Wisconsin (Diagnostic)"
(Wolberg, Street & Mangasarian, 1995; licence CC BY 4.0), as bundled with
scikit-learn's `load_breast_cancer`.

Usage:
    pip install scikit-learn pandas
    python prep_data.py
"""
from sklearn.datasets import load_breast_cancer
import pandas as pd


def snake(name):
    """'mean radius' -> 'mean_radius', 'radius error' -> 'radius_se'."""
    name = name.replace(" error", "_se").replace(" ", "_")
    return name


def main():
    d = load_breast_cancer()
    df = pd.DataFrame(d.data, columns=[snake(n) for n in d.feature_names])
    # In scikit-learn, target 0 = malignant, 1 = benign
    df.insert(0, "diagnosis", pd.Series(d.target).map({0: "malignant", 1: "benign"}))
    df.insert(0, "sample_index", range(1, len(df) + 1))
    df.to_csv("data/wdbc.csv", index=False)
    print(df.shape, df["diagnosis"].value_counts().to_dict())


if __name__ == "__main__":
    main()
