import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Constants for directories
REPORTS_DIR = "reports"
DATA_DIR = "data"

# Ensure folders exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ========================
# Core functions
# ========================

def list_files(directory: str = DATA_DIR):
    """Return all files in a directory."""
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def analyze_text_file(filepath: str) -> dict:
    """Analyze text file for lines, words, and unique words."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    report = {
        "lines": len(text.splitlines()),
        "words": len(words),
        "unique_words": len(set(words))
    }
    return report

def analyze_csv_numeric(filepath: str) -> pd.DataFrame:
    """Return statistics for numeric columns in a CSV."""
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=np.number)
    return numeric_cols.describe().T

def train_linear_model(csv_file: str, target_column: str):
    """Train Linear Regression on CSV numeric data and return MSE."""
    df = pd.read_csv(csv_file)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Linear Regression trained. MSE: {mse:.2f}")

    return model

def save_report(data: dict, filename: str):
    """Save a dictionary report as JSON."""
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Report saved: {filepath}")

def read_file_generator(filepath: str):
    """Yield lines one by one from a large file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line

def plot_column(csv_file: str, column_name: str):
    df = pd.read_csv(csv_file)
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found.")
        return
    df[column_name].plot(kind='line', title=column_name)
    plt.show()

# ========================
# CLI with argparse
# ========================

def main():
    parser = argparse.ArgumentParser(description="Advanced File Analyzer + ML Tool")
    subparsers = parser.add_subparsers(dest="command")

    # list-files
    subparsers.add_parser("list-files", help="List all files in the data directory")

    # analyze-text
    parser_text = subparsers.add_parser("analyze-text", help="Analyze a text file")
    parser_text.add_argument("filename", help="Text file name in data folder")

    # analyze-csv
    parser_csv = subparsers.add_parser("analyze-csv", help="Analyze numeric CSV")
    parser_csv.add_argument("filename", help="CSV file name in data folder")

    # train-ml
    parser_ml = subparsers.add_parser("train-ml", help="Train linear regression model")
    parser_ml.add_argument("filename", help="CSV file name in data folder")
    parser_ml.add_argument("target", help="Target column name")

    # plot
    parser_plot = subparsers.add_parser("plot", help="Plot numeric column from CSV")
    parser_plot.add_argument("filename", help="CSV file name in data folder")
    parser_plot.add_argument("column", help="Column to plot")

    args = parser.parse_args()

    try:
        if args.command == "list-files":
            files = list_files()
            print("Files in data folder:", files)

        elif args.command == "analyze-text":
            path = os.path.join(DATA_DIR, args.filename)
            report = analyze_text_file(path)
            print(report)
            save_report(report, f"{args.filename}_report.json")

        elif args.command == "analyze-csv":
            path = os.path.join(DATA_DIR, args.filename)
            stats = analyze_csv_numeric(path)
            print(stats)

        elif args.command == "train-ml":
            path = os.path.join(DATA_DIR, args.filename)
            train_linear_model(path, args.target)

        elif args.command == "plot":
            path = os.path.join(DATA_DIR, args.filename)
            plot_column(path, args.column)

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
