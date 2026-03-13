from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron_wrapper import Perceptron

def _get_weight_bias(model):
    return float(model._core.w[0]), float(model._core.w[1]), float(model._core.b)


DATASETS = {
    "iris": "iris.csv",
    "sonar": "sonar.csv",
    "wdbc": "wdbc.csv",
}


def get_local_dataset_paths(data_dir):
    dataset_paths = {name: data_dir / filename for name, filename in DATASETS.items()}
    missing = [str(path) for path in dataset_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files:\n" + "\n".join(missing))
    return dataset_paths


def _prepare_binary_data(name, csv_path):
    df = pd.read_csv(csv_path)

    if name == "iris":
        df = df[df["species"].isin(["Iris-setosa", "Iris-versicolor"])].copy()
        y = np.where(df["species"].to_numpy() == "Iris-setosa", 1, -1)
        X = df[["sepal_length", "sepal_width"]].to_numpy(dtype=float)
    elif name == "sonar":
        y = np.where(df["label"].to_numpy() == "R", 1, -1)
        X = df[["feature_1", "feature_2"]].to_numpy(dtype=float)
    elif name == "wdbc":
        y = np.where(df["diagnosis"].to_numpy() == "M", 1, -1)
        X = df[["feature_1", "feature_2"]].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    return X, y


def _plot_boundary(ax, model, X, y, title):
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="royalblue", label="+1", alpha=0.8)
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color="tomato", label="-1", alpha=0.8)

    w0, w1, b = _get_weight_bias(model)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_points = np.linspace(x_min, x_max, 200)
    if w1 != 0:
        y_points = -(w0 * x_points + b) / w1
        ax.plot(x_points, y_points, "k--", linewidth=2, label="Decision boundary")

    ax.set_title(title)
    ax.set_xlabel("Feature 1 (standardized)")
    ax.set_ylabel("Feature 2 (standardized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def train_and_visualize(dataset_paths):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, csv_path) in zip(axes, dataset_paths.items()):
        X, y = _prepare_binary_data(name, csv_path)
        model = Perceptron(learning_rate=0.05, max_iters=2000)
        model.fit(X, y)

        preds = model.predict(X)
        acc = (preds == y).mean()
        print(f"{name}: samples={len(y)}, training_accuracy={acc:.4f}")

        _plot_boundary(ax, model, X, y, f"{name.upper()} (acc={acc:.3f})")

    plt.tight_layout()
    plt.show()


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    dataset_paths = get_local_dataset_paths(data_dir)
    for name, path in dataset_paths.items():
        print(f"Using local dataset: {name} -> {path}")

    print("\nTraining perceptron models and visualizing decision boundaries...")
    train_and_visualize(dataset_paths)


if __name__ == "__main__":
    main()