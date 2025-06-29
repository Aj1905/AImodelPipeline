import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_counter_all(df: pd.DataFrame, figsize_per_plot=(6, 4), max_unique_values=20, ncols=3):
    """データフレームの全ての列についてcountplotまたはヒストグラムを表示する"""
    columns = df.columns.tolist()
    n_cols = len(columns)
    if n_cols == 0:
        print("データフレームに列がありません。")
        return
    nrows = math.ceil(n_cols / ncols)
    total_figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=total_figsize)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, column in enumerate(columns):
        ax = axes[i]
        unique_count = df[column].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        try:
            if unique_count > max_unique_values and is_numeric:
                ax.hist(df[column].dropna(), bins=30, edgecolor="black")
                ax.set_title(f"{column}(Histogram - {unique_count} unique)")
            elif unique_count > max_unique_values:
                top_values = df[column].value_counts().head(max_unique_values)
                top_values.plot(kind="bar", ax=ax)
                ax.set_title(f"{column}(Top {len(top_values)}/{unique_count} unique)")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
            else:
                sns.countplot(data=df, x=column, ax=ax)
                ax.set_title(f"{column}({unique_count} unique)")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:{e}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{column} (Error)")
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def basic_statistics(df: pd.DataFrame, top_n: int = 10) -> None:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        print("=== 数値列の統計量 ===")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            stats = {
                "count": series.count(),
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "var": series.var(),
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "q1": series.quantile(0.25),
                "q3": series.quantile(0.75),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
            }
            print(f"--- {col} ---")
            for k, v in stats.items():
                out = f"{k:>10}: {v:.4f}" if isinstance(v, int | float) else f"{k:>10}: {v}"
                print(out)
    else:
        print("数値列がありません。")
    categorical_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if categorical_cols:
        print("=== カテゴリ列の頻度ランキング ===")
        for col in categorical_cols:
            counts = df[col].value_counts(dropna=False)
            print(f"--- {col} --- ユニーク数: {counts.shape[0]}")
            for i, (val, cnt) in enumerate(counts.head(top_n).items(), 1):
                pct = cnt / len(df) * 100
                print(f"{i:>2}. {val} - {cnt} ({pct:.2f}%)")
            if counts.shape[0] > top_n:
                print(f"... その他 {counts.shape[0] - top_n} 件")
    else:
        print("カテゴリ列がありません。")


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Scatter plot of {x} vs {y}")
    plt.show()


def plot_3d_histogram(df: pd.DataFrame, x: str, y: str, bins: int = 10):
    hist, xedges, yedges = np.histogram2d(df[x], df[y], bins=bins)
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = dy = xedges[1] - xedges[0]
    dz = hist.ravel()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    plt.show()
