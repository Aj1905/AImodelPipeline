import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

try:
    import japanize_matplotlib  # noqa: F401
    import matplotlib.pyplot as plt

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ 可視化ライブラリ(matplotlib, japanize_matplotlib)が利用できません")


def _plot_mean_comparison(fold_details: list, plt, x_pos: np.ndarray, width: float) -> None:
    """平均値比較プロット"""
    plt.subplot(3, 4, 1)
    train_means = [fold["train_stats"]["mean"] for fold in fold_details]
    test_means = [fold["test_stats"]["mean"] for fold in fold_details]

    plt.bar(x_pos - width / 2, train_means, width, label="学習データ", alpha=0.7, color="skyblue")
    plt.bar(x_pos + width / 2, test_means, width, label="テストデータ", alpha=0.7, color="lightcoral")

    plt.xlabel("Fold")
    plt.ylabel("平均値")
    plt.title("各Foldの学習・テストデータ平均値比較")
    plt.legend()
    plt.xticks(x_pos, [f"Fold {i + 1}" for i in range(len(fold_details))])
    plt.grid(True, alpha=0.3)


def _plot_std_comparison(fold_details: list, plt, x_pos: np.ndarray, width: float) -> None:
    """標準偏差比較プロット"""
    plt.subplot(3, 4, 2)
    train_stds = [fold["train_stats"]["std"] for fold in fold_details]
    test_stds = [fold["test_stats"]["std"] for fold in fold_details]

    plt.bar(x_pos - width / 2, train_stds, width, label="学習データ", alpha=0.7, color="skyblue")
    plt.bar(x_pos + width / 2, test_stds, width, label="テストデータ", alpha=0.7, color="lightcoral")

    plt.xlabel("Fold")
    plt.ylabel("標準偏差")
    plt.title("各Foldの学習・テストデータ標準偏差比較")
    plt.legend()
    plt.xticks(x_pos, [f"Fold {i + 1}" for i in range(len(fold_details))])
    plt.grid(True, alpha=0.3)


def _plot_score_trend(fold_details: list, plt) -> None:
    """スコア推移プロット"""
    plt.subplot(3, 4, 3)
    fold_scores = [fold["score"] for fold in fold_details]

    plt.plot(range(1, len(fold_scores) + 1), fold_scores, "o-", color="green", linewidth=2, markersize=8)
    plt.axhline(
        y=np.mean(fold_scores), color="red", linestyle="--", alpha=0.7, label=f"平均: {np.mean(fold_scores):.3f}"
    )

    plt.xlabel("Fold")
    plt.ylabel("R²スコア")
    plt.title("各FoldのR²スコア推移")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_prediction_vs_actual(all_y_true: list, all_y_pred: list, plt) -> None:
    """予測値vs実測値プロット"""
    plt.subplot(3, 4, 4)
    plt.scatter(all_y_true, all_y_pred, alpha=0.6, color="blue")

    min_val = min(min(all_y_true), min(all_y_pred))
    max_val = max(max(all_y_true), max(all_y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="理想線")

    plt.xlabel("実測値")
    plt.ylabel("予測値")
    plt.title("予測値 vs 実測値")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_error_distribution(all_y_true: list, all_y_pred: list, plt) -> None:
    """誤差分布プロット"""
    plt.subplot(3, 4, 5)
    errors = np.array(all_y_true) - np.array(all_y_pred)

    plt.hist(errors, bins=30, alpha=0.7, color="orange", edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", alpha=0.8, label="誤差=0")
    plt.axvline(x=np.mean(errors), color="green", linestyle="--", alpha=0.8, label=f"平均誤差: {np.mean(errors):.3f}")

    plt.xlabel("予測誤差")
    plt.ylabel("頻度")
    plt.title("予測誤差の分布")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_time_series_errors(fold_details: list, plt) -> None:
    """時系列誤差プロット"""
    plt.subplot(3, 4, 6)

    fold_errors = []
    for fold in fold_details:
        fold_error = np.mean(np.abs(fold["y_true"] - fold["y_pred"]))
        fold_errors.append(fold_error)

    plt.plot(range(1, len(fold_errors) + 1), fold_errors, "o-", color="purple", linewidth=2, markersize=8)
    plt.axhline(
        y=np.mean(fold_errors),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"平均誤差: {np.mean(fold_errors):.3f}",
    )

    plt.xlabel("Fold")
    plt.ylabel("平均絶対誤差")
    plt.title("時系列での予測誤差推移")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_data_size_comparison(fold_details: list, plt, x_pos: np.ndarray, width: float) -> None:
    """データサイズ比較プロット"""
    plt.subplot(3, 4, 7)
    train_sizes = [fold["train_size"] for fold in fold_details]
    test_sizes = [fold["test_size"] for fold in fold_details]

    plt.bar(x_pos - width / 2, train_sizes, width, label="学習データ", alpha=0.7, color="lightgreen")
    plt.bar(x_pos + width / 2, test_sizes, width, label="テストデータ", alpha=0.7, color="lightpink")

    plt.xlabel("Fold")
    plt.ylabel("データサイズ")
    plt.title("各Foldのデータサイズ比較")
    plt.legend()
    plt.xticks(x_pos, [f"Fold {i + 1}" for i in range(len(fold_details))])
    plt.grid(True, alpha=0.3)


def _plot_overall_metrics(all_y_true: list, all_y_pred: list, plt) -> None:
    """全体性能指標プロット"""
    plt.subplot(3, 4, 8)
    errors = np.array(all_y_true) - np.array(all_y_pred)

    metrics = ["R²", "MAE", "RMSE"]
    overall_r2 = r2_score(all_y_true, all_y_pred)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    metric_values = [overall_r2, mae, rmse]

    colors = ["green", "orange", "red"]
    bars = plt.bar(metrics, metric_values, color=colors, alpha=0.7)

    for bar, value in zip(bars, metric_values, strict=False):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom")

    plt.ylabel("値")
    plt.title("全体の性能指標")
    plt.grid(True, alpha=0.3)


def _plot_performance_trend(scores: list, plt) -> None:
    """性能トレンド分析プロット"""
    plt.subplot(3, 4, 9)

    x = np.arange(len(scores))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
    line = slope * x + intercept

    plt.scatter(x + 1, scores, color="blue", s=100, alpha=0.7, label="実際のスコア")
    plt.plot(x + 1, line, "r--", alpha=0.8, label=f"トレンド線 (傾き: {slope:.3f})")

    plt.xlabel("Fold")
    plt.ylabel("R²スコア")
    plt.title("性能トレンド分析")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_expanding_window_trend(fold_details: list, plt) -> None:
    """拡張ウィンドウトレンドプロット"""
    plt.subplot(3, 4, 10)
    train_sizes = [fold["train_size"] for fold in fold_details]

    plt.plot(range(1, len(train_sizes) + 1), train_sizes, "o-", color="blue", linewidth=2, markersize=8)
    plt.xlabel("Fold")
    plt.ylabel("学習データサイズ")
    plt.title("拡張ウィンドウCV: 学習データサイズ推移")
    plt.grid(True, alpha=0.3)


def _plot_gap_effect_analysis(fold_details: list, cv_config: dict, plt) -> None:
    """Gap効果分析プロット"""
    plt.subplot(3, 4, 11)

    gap_folds = []
    no_gap_scores = []

    for i, fold in enumerate(fold_details):
        gap_folds.append(i + 1)
        no_gap_scores.append(fold["score"])

    plt.plot(gap_folds, no_gap_scores, "o-", color="purple", linewidth=2, markersize=8)
    plt.axhline(
        y=np.mean(no_gap_scores),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"平均: {np.mean(no_gap_scores):.3f}",
    )

    plt.xlabel("Fold")
    plt.ylabel("R²スコア")
    plt.title(f"Gap効果分析 (Gap: {cv_config['gap']}行)")
    plt.legend()
    plt.grid(True, alpha=0.3)


def _generate_config_text(cv_method: str, cv_config: dict, fold_details: list, scores: list) -> str:
    """CV設定のテキストを生成"""
    config_text = "CV設定概要\n"
    config_text += f"方法: {cv_method.upper()}\n"

    if cv_config["train_size_type"] == "ratio":
        config_text += f"学習データ: {cv_config['train_size_ratio'] * 100:.0f}%\n"
    else:
        config_text += f"学習データ: {cv_config['train_size_absolute']}行\n"

    if cv_config["val_size_type"] == "fixed":
        config_text += f"検証データ: {cv_config['val_size']}行\n"
    else:
        config_text += "検証データ: 残り全て\n"

    config_text += f"Gap: {cv_config.get('gap', 0)}行\n"

    if cv_config["step_size_type"] == "val_size":
        config_text += "ステップ: 検証データサイズ\n"
    else:
        config_text += f"ステップ: {cv_config['step_size']}行\n"

    config_text += f"Fold数: {len(fold_details)}\n"
    config_text += f"平均R²: {np.mean(scores):.3f}\n"
    config_text += f"標準偏差: {np.std(scores):.3f}"

    return config_text


def _create_comparison_visualizations(
    fold_details: list,
    cv_method: str,
    all_y_true: list,
    all_y_pred: list,
    scores: list | None = None,
    cv_config: dict | None = None,
):
    """学習データとテストデータの比較可視化を生成"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ 可視化ライブラリが利用できません。可視化をスキップします。")
        return

    try:
        if cv_method in ["rolling", "expanding"] and cv_config:
            plt.figure(figsize=(24, 18))
        else:
            plt.figure(figsize=(20, 15))

        x_pos = np.arange(len(fold_details))
        width = 0.35

        # 基本プロット
        _plot_mean_comparison(fold_details, plt, x_pos, width)
        _plot_std_comparison(fold_details, plt, x_pos, width)
        _plot_score_trend(fold_details, plt)
        _plot_prediction_vs_actual(all_y_true, all_y_pred, plt)
        _plot_error_distribution(all_y_true, all_y_pred, plt)

        # 時系列CV特有のプロット
        if cv_method in ["rolling", "expanding"] and scores:
            _plot_time_series_errors(fold_details, plt)

        # データサイズと性能指標
        _plot_data_size_comparison(fold_details, plt, x_pos, width)
        _plot_overall_metrics(all_y_true, all_y_pred, plt)

        # 時系列CVの追加分析
        if cv_method in ["rolling", "expanding"] and cv_config and len(scores) > 2:
            _plot_performance_trend(scores, plt)

            if cv_method == "expanding":
                _plot_expanding_window_trend(fold_details, plt)

            if cv_config.get("gap", 0) > 0:
                _plot_gap_effect_analysis(fold_details, cv_config, plt)

        plt.subplot(3, 4, 12)
        plt.axis("off")

        # 設定テキストを生成
        config_text = _generate_config_text(cv_method, cv_config, fold_details, scores)

        plt.text(
            0.1,
            0.9,
            config_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig("cv_comparison_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()
        print("✓ 可視化が完了しました: cv_comparison_analysis.png")

    except Exception as e:
        import traceback

        print(f"⚠️ 可視化の生成中にエラーが発生しました: {e}")
        print("詳細なエラー情報:")
        traceback.print_exc()
