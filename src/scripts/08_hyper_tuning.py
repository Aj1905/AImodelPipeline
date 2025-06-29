"""
ハイパーパラメータチューニングスクリプト

実行コマンド例:
    python src/scripts/08_hyper_tuning.py --data-path data.csv --n-trials 100
    python src/scripts/08_hyper_tuning.py --data-path data.csv --n-trials 50
    python src/scripts/08_hyper_tuning.py --data-path processed_data.csv --n-trials 200
"""

import argparse
from hyper_utils import load_data, feature_engineering, tune_hyperparams

# 使用する特徴量を直書き
FEATURES = ['feature1', 'feature2', 'feature3']  # 必要に応じて修正
TARGET = 'target_column'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True,
                   help="CSV ファイルのパス")
    p.add_argument('--n-trials', type=int, default=50,
                   help="ハイパーチューニングの試行回数")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data_path)
    X, y, _ = feature_engineering(df, FEATURES, TARGET)
    best_params = tune_hyperparams(X, y, args.n_trials)
    print("Best params:", best_params)


if __name__ == '__main__':
    main()
