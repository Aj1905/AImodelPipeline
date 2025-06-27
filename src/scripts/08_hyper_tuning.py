#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml import FeatureManager, HyperTuner, TargetManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    feature_manager = FeatureManager()
    target_manager = TargetManager()
    x = feature_manager.transform(df.drop(columns=[args.target_column]))
    y = target_manager.transform(df[args.target_column])

    def estimator_factory():
        return LGBMRegressor()

    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}

    tuner = HyperTuner(
        estimator_factory=estimator_factory,
        param_grid=param_grid,
        outer_splits=args.outer_splits,
        inner_splits=3,
    )
    result = tuner.tune(x, y)
    print(f"最適パラメータ: {result['best_params']}")
    print(f"平均スコア: {result['avg_score']:.4f}")


if __name__ == "__main__":
    main()
