#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import pandas as pd
from lightgbm import LGBMRegressor

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml import FeatureManager, HyperTuner, TargetManager, TreeModelPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    feature_manager = FeatureManager()
    target_manager = TargetManager()
    X = feature_manager.transform(df.drop(columns=[args.target_column]))
    y = target_manager.transform(df[args.target_column])

    estimator_factory = lambda: LGBMRegressor()
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
    tuner = HyperTuner(estimator_factory, param_grid, outer_splits=args.outer_splits, inner_splits=3)
    tune_result = tuner.tune(X, y)

    best_params = tune_result["best_params"]
    model = LGBMRegressor(**best_params)
    pipeline = TreeModelPipeline(model=model, feature_manager=feature_manager, target_manager=target_manager)
    results = pipeline.train(test_size=0.1, random_state=42)
    print(f"最終モデルテストR²: {results['test_r2']:.4f}")


if __name__ == "__main__":
    main()
