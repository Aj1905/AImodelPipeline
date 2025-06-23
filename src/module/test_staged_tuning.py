"""
段階チューニング機能のテスト
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.ml.module_3.hyperparameter_tuner import _get_continuation_config, _get_custom_stage_config
from src.ml.module_3.tuning_manager import TuningManager


class TestTuningManager:
    """TuningManagerのテスト"""

    def setup_method(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TuningManager(self.temp_dir)

        # テスト用データ
        self.test_data = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100), "target": np.random.randn(100)}
        )

        self.test_results = {
            "best_params": {"learning_rate": 0.1, "max_depth": 6, "num_leaves": 31},
            "best_score": 0.85,
            "strategy": "Optuna (3fold CV)",
            "cv_config": {"cv_method": "3fold", "cv_folds": 3},
        }

    def teardown_method(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir)

    def test_create_project(self):
        """プロジェクト作成のテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        assert project_id is not None
        assert project_id in self.manager.stage_history["projects"]

        project_info = self.manager.stage_history["projects"][project_id]
        assert project_info["project_name"] == "test_project"
        assert project_info["target_column"] == "target"
        assert project_info["feature_columns"] == ["feature1", "feature2"]
        assert project_info["feature_count"] == 2
        assert project_info["current_stage"] == 0
        assert len(project_info["stages"]) == 0

    def test_save_stage_result(self):
        """段階結果保存のテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        result_path = self.manager.save_stage_result(project_id, "test_stage", self.test_results)

        assert result_path != ""
        assert Path(result_path).exists()

        project_info = self.manager.stage_history["projects"][project_id]
        assert project_info["current_stage"] == 1
        assert len(project_info["stages"]) == 1
        assert project_info["best_score"] == 0.85
        assert project_info["best_params"] == self.test_results["best_params"]

    def test_load_stage_result(self):
        """段階結果読み込みのテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        self.manager.save_stage_result(project_id, "test_stage", self.test_results)

        loaded_results = self.manager.load_stage_result(project_id, 1)

        assert loaded_results is not None
        assert loaded_results["best_score"] == self.test_results["best_score"]
        assert loaded_results["best_params"] == self.test_results["best_params"]

    def test_get_project_summary(self):
        """プロジェクト概要取得のテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        summary = self.manager.get_project_summary(project_id)

        assert summary is not None
        assert summary["project_name"] == "test_project"
        assert summary["target_column"] == "target"
        assert summary["feature_count"] == 2
        assert summary["current_stage"] == 0
        assert summary["total_stages"] == 0

    def test_list_projects(self):
        """プロジェクト一覧取得のテスト"""
        # 複数のプロジェクトを作成
        self.manager.create_project("project1", "target1", ["feature1"])
        self.manager.create_project("project2", "target2", ["feature1", "feature2"])

        projects = self.manager.list_projects()

        assert len(projects) == 2
        project_names = [p["project_name"] for p in projects]
        assert "project1" in project_names
        assert "project2" in project_names

    def test_suggest_next_stage_params(self):
        """次の段階のパラメータ提案のテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        # 最初の段階では元の範囲を返す
        param_ranges = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 12},
        }

        suggested = self.manager.suggest_next_stage_params(project_id, param_ranges)
        assert suggested == param_ranges

        # 段階結果を保存
        self.manager.save_stage_result(project_id, "test_stage", self.test_results)

        # 次の段階では範囲が狭められる
        suggested = self.manager.suggest_next_stage_params(project_id, param_ranges)

        # learning_rateの範囲が狭められていることを確認
        original_range = param_ranges["learning_rate"]["high"] - param_ranges["learning_rate"]["low"]
        suggested_range = suggested["learning_rate"]["high"] - suggested["learning_rate"]["low"]
        assert suggested_range < original_range

    def test_get_optimal_params_for_next_stage(self):
        """次の段階の最適パラメータ取得のテスト"""
        project_id = self.manager.create_project("test_project", "target", ["feature1", "feature2"])

        # 段階がない場合はNone
        params = self.manager.get_optimal_params_for_next_stage(project_id)
        assert params is None

        # 段階結果を保存
        self.manager.save_stage_result(project_id, "test_stage", self.test_results)

        # 最適パラメータを取得
        params = self.manager.get_optimal_params_for_next_stage(project_id)
        assert params == self.test_results["best_params"]

    def test_invalid_project_id(self):
        """無効なプロジェクトIDのテスト"""
        with pytest.raises(ValueError):
            self.manager.save_stage_result("invalid_id", "test_stage", self.test_results)

        result = self.manager.load_stage_result("invalid_id")
        assert result is None

        summary = self.manager.get_project_summary("invalid_id")
        assert summary is None


class TestHyperparameterTuner:
    """ハイパーパラメータチューナーのテスト"""

    @patch("builtins.input")
    def test_get_custom_stage_config(self, mock_input):
        """カスタム段階設定取得のテスト"""
        mock_input.side_effect = [
            "custom_stage",  # 段階名
            "2",  # kfold選択
            "5",  # CV分割数
            "100",  # 試行回数
        ]

        config = _get_custom_stage_config()

        assert config["stage_name"] == "custom_stage"
        assert config["cv_method"] == "kfold"
        assert config["cv_folds"] == 5
        assert config["n_trials"] == 100

    @patch("builtins.input")
    def test_get_continuation_config(self, mock_input):
        """継続チューニング設定取得のテスト"""
        mock_input.side_effect = [
            "continuation_stage",  # 段階名
            "1",  # 3fold選択
            "50",  # 試行回数
        ]

        config = _get_continuation_config()

        assert config["stage_name"] == "continuation_stage"
        assert config["cv_method"] == "3fold"
        assert config["cv_folds"] == 3
        assert config["n_trials"] == 50


class TestIntegration:
    """統合テスト"""

    def setup_method(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TuningManager(self.temp_dir)

    def teardown_method(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir)

    def test_full_workflow(self):
        """完全なワークフローのテスト"""
        # 1. プロジェクト作成
        project_id = self.manager.create_project("integration_test", "target", ["feature1", "feature2"])

        # 2. 最初の段階結果保存
        stage1_results = {
            "best_params": {"learning_rate": 0.1, "max_depth": 6},
            "best_score": 0.8,
            "strategy": "Optuna (3fold CV)",
            "cv_config": {"cv_method": "3fold", "cv_folds": 3},
        }

        self.manager.save_stage_result(project_id, "stage1", stage1_results)

        # 3. 2番目の段階結果保存
        stage2_results = {
            "best_params": {"learning_rate": 0.15, "max_depth": 7},
            "best_score": 0.85,
            "strategy": "Optuna (5fold CV)",
            "cv_config": {"cv_method": "kfold", "cv_folds": 5},
        }

        self.manager.save_stage_result(project_id, "stage2", stage2_results)

        # 4. プロジェクト概要確認
        summary = self.manager.get_project_summary(project_id)
        assert summary["current_stage"] == 2
        assert summary["total_stages"] == 2
        assert summary["best_score"] == 0.85  # より良いスコアが記録される

        # 5. 履歴表示
        self.manager.display_project_history(project_id)

        # 6. 次の段階のパラメータ提案
        param_ranges = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
            "max_depth": {"type": "int", "low": 3, "high": 12},
        }

        suggested = self.manager.suggest_next_stage_params(project_id, param_ranges)
        assert suggested != param_ranges  # 範囲が調整されている

        # 7. レポート出力
        report_path = Path(self.temp_dir) / "test_report.html"
        self.manager.export_project_report(project_id, str(report_path))
        assert report_path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
