# ML Pipeline Refactoring: Modularization and Sampling Enhancement

## 概要 (Overview)

元の3245行の巨大なMLパイプラインファイル（06_draftmodel2.py）を機能別の8つのモジュールに分割し、テストデータのサンプリング方法を選択できる新機能を追加しました。

## 主な変更点 (Key Changes)

### 1. モジュール分割 (Module Separation)
- **data_loader.py**: データベース操作とテーブル結合機能
- **preprocessor.py**: 特徴量エンジニアリングとデータ変換
- **hyperparameter_tuner.py**: Optuna/Grid/Random探索による最適化
- **model_evaluator.py**: 性能評価とバリデーション
- **visualizer.py**: グラフ作成と分析プロット
- **model_persistence.py**: モデルと結果の保存・読み込み
- **session_manager.py**: セッション間の状態管理
- **utils.py**: 共通ユーティリティ関数

### 2. 新しいサンプリング機能 (New Sampling Feature)
- テストデータ分割時にランダムサンプリングまたは固定データを選択可能
- `_get_sampling_method_config()`関数でユーザーが選択
- 再現性が必要な場合は固定random_stateを使用
- 実験的な用途ではランダムサンプリングを選択可能

### 3. コード品質向上 (Code Quality Improvements)
- メインファイルを3245行から約170行に削減（95%削減）
- 既存の全機能とインタラクティブワークフローを維持
- 適切な型ヒントとエラーハンドリングを追加
- モジュール間の明確な責任分離

## ファイル構造 (File Structure)

```
ml_pipeline_refactor/
├── 06_draftmodel2.py          # メインファイル（大幅に縮小）
├── modules/
│   ├── __init__.py
│   ├── data_loader.py         # データ読み込み・結合
│   ├── preprocessor.py        # 前処理・特徴量生成
│   ├── hyperparameter_tuner.py # ハイパーパラメータ最適化
│   ├── model_evaluator.py     # モデル評価・新サンプリング機能
│   ├── visualizer.py          # 可視化・分析
│   ├── model_persistence.py   # モデル保存・読み込み
│   ├── session_manager.py     # セッション管理
│   └── utils.py              # ユーティリティ関数
└── test_modules.py           # モジュールテストスイート
```

## 新機能の使用方法 (How to Use New Features)

### サンプリング方法の選択
モデル評価時に以下の選択肢が表示されます：

1. **ランダムサンプリング** - 毎回異なるランダムな分割
2. **固定データ** - 常に同じ分割（再現性重視）

```python
# 内部実装例
sampling_config = _get_sampling_method_config()
train_data, test_data, _, _ = split_train_test_data(
    data, target_column, test_size=0.2,
    sampling_method=sampling_config["method"],
    random_state=sampling_config["random_state"]
)
```

## 互換性 (Compatibility)

- 既存の全機能を完全に保持
- 同じユーザーインターフェースとワークフロー
- 既存のデータベースファイルとの互換性を維持
- LightGBM、Optuna、その他の依存関係は変更なし

## テスト (Testing)

`test_modules.py`を実行してモジュールの正常性を確認：

```bash
python test_modules.py
```

## 利点 (Benefits)

1. **保守性向上**: 機能別に分離されたコードで修正・拡張が容易
2. **再利用性**: 個別モジュールを他のプロジェクトで再利用可能
3. **テスト容易性**: 各モジュールを独立してテスト可能
4. **可読性向上**: 小さなファイルで理解しやすい構造
5. **新機能**: 実験ニーズに応じたサンプリング方法の選択

## Link to Devin run
https://app.devin.ai/sessions/ee649d2448f14d919bee86e20adc7be6

## Requested by
AJ1905 (jun.akita57@gmail.com)
