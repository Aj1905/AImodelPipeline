# MLflow統合機械学習スクリプト

このプロジェクトでは、機械学習の実験結果をMLflowで追跡・管理できます。

## 🚀 クイックスタート

### 1. 学習の実行

```bash
# 基本的な使用方法（MLflow有効）
./run_training.sh

# または直接実行
source .venv/bin/activate
python src/scripts/07_concise_draftmodel.py
```

### 2. MLflow UIで結果を確認

```bash
# ヘルパースクリプトを使用
./start_mlflow_ui.sh

# または直接実行
source .venv/bin/activate
python -m mlflow ui
```

その後、ブラウザで `http://localhost:5000` にアクセス

## 📊 MLflowに記録される情報

### パラメータ
- テーブル名
- ターゲット列
- 特徴量数
- データサイズ
- 分割方法（時系列/ランダム）
- モデル設定（LightGBM）

### メトリクス
- 学習・テストのRMSE
- 学習・テストのR²
- 学習・テストのMAE
- 特徴量重要度（上位10個）

### アーティファクト
- 学習済みモデルファイル（.pkl）
- 設定ファイル（.json）

### タグ
- 学習時のコメント

## 🔧 オプション

### 基本的なオプション
```bash
# MLflowを無効にする
./run_training.sh --no-mlflow

# カスタム実験名を指定
./run_training.sh --experiment-name "my_experiment"

# カスタム実行名を指定
./run_training.sh --run-name "experiment_001"

# 時系列分割を使用
./run_training.sh --time-series-split

# モデルを保存しない
./run_training.sh --no-save
```

### データベース関連
```bash
# カスタムデータベースパス
./run_training.sh --db-path "path/to/database.sqlite"

# テーブル名を指定
./run_training.sh --table "my_table"

# ターゲット列を指定
./run_training.sh --target-column "target"

# 特徴量列を指定
./run_training.sh --feature-columns "feature1" "feature2" "feature3"
```

## 📁 ファイル構成

```
AImodelPipeline/
├── src/scripts/07_concise_draftmodel.py  # メイン学習スクリプト
├── run_training.sh                       # 学習実行ヘルパー
├── start_mlflow_ui.sh                    # MLflow UI起動ヘルパー
├── mlflow.db                             # MLflowデータベース
├── mlruns/                               # MLflow実行履歴
└── trained_model/                        # 保存されたモデル
```

## 🔍 MLflow UIでの確認方法

1. `./start_mlflow_ui.sh` を実行
2. ブラウザで `http://localhost:5000` にアクセス
3. 実験一覧から該当する実験を選択
4. 実行履歴から詳細な結果を確認

### 確認できる情報
- パラメータの比較
- メトリクスの推移
- 特徴量重要度
- 保存されたモデルファイル
- 学習時のコメント

## 🛠️ トラブルシューティング

### 仮想環境の問題
```bash
# 仮想環境を有効化
source .venv/bin/activate

# 依存関係を確認
pip list | grep mlflow
```

### MLflow UIが起動しない場合
```bash
# ポートが使用中の場合、別のポートを指定
python -m mlflow ui --port 5001
```

### データベースが見つからない場合
```bash
# データベースファイルの存在確認
ls -la data/database.sqlite
```

## 📝 例

### 基本的な学習実行
```bash
./run_training.sh
```

### カスタム設定での学習
```bash
./run_training.sh \
  --experiment-name "restaurant_forecast" \
  --run-name "v1_lightgbm" \
  --time-series-split \
  --table "sales_data" \
  --target-column "sales" \
  --feature-columns "temperature" "humidity" "day_of_week"
```

### 結果の確認
```bash
./start_mlflow_ui.sh
# ブラウザで http://localhost:5000 にアクセス
``` 