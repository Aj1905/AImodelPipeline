#!/bin/bash

# 機械学習スクリプト実行ヘルパー

echo "🚀 機械学習スクリプトを実行中..."

# 仮想環境を有効化
source .venv/bin/activate

# 引数をそのまま渡してスクリプトを実行
python src/scripts/07_concise_draftmodel.py "$@"

echo "✅ 学習完了!"
echo ""
echo "📊 MLflow UIで結果を確認するには:"
echo "  ./start_mlflow_ui.sh"
echo "  または"
echo "  python -m mlflow ui" 