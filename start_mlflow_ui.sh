#!/bin/bash

# MLflow UI起動スクリプト

echo "🚀 MLflow UIを起動中..."

# 仮想環境を有効化
source .venv/bin/activate

# MLflow UIを起動
echo "📊 MLflow UIを起動しました"
echo "🌐 ブラウザで http://localhost:5000 にアクセスしてください"
echo "🛑 停止するには Ctrl+C を押してください"

python -m mlflow ui 