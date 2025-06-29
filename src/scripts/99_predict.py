"""
予測実行スクリプト

実行コマンド例:
    python src/scripts/99_predict.py --db-path data/database.sqlite --table my_table --features col1 col2 col3 --model-path model.joblib --output predictions.csv
    python src/scripts/99_predict.py --db-path data/database.sqlite --table my_table --features feature1 feature2 --model-path trained_model/lightgbm_model.pkl
    python src/scripts/99_predict.py --db-path data/database.sqlite --table sales_data --features sales_trailing_ma_3 sales_trailing_ma_14 --model-path final_model.joblib --output sales_predictions.csv
"""

import argparse
import sqlite3
import pandas as pd
import joblib
from hyper_utils import feature_engineering


def load_sqlite_data(db_path: str, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--db-path', required=True,
                   help="SQLiteデータベースファイルのパス")
    p.add_argument('--table', required=True,
                   help="読み込むテーブル名")
    p.add_argument('--features', nargs='+', required=True,
                   help="予測に使う特徴量のカラムリスト")
    p.add_argument('--model-path', required=True,
                   help="学習済みモデルファイルのパス")
    p.add_argument('--output', default='predictions.csv',
                   help="予測結果を保存するCSVファイル名")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_sqlite_data(args.db_path, args.table)
    df_feat = feature_engineering(df, args.features)
    model = joblib.load(args.model_path)
    preds = model.predict(df_feat)
    df['prediction'] = preds
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()
