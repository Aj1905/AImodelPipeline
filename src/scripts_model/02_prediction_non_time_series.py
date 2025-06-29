"""
保存済みモデルを使用した予測実行スクリプト

実行例:
python src/scripts_model/02_prediction_non_time_series.py \
    --db-path data/database.sqlite \
    --predict-table new_data \
    --output-table predictions

python src/scripts_model/02_prediction_non_time_series.py \
    --db-path data/database.sqlite \
    --predict-table sales_test \
    --output-table sales_predictions

python src/scripts_model/02_prediction_non_time_series.py \
    --db-path data/database.sqlite \
    --predict-table customer_new \
    --output-table customer_predictions

オプション:
    --db-path: SQLiteデータベースファイルのパス
    --predict-table: 入力テーブル名（目的変数列は不要）
    --output-table: 予測結果を書き込むテーブル名
"""

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import polars as pl

# アーティファクトディレクトリ
ARTIFACT_DIR = Path("artifacts")

# 既存の列を特徴量として利用するリスト（01_model_non_time_series.pyと同じ）
FEATURE_COLUMNS = [
    "date_and_time",
    "prev_year_same_weekday_sales_daily_sum",
    "Yokohama_Sea_Level_Pressure",
    "time",
    "prev_year_same_date_sales_daily_sum",
    "time_flag",
    "date_day",
    "sales_trailing_ma_11",
    "Yokohama_Temperature",
    "Yokohama_Wind_Speed",
    "prev_year_same_weekday_sales",
    "sales_trailing_ma_1",
    "prev_year_same_weekday_sales_monthly_sum",
    "Yokohama_Relative_Humidity",
    "sales_trailing_ma_3",
    "prev_year_same_date_sales",
]

# feature_engineering関数で作成する特徴量の名前（01_model_non_time_series.pyと同じ）
ENGINEERED_FEATURES = [
    "is_lunch",
    "weekday",
    "time_category",
    "season"
]

def feature_engineering(data: pl.DataFrame) -> pl.DataFrame:
    """一部の列を変換し、新たな特徴量列を返す（01_model_non_time_series.pyと同じ）"""
    df = data.clone()

    # 日付をdatetime型に変換
    df = df.with_columns(pl.col("date_and_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime"))

    # 新しい特徴量を作成
    engineered = df.with_columns(
        [
            pl.when(pl.col("time").is_in([11, 12, 13])).then(1).otherwise(0).alias("is_lunch"),
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.when(pl.col("time") < 12)
            .then(0)
            .when(pl.col("time") < 15)
            .then(1)
            .when(pl.col("time") < 18)
            .then(2)
            .otherwise(3)
            .alias("time_category"),
        ]
    )

    # date_month列が存在する場合のみseason特徴量を作成
    if "date_month" in df.columns:
        engineered = engineered.with_columns([
            pl.when(pl.col("date_month").is_in([12, 1, 2]))
            .then(0)
            .when(pl.col("date_month").is_in([3, 4, 5]))
            .then(1)
            .when(pl.col("date_month").is_in([6, 7, 8]))
            .then(2)
            .otherwise(3)
            .alias("season"),
        ])
    else:
        # date_month列が存在しない場合は季節を0で初期化
        engineered = engineered.with_columns(pl.lit(0).alias("season"))

    # 欠損値処理
    result = engineered.select(ENGINEERED_FEATURES)
    
    # Polarsで欠損値を処理
    for col in ENGINEERED_FEATURES:
        if col in result.columns:
            # 数値列の場合は中央値で補完
            result = result.with_columns(
                pl.col(col).fill_null(pl.col(col).median()).alias(col)
            )

    return result

def main():
    parser = argparse.ArgumentParser(description="保存済みモデルを読み込み、本番データで予測を実行します")
    parser.add_argument("--db-path",       type=str, required=True, help="SQLiteデータベースファイルのパス")
    parser.add_argument("--predict-table", type=str, required=True, help="入力テーブル名（目的変数列は不要）")
    parser.add_argument("--output-table",  type=str, required=True, help="予測結果を書き込むテーブル名")
    args = parser.parse_args()

    # アーティファクト読み込み
    fm       = joblib.load(ARTIFACT_DIR/"feature_manager.pkl")
    tm       = joblib.load(ARTIFACT_DIR/"target_manager.pkl")
    model    = joblib.load(ARTIFACT_DIR/"model.pkl")
    metadata = json.load(open(ARTIFACT_DIR/"metadata.json", encoding="utf-8"))

    print("📊 メタデータ情報:")
    print(f"   元の特徴量: {metadata.get('original_features', [])}")
    print(f"   エンジニアリング特徴量: {metadata.get('engineered_features', [])}")

    # 本番データ読み込み
    conn = sqlite3.connect(args.db_path)
    df = pd.read_sql(f"SELECT * FROM {args.predict_table}", conn)
    conn.close()
    print(f"✅ データ読込完了: {len(df)} 行, {len(df.columns)} 列")

    # 指定された特徴量のみを選択
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    if missing_features:
        print(f"⚠️ 指定された特徴量が見つかりません: {missing_features}")

    print(f"📊 使用する特徴量: {available_features}")

    # 指定された特徴量のみを選択
    df_selected = df[available_features].copy()

    # 欠損値処理
    print("🔧 欠損値処理実行中...")
    for col in df_selected.columns:
        if df_selected[col].dtype in ['int64', 'float64']:
            # 数値列の場合
            if df_selected[col].isnull().sum() > 0:
                median_val = df_selected[col].median()
                df_selected[col] = df_selected[col].fillna(median_val)
                print(f"  {col}: 中央値 ({median_val:.2f}) で補完")
        else:
            # カテゴリ列の場合
            if df_selected[col].isnull().sum() > 0:
                mode_val = df_selected[col].mode().iloc[0] if not df_selected[col].mode().empty else 'Unknown'
                df_selected[col] = df_selected[col].fillna(mode_val)
                print(f"  {col}: 最頻値 ({mode_val}) で補完")

    # 特徴量エンジニアリング
    print("🔧 特徴量エンジニアリング実行中...")
    df_pl = pl.from_pandas(df_selected)
    engineered_features = feature_engineering(df_pl)
    engineered_features_pd = engineered_features.to_pandas()

    # 元の特徴量とエンジニアリングされた特徴量を結合
    X_input = pd.concat([df_selected[available_features], engineered_features_pd], axis=1)

    print(f"✅ 特徴量エンジニアリング完了: {len(X_input.columns)} 列")
    print(f"   元の特徴量: {len(available_features)} 列")
    print(f"   エンジニアリング特徴量: {len(ENGINEERED_FEATURES)} 列")

    # 前処理 → 予測
    X_processed = fm.transform(X_input)
    y_pred = model.predict(X_processed)

    # 予測結果を元のデータフレームに追加
    target_col = metadata.get("target_column", "sales")
    df[target_col] = y_pred

    # 結果書き戻し
    conn = sqlite3.connect(args.db_path)
    df.to_sql(args.output_table, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ 予測結果をテーブル '{args.output_table}' に保存しました")
    print("📊 予測値の統計:")
    print(f"   平均: {y_pred.mean():.2f}")
    print(f"   標準偏差: {y_pred.std():.2f}")
    print(f"   最小値: {y_pred.min():.2f}")
    print(f"   最大値: {y_pred.max():.2f}")

if __name__ == "__main__":
    main()
