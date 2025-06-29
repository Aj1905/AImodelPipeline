"""
モデル学習スクリプト

実行コマンド例:
    python src/scripts/99_train.py --data-path data.csv --tune --n-trials 200 --model-output model.joblib
    python src/scripts/99_train.py --data-path data.csv --manual-params '{"param1": 0.1, "param2": 100}' --model-output my_model.joblib
    python src/scripts/99_train.py --data-path processed_data.csv --tune --n-trials 50 --model-output final_model.joblib
    python src/scripts/99_train.py --data-path data.sqlite --tables mores_sales weather_yokohama --tune --n-trials 100 --model-output model.joblib

"""

import argparse
import json
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from hyper_utils import load_data, feature_engineering, tune_hyperparams, train_model

# 使用する特徴量と目的変数を直書き
FEATURES = [
    "dow_flag",
    "prev_year_same_weekday_sales_daily_sum",
    "time_flag",
    "Yokohama_Temperature",
    "is_weekend",
    "holiday",
    "sales_lag153_ma3",
    "sales_lag153_ma94",
]
TARGET = 'sales'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True,
                   help="CSV ファイルまたは SQLite データベースのパス")
    p.add_argument('--tables', nargs='+', default=['mores_sales', 'weather_yokohama'],
                   help="読み込むテーブル名のリスト")
    p.add_argument('--tune', action='store_true',
                   help="チューニングを行う場合に指定")
    p.add_argument('--manual-params', type=str, default=None,
                   help="JSON 形式での手動パラメータ")
    p.add_argument('--n-trials', type=int, default=50,
                   help="チューニングの試行回数")
    p.add_argument('--model-output', type=str, default='model.joblib',
                   help="保存先モデルファイル名")
    return p.parse_args()


def load_data_with_tables(data_path, tables):
    """テーブル名を指定してデータを読み込む"""
    from pathlib import Path
    path_obj = Path(data_path)
    
    if path_obj.suffix.lower() in ['.sqlite', '.db']:
        # SQLiteデータベースの場合
        return load_sqlite_data_with_tables(data_path, tables)
    else:
        # CSVファイルの場合
        import pandas as pd
        return pd.read_csv(data_path)


def load_sqlite_data_with_tables(db_path, tables):
    """指定されたテーブル名でSQLiteデータベースから複数のテーブルを結合して読み込む"""
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect(db_path)
    
    try:
        # 利用可能なテーブルを確認
        tables_df = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        available_tables = tables_df['name'].tolist()
        print(f"利用可能なテーブル: {available_tables}")
        
        # 指定されたテーブルが存在するかチェック
        for table in tables:
            if table not in available_tables:
                raise ValueError(f"テーブル '{table}' が見つかりません。利用可能なテーブル: {available_tables}")
        
        # 各テーブルを読み込み
        dataframes = {}
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            dataframes[table] = df
            print(f"テーブル {table}: {df.shape}")
        
        # テーブルを結合（日付列で結合を試行）
        merged_df = None
        
        if len(tables) == 1:
            # テーブルが1つの場合
            merged_df = dataframes[tables[0]]
        else:
            # 複数テーブルの場合、日付列で結合を試行
            merged_df = dataframes[tables[0]]
            
            for i, table in enumerate(tables[1:], 1):
                df = dataframes[table]
                
                # 日付列を探す
                date_columns = []
                for col in df.columns:
                    if any(date_keyword in col.lower() for date_keyword in ['date', 'time', 'datetime']):
                        date_columns.append(col)
                
                if date_columns:
                    # 日付列が見つかった場合、最初の日付列を使用
                    date_col = date_columns[0]
                    print(f"テーブル {table} の日付列 '{date_col}' で結合")
                    
                    # 日付列を統一
                    if 'date' in merged_df.columns:
                        left_date_col = 'date'
                    elif 'Date_and_Time' in merged_df.columns:
                        left_date_col = 'Date_and_Time'
                    else:
                        # 最初のテーブルから日付列を探す
                        left_date_columns = []
                        for col in merged_df.columns:
                            if any(date_keyword in col.lower() for date_keyword in ['date', 'time', 'datetime']):
                                left_date_columns.append(col)
                        left_date_col = left_date_columns[0] if left_date_columns else None
                    
                    if left_date_col:
                        merged_df[left_date_col] = pd.to_datetime(merged_df[left_date_col])
                        df[date_col] = pd.to_datetime(df[date_col])
                        
                        # 結合
                        merged_df = pd.merge(
                            merged_df, 
                            df, 
                            left_on=left_date_col, 
                            right_on=date_col, 
                            how='left',
                            suffixes=('', f'_{table}')
                        )
                    else:
                        # 日付列が見つからない場合は単純に結合
                        merged_df = pd.concat([merged_df, df], axis=1)
                else:
                    # 日付列が見つからない場合は単純に結合
                    merged_df = pd.concat([merged_df, df], axis=1)
        
        print(f"結合後データ: {merged_df.shape}")
        print(f"結合後のカラム: {list(merged_df.columns)}")
        
        return merged_df
        
    finally:
        conn.close()


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """モデルの詳細な評価を行う"""
    # 予測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 訓練データの評価
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # テストデータの評価
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("学習結果")
    print("="*50)
    print(f"訓練データ:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"\nテストデータ:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    # 特徴量重要度の表示
    if hasattr(model, 'feature_importances_'):
        print(f"\n特徴量重要度 (上位10件):")
        feature_importance = list(zip(model.feature_names_, model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    return {
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae
    }


def main():
    args = parse_args()
    
    print("データ読み込み中...")
    if args.data_path.endswith(('.sqlite', '.db')):
        df = load_data_with_tables(args.data_path, args.tables)
    else:
        df = load_data(args.data_path)
    print(f"データ形状: {df.shape}")
    
    print("特徴量エンジニアリング中...")
    X, y, _ = feature_engineering(df, FEATURES, TARGET)
    print(f"特徴量数: {X.shape[1]}")
    print(f"サンプル数: {X.shape[0]}")

    if args.tune:
        print(f"\nハイパーパラメータチューニング開始 (試行回数: {args.n_trials})...")
        params = tune_hyperparams(X, y, args.n_trials)
        print(f"最適パラメータ: {params}")
    else:
        if not args.manual_params:
            raise ValueError("手動パラメータを --manual-params で渡してください")
        params = json.loads(args.manual_params)
        print(f"使用パラメータ: {params}")

    print("\nモデル学習中...")
    model = train_model(X, y, params)
    
    # 詳細な評価を実行
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluation_results = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # モデル保存
    joblib.dump(model, args.model_output)
    print(f"\nモデルを {args.model_output} に保存しました")
    
    # 評価結果も保存
    results_file = args.model_output.replace('.joblib', '_results.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"評価結果を {results_file} に保存しました")


if __name__ == "__main__":
    main()
