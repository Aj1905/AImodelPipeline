import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split
import optuna
from typing import Tuple, List, Optional


def load_data(path: str) -> pd.DataFrame:
    """CSV または SQLite から DataFrame を読み込む"""
    path_obj = Path(path)
    
    if path_obj.suffix.lower() == '.sqlite' or path_obj.suffix.lower() == '.db':
        # SQLiteデータベースの場合
        return load_sqlite_data(path)
    else:
        # CSVファイルの場合
        return pd.read_csv(path)


def load_sqlite_data(db_path: str) -> pd.DataFrame:
    """SQLiteデータベースから売上データと天気データを結合して読み込む"""
    conn = sqlite3.connect(db_path)
    
    try:
        # 売上データを読み込み
        sales_df = pd.read_sql_query("SELECT * FROM mores_sales", conn)
        
        # 天気データを読み込み
        weather_df = pd.read_sql_query("SELECT * FROM weather_yokohama", conn)
        
        # 日付列を統一（mores_sales.date と weather_yokohama.Date_and_Time）
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        weather_df['Date_and_Time'] = pd.to_datetime(weather_df['Date_and_Time'])
        
        # 日付で結合
        merged_df = pd.merge(
            sales_df, 
            weather_df, 
            left_on='date', 
            right_on='Date_and_Time', 
            how='left'
        )
        
        print(f"売上データ: {sales_df.shape}")
        print(f"天気データ: {weather_df.shape}")
        print(f"結合後データ: {merged_df.shape}")
        
        return merged_df
        
    finally:
        conn.close()


def feature_engineering(
    df: pd.DataFrame,
    feature_list: List[str],
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    指定された特徴量リストに基づき新規特徴量を追加・抽出し、
    X, y, 使用した特徴量リストを返す。

    Args:
        df: 元データの DataFrame
        feature_list: 学習に使う説明変数のカラム名リスト
        target_col: 目的変数のカラム名 (指定があれば y を返す)

    Returns:
        X: 説明変数 DataFrame
        y: 目的変数 Series (target_col が None の場合は None)
        feature_list: 実際に使った説明変数リスト
    """
    # ここで必要な新規特徴量を生成可能
    df_feat = df.copy()

    # 日付列の処理
    date_columns = []
    for col in feature_list:
        if col in df_feat.columns:
            if any(date_keyword in col.lower() for date_keyword in ['date', 'time', 'datetime']):
                date_columns.append(col)
    
    # 日付列を数値特徴量に変換
    for col in date_columns:
        if col in df_feat.columns:
            try:
                # 日付列をdatetime型に変換
                df_feat[col] = pd.to_datetime(df_feat[col])
                
                # 日付から数値特徴量を抽出
                df_feat[f'{col}_year'] = df_feat[col].dt.year
                df_feat[f'{col}_month'] = df_feat[col].dt.month
                df_feat[f'{col}_day'] = df_feat[col].dt.day
                df_feat[f'{col}_dayofweek'] = df_feat[col].dt.dayofweek
                df_feat[f'{col}_hour'] = df_feat[col].dt.hour
                df_feat[f'{col}_minute'] = df_feat[col].dt.minute
                
                # 元の日付列を削除
                df_feat = df_feat.drop(columns=[col])
                
                # 特徴量リストから元の日付列を削除し、新しい数値特徴量を追加
                feature_list = [f for f in feature_list if f != col]
                feature_list.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', 
                    f'{col}_dayofweek', f'{col}_hour', f'{col}_minute'
                ])
                
                print(f"✅ 日付列 '{col}' を数値特徴量に変換しました")
                
            except Exception as e:
                print(f"⚠️ 日付列 '{col}' の変換に失敗しました: {e}")
                # 変換に失敗した場合は元の列を削除
                df_feat = df_feat.drop(columns=[col])
                feature_list = [f for f in feature_list if f != col]

    # 欠損値の確認
    print(f"欠損値の確認:")
    for col in feature_list:
        if col in df_feat.columns:
            missing_count = df_feat[col].isnull().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} 個の欠損値")

    # 使用可能な特徴量のみを選択
    available_features = [col for col in feature_list if col in df_feat.columns]
    missing_features = [col for col in feature_list if col not in df_feat.columns]
    
    if missing_features:
        print(f"⚠️ 以下の特徴量が見つかりません: {missing_features}")
    
    X = df_feat[available_features].copy()
    
    # データ型の確認と変換
    print(f"データ型の確認:")
    for col in X.columns:
        print(f"  {col}: {X[col].dtype}")
        # object型の場合は数値に変換を試行
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                print(f"    → 数値型に変換しました")
            except:
                print(f"    ⚠️ 数値型への変換に失敗しました")
    
    # 欠損値処理
    # 数値列の場合は中央値で補完、カテゴリ列の場合は最頻値で補完
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # 数値列の場合
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                print(f"  {col}: 中央値 ({median_val:.2f}) で補完")
        else:
            # カテゴリ列の場合
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'
                X[col] = X[col].fillna(mode_val)
                print(f"  {col}: 最頻値 ({mode_val}) で補完")
    
    # 目的変数も同様に処理
    y = None
    if target_col is not None and target_col in df_feat.columns:
        y = df_feat[target_col].copy()
        if y.isnull().sum() > 0:
            print(f"⚠️ 目的変数 {target_col} に {y.isnull().sum()} 個の欠損値があります")
            # 目的変数に欠損値がある場合は該当行を削除
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
            print(f"  欠損値を含む行を削除: {len(df_feat)} → {len(X)} 行")
    
    print(f"✅ 前処理完了: {X.shape[1]} 特徴量, {X.shape[0]} サンプル")
    
    return X, y, available_features


def get_search_space(trial: optuna.Trial) -> dict:
    """
    Optuna Trial から LightGBM 用のハイパーパラメータ探索空間を定義
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    }


def tune_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50
) -> dict:
    """
    Optuna を用いてハイパーパラメータをチューニングし、最良パラメータを返す
    """
    import lightgbm as lgb

    def objective(trial):
        params = get_search_space(trial)
        dtrain = lgb.Dataset(X, label=y)
        cv_results = lgb.cv(
            params, dtrain, nfold=3, seed=42,
            metrics=['rmse'], stratified=False
        )
        # LightGBMのCV結果のキー名を確認して修正
        if 'valid rmse-mean' in cv_results:
            return min(cv_results['valid rmse-mean'])
        elif 'rmse-mean' in cv_results:
            return min(cv_results['rmse-mean'])
        else:
            # 利用可能なキーを確認
            print(f"利用可能なキー: {list(cv_results.keys())}")
            return float('inf')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # チューニング結果の詳細表示
    print(f"\nチューニング結果:")
    print(f"  最良スコア (RMSE): {study.best_value:.4f}")
    print(f"  最適パラメータ: {study.best_params}")
    print(f"  試行回数: {len(study.trials)}")
    
    # 上位5件の結果を表示
    print(f"\n上位5件の結果:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value is not None:
            print(f"  {i+1:2d}. RMSE: {trial.value:.4f}, パラメータ: {trial.params}")
    
    return study.best_params


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    指定したハイパーパラメータでモデルを学習し、テスト R² を出力して学習済みモデルを返す
    モデルには使った特徴量名を属性に保持
    """
    import lightgbm as lgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  訓練データサイズ: {X_train.shape}")
    print(f"  テストデータサイズ: {X_test.shape}")
    
    model = lgb.LGBMRegressor(**params, random_state=random_state)
    model.fit(X_train, y_train)

    # 訓練データとテストデータのスコア
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"  訓練データ R²: {train_score:.4f}")
    print(f"  テストデータ R²: {test_score:.4f}")

    # 使った特徴量名を保存
    model.feature_names_ = X.columns.tolist()
    return model