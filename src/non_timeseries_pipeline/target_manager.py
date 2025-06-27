from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class TargetManager:
    """Manage target variable."""

    target: pd.Series | None = None

    def transform(self, series: pd.Series) -> pd.Series:
        """Transform target variable to be compatible with LightGBM."""
        self.target = series.copy()
        
        # データ型の情報を表示
        print(f"🎯 ターゲット変数のデータ型: {self.target.dtype}")
        print(f"🎯 ターゲット変数の統計情報:")
        print(f"   - 平均: {self.target.mean():.4f}")
        print(f"   - 標準偏差: {self.target.std():.4f}")
        print(f"   - 最小値: {self.target.min():.4f}")
        print(f"   - 最大値: {self.target.max():.4f}")
        print(f"   - 欠損値: {self.target.isna().sum()}")
        
        # データ型の変換
        if self.target.dtype == 'object':
            try:
                # 文字列を数値に変換
                self.target = pd.to_numeric(self.target, errors='coerce')
                print("✅ ターゲット変数を数値型に変換しました")
            except Exception as e:
                print(f"❌ ターゲット変数の変換に失敗: {e}")
                raise
        
        # 無限大の値をNaNに変換
        self.target = self.target.replace([np.inf, -np.inf], np.nan)
        
        # 欠損値を明らかに欠損値だとわかる値で埋める（行数の一致を保つため）
        if self.target.isna().sum() > 0:
            print(f"⚠️ ターゲット変数の欠損値を明らかな欠損値として埋めます（{self.target.isna().sum()}個）")
            # 非常に大きな負の値を使用して欠損値を明示的に表現
            missing_value = -999999
            self.target = self.target.fillna(missing_value)
            print(f"   - 欠損値を {missing_value} で埋めました")
        
        # 最終的なデータ型を確認
        print(f"✅ 前処理後のターゲット変数のデータ型: {self.target.dtype}")
        print(f"✅ 処理後の欠損値: {self.target.isna().sum()}")
        
        return self.target
