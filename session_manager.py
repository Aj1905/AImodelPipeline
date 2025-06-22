from datetime import datetime


class TuningSession:
    """チューニングセッションを管理するクラス"""

    def __init__(self):
        self.session_history = []
        self.current_results = None
        self.db_path = None  # データベースパスを保存

    def set_db_path(self, db_path: str):
        """データベースパスを設定"""
        self.db_path = db_path
        print(f"✓ データベースパスを設定しました: {db_path}")

    def get_db_path(self) -> str | None:
        """データベースパスを取得"""
        return self.db_path

    def add_results(self, results: dict, target_column: str, feature_columns: list[str]) -> dict:
        """チューニング結果をセッションに追加"""
        session_id = len(self.session_history) + 1
        timestamp = datetime.now().isoformat()

        enhanced_results = {
            **results,
            "session_id": session_id,
            "timestamp": timestamp,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
        }

        self.session_history.append(enhanced_results)
        self.current_results = enhanced_results

        print(f"\n✓ セッション #{session_id} に結果を保存しました")
        print(f"  実行時刻: {timestamp}")
        print(f"  目的変数: {target_column}")
        print(f"  特徴量数: {len(feature_columns)}")

        return enhanced_results

    def get_latest_results(self) -> dict | None:
        return self.current_results

    def get_results_by_id(self, session_id: int) -> dict | None:
        for result in self.session_history:
            if result.get("session_id") == session_id:
                return result
        return None

    def get_session_history(self) -> dict:
        """セッション履歴を辞書形式で取得"""
        history_dict = {}
        for result in self.session_history:
            session_id = result.get("session_id")
            if session_id:
                history_dict[session_id] = result
        return history_dict

    def list_session_history(self) -> list:
        """セッション履歴を表示"""
        if not self.session_history:
            print("セッション履歴がありません")
            return []

        print("\n=== セッション履歴 ===")
        print(f"総セッション数: {len(self.session_history)}")

        for result in self.session_history:
            session_id = result.get("session_id", "不明")
            timestamp = result.get("timestamp", "不明")
            target = result.get("target_column", "不明")
            feature_count = result.get("feature_count", "不明")
            strategy = result.get("strategy", result.get("method", "不明"))
            best_score = result.get("best_score", "不明")

            # チューニング結果かどうかを判定
            has_tuning = "best_params" in result
            tuning_status = "✓ チューニング済み" if has_tuning else "⚠️ 評価のみ"

            print(f"\nセッション #{session_id} ({tuning_status}):")
            print(f"  実行時刻: {timestamp}")
            print(f"  目的変数: {target}")
            print(f"  特徴量数: {feature_count}")
            print(f"  手法: {strategy}")
            print(f"  最良スコア: {best_score}")

            # チューニング結果の場合はパラメータも表示
            if has_tuning:
                best_params = result.get("best_params", {})
                if best_params:
                    print("  最適パラメータ:")
                    for key, value in list(best_params.items())[:3]:  # 最初の3つだけ表示
                        print(f"    {key}: {value}")
                    if len(best_params) > 3:
                        print(f"    ... (他{len(best_params) - 3}個のパラメータ)")

        return self.session_history


tuning_session = TuningSession()
