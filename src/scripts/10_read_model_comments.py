#!/usr/bin/env python3
"""
保存されたモデルのコメントと学習情報を読み取るスクリプト

このスクリプトは、pklファイルに保存されたモデルの学習情報とカスタムコメントを
表示します。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.implementations.lightgbm_model import LightGBMRegressor


def load_and_display_model_info(model_path: Path):
    """モデルを読み込んで情報を表示"""
    print(f"📥 モデルを読み込み中: {model_path}")

    if not model_path.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return False

    try:
        # モデルを読み込み
        model = LightGBMRegressor()
        model.load_model(model_path)

        print("✅ モデル読み込み完了!")

        # 学習情報を表示
        if hasattr(model, "print_training_info"):
            model.print_training_info()
        else:
            print("⚠️  このモデルには学習情報が含まれていません")

        # コメントのみを表示
        if hasattr(model, "get_comments"):
            comments = model.get_comments()
            if comments:
                print(f"\n📝 カスタムコメント ({len(comments)}個):")
                for i, comment in enumerate(comments, 1):
                    print(f"  {i}. {comment}")
            else:
                print("\n📝 カスタムコメント: なし")

        # 特徴量重要度も表示
        try:
            importance = model.get_feature_importance()
            print("\n🎯 特徴量重要度 (トップ10):")
            sorted_importance = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, imp in sorted_importance[:10]:
                print(f"  {feature}: {imp:.4f}")
        except Exception as e:
            print(f"⚠️  特徴量重要度の取得に失敗しました: {e}")

        return True

    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="保存されたモデルのコメントと学習情報を読み取る"
    )
    parser.add_argument("model_path", type=str, help="読み取るモデルファイルのパス")

    args = parser.parse_args()
    model_path = Path(args.model_path)

    print("🔍 モデル情報読み取りツール")
    print("=" * 50)

    success = load_and_display_model_info(model_path)

    if success:
        print("\n✨ 読み取り完了!")
    else:
        print("\n❌ 読み取りに失敗しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
