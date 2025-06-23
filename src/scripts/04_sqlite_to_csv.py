#!/usr/bin/env python3
"""
SQLiteデータベースの指定したテーブル(または全テーブル)をCSVファイルとしてエクスポートするスクリプト
"""

import argparse
import csv
import sqlite3
from pathlib import Path


def get_table_names(db_path: Path) -> list[str]:
    """データベースからテーブル名のリストを取得する"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"データベースエラー: {e}")
        return []


def select_tables_interactively(all_tables: list[str]) -> list[str]:
    """ユーザーにテーブルを対話形式で選択させる"""
    print("\n" + "=" * 50)
    print("📋 エクスポートするテーブルを選択してください")
    print("=" * 50)

    # テーブル一覧を表示
    for i, table in enumerate(all_tables, 1):
        print(f"{i:2d}. {table}")

    print("\n選択方法:")
    print("  - 単一テーブル: 番号を入力 (例: 1)")
    print("  - 複数テーブル: カンマ区切りで番号を入力 (例: 1,3,5)")
    print("  - 全テーブル: 'all' または 'a' と入力")
    print("  - 終了: 'q' または 'exit' と入力")
    print("-" * 50)

    while True:
        choice = input("選択してください >> ").strip()

        if choice.lower() in ["q", "exit"]:
            print("処理を中断します")
            return []

        if choice.lower() in ["all", "a"]:
            print("全テーブルを選択しました")
            return all_tables

        try:
            # 選択された番号を処理
            selected_indices = []
            for part in choice.split(","):
                part = part.strip()
                if part.isdigit():
                    index = int(part) - 1
                    if 0 <= index < len(all_tables):
                        selected_indices.append(index)

            if not selected_indices:
                print("有効な番号を入力してください")
                continue

            selected_tables = [all_tables[i] for i in selected_indices]
            print(f"選択されたテーブル: {', '.join(selected_tables)}")
            return selected_tables

        except ValueError:
            print("無効な入力です。数字か'all'を入力してください")


def export_tables_to_csv(db_path: Path, output_dir: Path, tables_to_export: list[str]):
    """
    SQLiteデータベースのテーブルをCSVにエクスポート

    Args:
        db_path: SQLiteデータベースファイルのパス
        output_dir: CSV出力ディレクトリ
        tables_to_export: エクスポートするテーブル名のリスト
    """
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tables_to_export:
        print("エクスポートするテーブルが選択されていません")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            total_exported = 0

            print(f"\nエクスポートを開始します: {len(tables_to_export)} テーブル")

            for table in tables_to_export:
                # 出力ファイルパス
                csv_path = output_dir / f"{table}.csv"

                # テーブルデータ取得
                cursor.execute(f"SELECT * FROM {table};")
                rows = cursor.fetchall()

                # カラム名取得
                column_names = [description[0] for description in cursor.description]

                # CSV書き込み
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(column_names)  # ヘッダー
                    writer.writerows(rows)  # データ

                print(f"  ✓ {table}: {len(rows)}行 -> {csv_path}")
                total_exported += len(rows)

            print(f"\n✅ {len(tables_to_export)}テーブル、合計{total_exported}行のエクスポートが完了しました")
            print(f"出力先: {output_dir}")

    except sqlite3.Error as e:
        print(f"データベースエラー: {e}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SQLiteデータベースのテーブルをCSVにエクスポート")
    parser.add_argument(
        "--db-file",
        type=str,
        default="data/database.sqlite",
        help="SQLiteデータベースファイルパス (デフォルト: data/database.sqlite)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/aj/Documents/forecasting_poc/data",
        help="CSV出力ディレクトリ (デフォルト: sqlite_export)",
    )
    parser.add_argument(
        "--tables", type=str, default="", help="エクスポートするテーブル名(カンマ区切り、未指定の場合は選択モード)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="非対話モードで実行(--tables未指定時は全テーブルをエクスポート)",
    )

    args = parser.parse_args()

    # パス解決
    db_path = Path(args.db_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # データベース存在チェック
    if not db_path.exists():
        print(f"❌ エラー: データベースファイルが見つかりません: {db_path}")
        return

    print(f"🗃️  データベース: {db_path}")
    print(f"📂 出力先ディレクトリ: {output_dir}")

    # 全テーブル名を取得
    all_tables = get_table_names(db_path)
    if not all_tables:
        print("エクスポート可能なテーブルが見つかりませんでした")
        return

    print(f"📋 利用可能なテーブル数: {len(all_tables)}")

    # エクスポート対象テーブルの決定
    if args.tables:
        # コマンドラインからテーブル指定
        tables_to_export = [t.strip() for t in args.tables.split(",")]

        # 存在しないテーブルをフィルタリング
        valid_tables = [t for t in tables_to_export if t in all_tables]
        invalid_tables = set(tables_to_export) - set(valid_tables)

        if invalid_tables:
            print(f"⚠️  警告: 以下のテーブルは存在しません: {', '.join(invalid_tables)}")

        if not valid_tables:
            print("❌ エクスポート対象の有効なテーブルがありません")
            return

        tables_to_export = valid_tables
        print(f"コマンドライン指定テーブル: {', '.join(tables_to_export)}")

    elif args.non_interactive:
        # 非対話モードで全テーブルをエクスポート
        tables_to_export = all_tables
        print(f"非対話モード: 全{len(all_tables)}テーブルをエクスポートします")

    else:
        # 対話型選択モード
        tables_to_export = select_tables_interactively(all_tables)
        if not tables_to_export:
            return  # ユーザーがキャンセルした場合

    # エクスポート実行
    export_tables_to_csv(db_path, output_dir, tables_to_export)


if __name__ == "__main__":
    main()
