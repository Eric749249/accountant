
# ネット世論ダッシュボード (MVP)

高市首相を例に、ネット上のコメントを可視化する最小実装。  
YouTubeの公式APIでコメント収集 → CSV → Streamlitで可視化。

## 1) セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) データ取得（YouTube）

YouTube Data APIキーを環境変数に設定して、コメントを取得します。

```bash
export YOUTUBE_API_KEY=YOUR_API_KEY  # Windows: set YOUTUBE_API_KEY=YOUR_API_KEY
python ingest_youtube.py <VIDEO_ID> > comments_youtube.csv
```

※ Yahoo!ニュース等のスクレイピングは規約上の制約があるため、本MVPでは扱いません。公開API・許諾範囲で取得してください。

## 3) ダッシュボード起動

```bash
streamlit run app.py
```

- `sample_comments.csv` を同梱しています。まずはこれで挙動確認できます。
- 独自に取得した `comments_*.csv` をサイドバーからアップロードして分析可能です。

## 4) 切り替え可能な分析器

現状は **簡易ルールベース**（小辞書）でセンチメントとトピック分類を実装。  
本番では下記いずれかに差し替えを推奨：

- OpenAI API (gpt-4o-mini 等) を用いたバッチ判定
- 日本語感情辞書（用言・名詞スコア）を用いたスコアリング
- 学習済みBERT系の日本語感情分類モデル

## 5) 収益化に向けた次ステップ

- 対象プラットフォームを拡張（X, note, 掲示板など）
- 高精度モデルでの判定・イベント相関分析
- 定期レポート自動生成（週報・月報）
- 有料購読・法人向けカスタムダッシュボード
