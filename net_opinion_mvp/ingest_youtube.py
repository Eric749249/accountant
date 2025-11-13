"""
YouTubeコメント取得（公式API）。
使い方:
  export YOUTUBE_API_KEY=...
  python ingest_youtube.py VIDEO_ID > comments_youtube.csv

注意: API割当・利用規約を順守すること。
"""
import os, sys, time
import pandas as pd
from googleapiclient.discovery import build

API_KEY = os.environ.get("YOUTUBE_API_KEY")

def fetch_comments(video_id: str, max_pages: int = 5) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
        order="relevance"
    )
    pages = 0
    while req and pages < max_pages:
        res = req.execute()
        for item in res.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "source": "YouTube",
                "text": sn.get("textDisplay",""),
                "likes": sn.get("likeCount",0),
                "published_at": sn.get("publishedAt","")
            })
        req = youtube.commentThreads().list_next(req, res)
        pages += 1
        time.sleep(0.2)
    return pd.DataFrame(comments)

if __name__ == "__main__":
    if API_KEY is None:
        print("環境変数YOUTUBE_API_KEYが未設定です。", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) < 2:
        print("使い方: python ingest_youtube.py VIDEO_ID [OUTPUT_CSV]", file=sys.stderr)
        sys.exit(1)
    video_id = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) >= 3 else "comments_youtube.csv"
    df = fetch_comments(video_id)
    # Excelでも文字化けしにくいUTF-8 BOM付き
    df.to_csv(output, index=False, encoding="utf-8-sig")
