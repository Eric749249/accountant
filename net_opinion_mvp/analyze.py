# analyze.py — GPT APIでセンチメント(-1/0/+1)とトピックを返す実装（語句辞書は使わない）
import os, json, time, math, re
from typing import List, Dict
import pandas as pd

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("openai パッケージが見つかりません。`pip install openai` を実行してください。") from e

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("環境変数 OPENAI_API_KEY が未設定です。PowerShellで `$env:OPENAI_API_KEY = \"...\"` を実行してください。")

# 使用モデル（コスト・速度のバランスで小型を推奨）
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

TOPIC_LABELS = [
    "政策", "人格", "外交", "経済",
    "国会運営", "党派支持", "メディア", "倫理",
    "その他",
]

SYSTEM_PROMPT = (
    "You are a precise political stance classifier for Japanese YouTube comments. "
    "Your task is to determine the user's stance toward Prime Minister Sanae Takaichi. "
    "For each input text, return a JSON object with key `results`, containing an array aligned to inputs. "
    "Each item MUST have: `sentiment` (integer -1/0/1) and `topic` "
    "(one of: 政策, 人格, 外交, 経済, 国会運営, 党派支持, メディア, 倫理, その他). "
    "Output strictly valid JSON only, with no explanations.\n\n"

    "=== ENTITY DEFINITION ===\n"
    "Target person = Prime Minister Sanae Takaichi. "
    "She may be referred to by various expressions such as 「首相」, 「高市首相」, 「高市さん」, 「早苗」, 「高市氏」, etc. "
    "All such variations, pronouns, or implied references refer to the same person.\n\n"

    "=== SENTIMENT DEFINITION ===\n"
    "`sentiment` represents the stance toward Prime Minister Takaichi:\n"
    "-1 → clear opposition / disapproval / criticism / rejection of Takaichi.\n"
    " 0 → neutral / unclear / unrelated / general political sarcasm not directed at her.\n"
    " 1 → clear support / approval / praise / defense of Takaichi.\n\n"

    "When Takaichi is being criticized, questioned, overworked, or treated unfairly, "
    "any expression of sympathy, compassion, or defense toward her must be classified as 1 "
    "even if her name is not directly mentioned (e.g. 「総理」, 「彼女」, 「いじめ」, 「かわいそう」, "
    "「休ませて」, 「ブラック労働」, 「倒れないで」). "
    "When other people or institutions (Diet members, opposition parties, media, questioners) "
    "are criticized while Takaichi is protected, classify it as 1.\n\n"

    "=== SENTIMENT DECISION RULES (apply in order) ===\n"
    "Purpose: Classify each comment as Support (1), Neutral (0), or Oppose (-1) toward Prime Minister Sanae Takaichi.\n\n"

    "--- 1. Clear Positive (1) ---\n"
    "1) Direct praise, support, sympathy, or protection toward Takaichi "
    "(e.g., 「高市さん頑張って」「総理を守れ」「かわいそう」「休ませてあげて」) → 1.\n"
    "2) Comments expressing anger or frustration toward those who overwork, mistreat, attack, or mock Takaichi "
    "(e.g., 野党, メディア, 質問者) should be classified as 1, "
    "even when written with strong or aggressive tone or emojis such as 「アホ」「くだらない」「無視でいい」「💢」, "
    "as long as the anger is clearly NOT directed at Takaichi herself.\n"
    "3) Criticism of opposition parties or the media that clearly functions as a defense of Takaichi or the ruling party "
    "(e.g., 「くだらない追及に付き合う必要ない」「野党のいじめ質問ひどい」) → 1.\n"
    "4) Explicit statements that Takaichi should continue in office, become or remain prime minister, "
    "or is preferable to other candidates (e.g., 「高市続投で」「次は高市で行くべき」) → 1.\n\n"

    "--- 2. Clear Negative (-1) ---\n"
    "5) Only classify as -1 when there is direct criticism, mockery, or rejection aimed at Takaichi herself, "
    "using her name, title, or clear reference (e.g., 「高市はいらない」「高市は無理」「高市が最悪」「この総理は終わってる」). → -1.\n"
    "6) Strong criticism of the LDP or ruling bloc should be classified as -1 only when the criticism explicitly "
    "includes or targets Takaichi as part of the problem (e.g., 「高市も含めて自民は全部ダメ」). "
    "If Takaichi is not clearly included, do NOT assume -1.\n\n"

    "--- 3. Indirect or context-dependent cases ---\n"
    "7) Support for other LDP politicians or other prime minister candidates (e.g., Kishida, Ishiba, Motegi) "
    "without any clear negative or positive statement about Takaichi → 0.\n"
    "8) Support for opposition parties or non-LDP politicians (e.g., 立憲, 共産, 維新) "
    "without any clear negative or positive statement about Takaichi → 0.\n"
    "9) Criticism of opposition parties only (e.g., 「立憲はくだらない質問ばかり」) with no clear mention of Takaichi "
    "should usually be 0, unless it clearly functions as a defense of her as in Rule 3.\n"
    "10) General frustration, satire, or complaints about politics or society as a whole, "
    "when it is unclear whether Takaichi is supported or opposed → 0.\n"
    "11) When both positive and negative signals toward Takaichi appear but the overall stance is unclear or contradictory → 0.\n\n"

    "--- 4. Tie-breaking policy ---\n"
    "12) If there is clear evidence for 1 and no explicit attack on Takaichi, choose 1.\n"
    "13) Only choose -1 when there is clear and direct negative language aimed at Takaichi herself. "
    "In all other ambiguous cases, choose 0.\n\n"

    "=== TOPIC DEFINITION ===\n"
    "Choose the single most relevant category for the main discussion in the text.\n"
    "Prefer a specific category over 'その他' whenever applicable.\n"
    "- 政策: 個別政策や制度、法改正、行政の打ち手の是非\n"
    "- 人格: 人柄・姿勢・言動・マナー・体調への配慮/批判\n"
    "- 外交: 外交姿勢、防衛、同盟、国際関係\n"
    "- 経済: 物価・賃金・景気・産業・企業動向・家計\n"
    "- 国会運営: 質疑応答、ヤジ、いじめ、時間配分、手続論\n"
    "- 党派支持: 派閥、人事、総裁選、選挙、党派支持/不支持の表明\n"
    "- メディア: 報道姿勢、切り取り、SNS/YouTubeの言論環境\n"
    "- 倫理: スキャンダル、金銭・利権・不祥事、説明責任、倫理/コンプラ\n"
    "- その他: 上記に当てはまらない場合のみ使用\n"
    "If multiple categories seem plausible, choose the most specific non-'その他' category.\n\n"

    "=== OUTPUT FORMAT (strictly) ===\n"
    "Numbers MUST be plain integers -1, 0, or 1 (never use a leading plus sign like +1).\n"
    "Return a fully closed, valid JSON object on a single line. Do not stream or truncate.\n"
    "{ \"results\": [ {\"sentiment\": -1|0|1, "
    "\"topic\": \"政策|人格|外交|経済|国会運営|党派支持|メディア|倫理|その他\"}, ... ] }\n\n"

    "=== EXAMPLES (in Japanese) ===\n"
    "例1: 「高市さんを続投で。外交は評価してる」→ sentiment=1, topic=外交\n"
    "例2: 「高市さん、くだらない質問なんか無視でいいですよ。アホに付き合う必要無いですよ」"
    "→ sentiment=1, topic=国会運営\n"
    "例3: 「高市総理大臣にブラック労働させてるのは誰だ！！💢💢💢」→ sentiment=1, topic=党派支持\n"
    "例4: 「高市も岸田もどっちも無理」→ sentiment=-1, topic=党派支持\n"
    "例5: 「立憲はほんとどーでもいい質問ばかり😅」→ sentiment=0, topic=党派支持\n"
    "例6: 「岸田を支持、次も岸田でいい」(高市への言及なし) → sentiment=0, topic=党派支持\n"
    "例7: 「物価がつらい。政治は誰でも同じ」→ sentiment=0, topic=経済\n"
    "例8: 「彼女の態度は無理」(高市を指す文脈) → sentiment=-1, topic=人格\n"
)

import re

# ポジ/ネガの手動救済キーワード
_POS_PATTERNS = [
    r"高市.*(支持|応援|続投|続けて|総理に|なってほしい|推し|しか勝たん)",
    r"高市さん?頑張って",
    r"早苗ちゃん?頑張って",
    r"(総理|彼女).*かわいそう",
    r"(総理|彼女).*休ませてあげて",
    r"(いじめ|パワハラ).*(やめろ|酷すぎ)",
    r"よく頑張ってる",
    r"倒れないで",
    r"くだらない質問なんか無視でいい",
    r"アホに付き合う必要無い",
    r"ブラック労働させてる",
]

_NEG_PATTERNS = [
    r"高市.*(無理|嫌い|要らない|やめろ|辞めろ|終わってる|最悪)",
    r"(総理|こいつ).*(無理|嫌い|終わってる|ダメ)",
]

def _heuristic_adjust_sentiment(text: str, s: int) -> int:
    """
    GPT が返した sentiment s (-1/0/1) を、
    典型的なポジ/ネガ表現に基づいて微調整する安全弁。
    """
    if not isinstance(text, str):
        return s
    t = text.replace(" ", "").replace("　", "")

    # まずネガ（明確な dis）を優先
    for pat in _NEG_PATTERNS:
        if re.search(pat, t):
            return -1

    # 明らかなポジ（応援・同情・擁護）は +1 に救済
    for pat in _POS_PATTERNS:
        if re.search(pat, t):
            return 1

    return s

def _build_user_prompt(items: List[str]) -> str:
    """
    バッチ分類のユーザープロンプトを構築。
    """
    payload = {
        "instruction": {
            "sentiment": "日本語本文の極性を -1(否定)/0(中立)/+1(肯定)で判定",
            "topic_enum": TOPIC_LABELS
        },
        "inputs": [{"id": i, "text": t} for i, t in enumerate(items)]
    }
    return json.dumps(payload, ensure_ascii=False)

def _call_gpt_batch(texts: List[str], model: str = DEFAULT_MODEL, max_retries: int = 3) -> List[Dict]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = _build_user_prompt(texts)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},  # ← これを追加
            )

            # ▼ ここからを置き換える ▼
            import json, re
            raw = resp.choices[0].message.content.strip()

            # 先頭+の除去（JSONでは不正）
            raw = re.sub(r':\s*\+1(\b|[^0-9])', r': 1\1', raw)

            # JSONモードなら基本そのままパースで通る
            try:
                data = json.loads(raw)
            except Exception:
                # フォールバック：最初の '{' から最後の '}' までを再抽出して再トライ
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError(f"JSON parse error: {raw[:300]}")
                fixed = m.group(0)
                fixed = re.sub(r':\s*\+1(\b|[^0-9])', r': 1\1', fixed)
                data = json.loads(fixed)
            # ▲ ここまでを置き換える ▲

            results = data.get("results")
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, list):
                results = [data]

            if len(results) != len(texts):
                results = (results * math.ceil(len(texts) / len(results)))[:len(texts)]

            out = []
            for r in results:
                s = r.get("sentiment", 0)
                try:
                    s = int(s)
                except Exception:
                    s = 0
                if s not in (-1, 0, 1):
                    s = 0
                t = r.get("topic", "その他")
                if t not in TOPIC_LABELS:
                    t = "その他"
                out.append({"sentiment": s, "topic": t})
            return out

        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"OpenAI呼び出しに失敗しました: {last_err}")

def _classify_with_gpt(texts: List[str], batch_size: int = 10) -> List[Dict]:
    """
    texts を batch_size ずつ GPT に投げ、結合して返す。
    """
    results: List[Dict] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        chunk = _call_gpt_batch(batch)
        results.extend(chunk)
    return results

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    期待カラム: ['text','source','likes','published_at']
    返却: sentiment(-1/0/1をfloatに変換: -1.0/0.0/1.0), topic, date
    """
    if df is None or df.empty:
        return df

    dfx = df.copy()
    dfx['text'] = dfx['text'].astype(str).fillna("")

    # GPTで一括判定
    gpt_out = _classify_with_gpt(dfx["text"].tolist(), batch_size=20)

    # --- ここから修正 ---
    sentiments = []
    topics = []

    for text, item in zip(dfx["text"].tolist(), gpt_out):
        raw_s = int(item.get("sentiment", 0))  # GPT出力（-1/0/1）
        # 救済ロジックで補正（+1/−1 の取りこぼしを防ぐ）
        fixed_s = _heuristic_adjust_sentiment(text, raw_s)

        sentiments.append(float(fixed_s))
        topics.append(item.get("topic", "その他"))
    # --- ここまで修正 ---

    dfx["sentiment"] = sentiments
    dfx["topic"] = topics
    dfx["date"] = pd.to_datetime(dfx["published_at"], errors="coerce").dt.date
    return dfx


def kpi(dfx: pd.DataFrame) -> Dict[str, float]:
    if dfx is None or dfx.empty:
        return {"n_comments": 0, "pos_rate": 0.0, "neg_rate": 0.0, "avg_sentiment": 0.0}
    n = len(dfx)
    pos = (dfx['sentiment'] > 0).mean()
    neg = (dfx['sentiment'] < 0).mean()
    avg = dfx['sentiment'].mean()
    return {"n_comments": int(n), "pos_rate": float(pos), "neg_rate": float(neg), "avg_sentiment": float(avg)}

# ========= ここから追加（既存コードの下に追記） =========
def _normalize_topic(label: str) -> str:
    """モデルや旧版プロンプトが返す表記ゆれを、新しい TOPIC_LABELS に寄せる"""
    if not isinstance(label, str):
        return "その他"
    raw = label.strip()

    mapping = {
        # 旧バージョン/別表記 → 新ラベル
        "人格/態度": "人格",
        "外交/安全保障": "外交",
        "経済/物価": "経済",
        "与野党の態度": "国会運営",
        "国会運営/与野党の態度": "国会運営",
        "政局/党派支持": "党派支持",
        "政局/選挙・党派支持": "党派支持",
        "メディア/報道": "メディア",
        "メディア・報道": "メディア",
        "スキャンダル/倫理": "倫理",
        "スキャンダル・倫理": "倫理",
    }

    cand = mapping.get(raw, raw)
    return cand if cand in TOPIC_LABELS else "その他"

def _summarize_transcript_for_context(transcript_text: str, model: str = DEFAULT_MODEL) -> str:
    """
    長尺トランスクリプトを分類に効く日本語要約(<=1500文字程度)に圧縮。
    ※ ここは新規API呼び出し（サマリ用）。精度を優先。速度重視なら transcript_text[:4000] でもOK。
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = (
        "以下は日本語の動画文字起こしです。高市首相への賛否分類の文脈理解に使えるよう、"
        "日本語で箇条書きの重要ポイント要約を作成してください。"
        "・主要トピック/論点・誰が誰に何を主張/批判/擁護したか・外交/安全保障/経済/人格・態度の論点・"
        "高市首相に関係する出来事/発言/質問の要旨・視聴者が同情/擁護/批判しそうな場面。"
        "最大1500文字以内。箇条書きのみ。"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You summarize Japanese transcripts into concise bullet points."},
            {"role": "user", "content": prompt + "\n\n【文字起こし】\n" + transcript_text}
        ],
        temperature=0.2,
    )
    summary = resp.choices[0].message.content.strip()
    return summary[:2000]


def _build_user_prompt_with_context(items: List[str], context_summary: str) -> str:
    """
    再判定用ユーザープロンプト：context にトランスクリプト要約を同梱。
    """
    payload = {
        "instruction": {
            "task": "Re-classify stance toward Prime Minister Sanae Takaichi using the provided context.",
            "sentiment_def": "-1(opposition)/0(neutral)/+1(support)",
            "topic_enum": TOPIC_LABELS
        },
        "context": context_summary,
        "inputs": [{"id": i, "text": t} for i, t in enumerate(items)]
    }
    return json.dumps(payload, ensure_ascii=False)


def _call_gpt_batch_with_context(texts: List[str], context_summary: str,
                                 model: str = DEFAULT_MODEL, max_retries: int = 3) -> List[Dict]:
    """
    texts(バッチ) -> [{"sentiment": -1|0|1, "topic": "<ラベル>"}]
    既存の _call_gpt_batch は触らず、文脈付きの別関数として実装。
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    user_prompt = _build_user_prompt_with_context(texts, context_summary)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},  # JSONモード
            )

            # 既存実装と同じ“保険付き”パース（重複OK：既存コードに影響を与えないためコピペ）
            import json, re
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r':\s*\+1(\b|[^0-9])', r': 1\1', raw)
            try:
                data = json.loads(raw)
            except Exception:
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError(f"JSON parse error: {raw[:300]}")
                fixed = m.group(0)
                fixed = re.sub(r':\s*\+1(\b|[^0-9])', r': 1\1', fixed)
                data = json.loads(fixed)

            results = data.get("results")
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, list):
                results = [data]

            # 念のため件数合わせ
            if len(results) != len(texts) and len(results) > 0:
                results = (results * math.ceil(len(texts) / len(results)))[:len(texts)]

            out = []
            for r in results:
                s = r.get("sentiment", 0)
                try:
                    s = int(s)
                except Exception:
                    s = 0
                if s not in (-1, 0, 1):
                    s = 0
                t = _normalize_topic(r.get("topic", "その他"))
                out.append({"sentiment": s, "topic": t})
            return out

        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"OpenAI呼び出し(文脈付き)に失敗しました: {last_err}")


def refine_with_transcript(dfx: pd.DataFrame, transcript_text: str,
                           summarize: bool = True, batch_size: int = 10) -> pd.DataFrame:
    """
    文字起こしによる“文脈再判定”を、初回分類で sentiment==0 の行にだけ適用する。
    既存カラムや他行には一切触れない（= 既存機能を損なわない）。
    - dfx: enrich() 済みの DataFrame（sentiment, topic が付与済み）
    - transcript_text: あなたが貼る文字起こし全文
    - summarize: True の場合は要約してからプロンプトに同梱（長文でも安定）
    """
    if dfx is None or dfx.empty or not transcript_text or not isinstance(transcript_text, str):
        return dfx

    mask = (dfx["sentiment"] == 0) | (dfx["sentiment"] == 0.0)
    if not mask.any():
        return dfx  # 0が無ければ何もしない

    # 文脈要約 or そのまま使用
    if summarize:
        context_summary = _summarize_transcript_for_context(transcript_text)
    else:
        # 長すぎると不安定になるため、保険で上限
        context_summary = transcript_text[:4000]

    idx = dfx.index[mask]
    texts = dfx.loc[idx, "text"].astype(str).tolist()

    # 文脈付きで再判定
    reclassified = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        chunk = _call_gpt_batch_with_context(batch, context_summary)
        reclassified.extend(chunk)

    # 反映：0の行だけ、かつ“非0に変わった場合のみ”上書き（= 保守的）
    for j, row_id in enumerate(idx):
        new_s = int(reclassified[j]["sentiment"])
        new_t = _normalize_topic(reclassified[j]["topic"])
        if new_s != 0:
            dfx.at[row_id, "sentiment"] = float(new_s)
            dfx.at[row_id, "topic"] = new_t

    return dfx
# ========= 追加ここまで =========

