import os
import pandas as pd
import plotly.express as px
import streamlit as st
from analyze import enrich, kpi, refine_with_transcript

st.set_page_config(page_title="ネット世論ダッシュボード(MVP)", layout="wide")

st.title("ネット世論ダッシュボード (MVP)")
st.caption("対象: 高市首相（初期MVP）。一般・識者の“見えにくい批判/支持”を簡易可視化。")

st.sidebar.header("データ入力")
uploaded = st.sidebar.file_uploader("コメントCSVをアップロード（columns: text, source, likes, published_at）", type=["csv"])

if uploaded is None:
    st.info("サンプルデータを表示中。自分のCSVをアップロードすると切り替わります。")
    df = pd.read_csv("sample_comments.csv")
else:
    df = pd.read_csv(uploaded)

if df.empty:
    st.warning("データが空です")
    st.stop()

# --- 文字起こし（任意） ---
st.sidebar.subheader("動画の文字起こし（任意）")
transcript_text = st.sidebar.text_area(
    "ここに文字起こしを貼るだけで、0（不明）のみ文脈で再判定します。",
    height=180,
    placeholder="YouTube等の文字起こしをコピペ",
)
summarize_ctx = st.sidebar.checkbox("長文は要約して使う（推奨）", value=True)
apply_ctx = st.sidebar.button("文脈で再判定（sentiment==0のみ）")

# 前処理・特徴量付与
dfx = enrich(df)

# 文脈で再判定（0のみ・任意）
if apply_ctx and transcript_text and transcript_text.strip():
    dfx_before = dfx.copy()
    dfx = refine_with_transcript(dfx, transcript_text.strip(), summarize=summarize_ctx)
    try:
        updated = (dfx_before['sentiment'] != dfx['sentiment']).sum()
        st.sidebar.success(f"文脈再判定を適用：{updated} 件更新")
    except Exception:
        pass

# KPI
metrics = kpi(dfx)
c1, c2, c3, c4 = st.columns(4)
c1.metric("コメント件数", f"{metrics['n_comments']:,}")
c2.metric("ポジ率", f"{metrics['pos_rate']*100:.1f}%")
c3.metric("ネガ率", f"{metrics['neg_rate']*100:.1f}%")
c4.metric("平均センチメント", f"{metrics['avg_sentiment']:.2f}")

st.divider()

# 時系列（件数）
ts = dfx.groupby('date').size().reset_index(name='count')
fig_ts = px.bar(ts, x="date", y="count", title="コメント件数の推移")
st.plotly_chart(fig_ts, use_container_width=True)

# 時系列（平均センチメント）
ts_s = dfx.groupby('date')['sentiment'].mean().reset_index()
fig_ts_s = px.line(ts_s, x="date", y="sentiment", title="平均センチメントの推移")
st.plotly_chart(fig_ts_s, use_container_width=True)

# トピック円グラフ
topic_counts = dfx['topic'].value_counts().reset_index()
topic_counts.columns = ['topic','count']
fig_topic = px.pie(topic_counts, names='topic', values='count', title="話題トピック構成比")
st.plotly_chart(fig_topic, use_container_width=True)

# ソース別ポジ/ネガ率
src_stats = dfx.groupby('source').agg(pos_rate=('sentiment', lambda s: (s>0).mean()),
                                      neg_rate=('sentiment', lambda s: (s<0).mean()),
                                      n=('sentiment','count')).reset_index()
st.subheader("ソース別ポジ/ネガ率")
st.dataframe(src_stats)

# 明細テーブル（フィルタ）
st.subheader("コメント一覧（フィルター可）")
col1, col2 = st.columns(2)
with col1:
    f_source = st.multiselect("ソースで絞り込み", sorted(dfx['source'].dropna().unique().tolist()))
with col2:
    f_topic = st.multiselect("トピックで絞り込み", sorted(dfx['topic'].dropna().unique().tolist()))

view = dfx.copy()
if f_source:
    view = view[view['source'].isin(f_source)]
if f_topic:
    view = view[view['topic'].isin(f_topic)]

st.dataframe(view[['published_at','source','topic','sentiment','likes','text']].sort_values('published_at', ascending=False), use_container_width=True)

st.caption("※ センチメントは簡易辞書ベースのスコア（-1〜1）。本番では高精度モデル/外部APIに置換してください。")
