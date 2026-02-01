import streamlit as st

# ============================
# Step0: アプリ初期化（Streamlit設定）
# ============================
st.set_page_config(page_title="RAG Chatbot (TF-IDF + Gemini)", layout="centered")

# ============================
# Step0: import
# ============================
import os
import re
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ============================
# Step0: 検索パラメータ（固定でOK）
# ============================
ALPHA = 0.6          # 統合スコアの重み：word寄り=1.0 / char寄り=0.0
TOP_K = 3            # 参照する記事数
MIN_SCORE = 0.05     # これ未満は「ヒットなし」扱い（広すぎる質問の誤ヒット抑制）

# ============================
# Step0: .env 読み込み → APIキー確認
# ============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("環境変数 GOOGLE_API_KEY が設定されていません。.env を確認してください。")
    st.stop()

# ============================
# Step1: Gemini API（LLM）準備
# ============================
@st.cache_resource
def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")

def gemini_ask(model, prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return ""
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "") or ""

# ============================
# Step2: CSV読込 → docs 作成
# ============================
@st.cache_data
def load_news_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["topic", "url", "title", "text", "text_tokenized"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}. columns={list(df.columns)}")

    df["topic"] = df["topic"].fillna("")
    df["url"] = df["url"].fillna("")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["text_tokenized"] = df["text_tokenized"].fillna("")
    return df

def build_docs(df: pd.DataFrame) -> list[dict]:
    docs: list[dict] = []
    for i, row in df.reset_index(drop=True).iterrows():
        title = str(row["title"]).strip()
        text = str(row["text"]).strip()

        docs.append(
            {
                "doc_id": int(i),
                "topic": str(row["topic"]),
                "url": str(row["url"]),
                "title": title,
                "text": text,
                # word検索用（CSV側が分かち書き済み前提）
                "tfidf_word": str(row["text_tokenized"]),
                # char検索用（自然文に強い：タイトルも混ぜる）
                "tfidf_char": f"{title} {text}",
                # Geminiに渡す根拠
                "display_text": f"【タイトル】{title}\n\n【本文】\n{text}",
            }
        )
    return docs

# ============================
# Step3: TF-IDF インデックス作成（word版 / char版）
# ============================
@st.cache_data
def build_tfidf_indexes(docs: list[dict]):
    # ---- word（分かち書き前提）
    word_corpus = [d["tfidf_word"] for d in docs]
    word_vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
    )
    word_matrix = word_vectorizer.fit_transform(word_corpus)

    # ---- char（自然文・表記ゆれに強い）
    char_corpus = [d["tfidf_char"] for d in docs]
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
    )
    char_matrix = char_vectorizer.fit_transform(char_corpus)

    return word_vectorizer, word_matrix, char_vectorizer, char_matrix

# ============================
# Step3.5: 質問文をTF-IDF向けに簡易正規化（汎用版）
# - vocab（word側の語彙）に寄せる
# ============================
_CHUNK_RE = re.compile(r"[一-龥]+|[ぁ-ん]+|[ァ-ン]+|[A-Za-z0-9]+")

def normalize_query_for_tfidf(query: str, vocab: set[str]) -> str:
    """
    自然文クエリを、CSV側(text_tokenized)の語彙に寄せた「スペース区切りトークン」にする（汎用版）
    - 形態素解析なし
    - 語彙(vocab)に存在する語だけ残す
    - vocabに無い複合語は、vocabに載るように分割できるなら分割語を追加
    """
    q = (query or "").strip()
    if not q:
        return ""

    # 記号をスペースへ
    q = re.sub(r"[^\wぁ-んァ-ン一-龥]+", " ", q)

    # ざっくり分割（漢字の塊 / ひらがな / カタカナ / 英数字）
    chunks = _CHUNK_RE.findall(q)

    # 最低限のストップ（助詞・定型）
    stop = {
        "について","教えて","ください","とは","です","ます","する","したい",
        "の","が","を","に","へ","と","も","や","から","まで","より",
        "ですか","ますか","ある","あります","いる","います",
        "最近","何","どんな","どう"
    }

    # まずはそのまま候補化
    candidates = []
    for c in chunks:
        if c in stop:
            continue
        candidates.append(c)

    # vocabに無い語は「語彙に載るように分割できるか」を試す（汎用）
    expanded = []
    for t in candidates:
        expanded.append(t)

        # vocabに無く、ある程度長い場合だけ分割探索
        if (t not in vocab) and (len(t) >= 4):
            for i in range(2, len(t) - 1):
                a, b = t[:i], t[i:]
                if (a in vocab) and (b in vocab):
                    expanded.extend([a, b])

    # vocabに存在する語だけ残す（ここが重要）
    expanded = [t for t in expanded if t in vocab]

    # 重複除去（順序維持）
    seen = set()
    uniq = []
    for t in expanded:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    return " ".join(uniq)

# ============================
# Step4: 検索（統合スコアで上位抽出）
# ============================
def search_docs(
    user_query: str,
    docs: list[dict],
    word_vectorizer,
    word_matrix,
    char_vectorizer,
    char_matrix,
    vocab: set[str],
    top_k: int = 3,
    alpha: float = 0.6,
    min_score: float = 0.05,
):
    user_query = (user_query or "").strip()
    if not user_query:
        return [], {"q_word": "", "max_score": 0.0}

    # word用（語彙寄せの簡易正規化）
    q_word = normalize_query_for_tfidf(user_query, vocab)

    # ベクトル化
    qv_char = char_vectorizer.transform([user_query])
    char_scores = cosine_similarity(qv_char, char_matrix).flatten()

    if q_word:
        qv_word = word_vectorizer.transform([q_word])
        word_scores = cosine_similarity(qv_word, word_matrix).flatten()
        combined = alpha * word_scores + (1 - alpha) * char_scores
    else:
        # wordが作れない＝語彙に寄らない自然文なので、charを100%採用（薄めない）
        combined = char_scores

    # 上位
    top_idx = np.argsort(combined)[::-1][:top_k]

    # 閾値未満は捨てる（誤ヒット抑制）
    top_idx = [i for i in top_idx if combined[i] >= min_score]

    results = []
    for i in top_idx:
        results.append(
            {
                "doc_id": docs[i]["doc_id"],
                "title": docs[i]["title"],
                "url": docs[i]["url"],
                "topic": docs[i]["topic"],
                "score": float(combined[i]),
                "display_text": docs[i]["display_text"],
            }
        )

    debug = {"q_word": q_word, "max_score": float(combined.max()) if combined.size else 0.0}
    return results, debug

# ============================
# Step5: RAG（根拠付きプロンプト）
# ============================
def build_rag_prompt(question: str, results: list[dict], query_keywords: str) -> str:
    evidence_blocks = []
    for i, r in enumerate(results, start=1):
        evidence = (r.get("display_text") or "")[:2000]
        evidence_blocks.append(f"[根拠{i}] topic={r.get('topic')} url={r.get('url')}\n{evidence}")

    evidence_text = "\n\n---\n\n".join(evidence_blocks)

    prompt = f"""
あなたはニュース記事の要約・解説アシスタントです。
次の【根拠】に書かれている内容だけを使って、ユーザーの質問に答えてください。
根拠に書かれていないことは推測せず、「根拠不足」と明確に伝えてください。

# 質問
{question}

# 質問の重要キーワード（参考）
{query_keywords}

# 根拠
{evidence_text}

# 指示
- まず結論を1〜2行
- 次に根拠に基づく説明（箇条書きでOK）
- 最後に「参照した根拠番号（例：[根拠1][根拠3]）」を付ける
""".strip()

    return prompt

# ============================
# Step2〜Step5: 起動時に一度だけ準備（キャッシュも効く）
# ============================
CSV_PATH = "dataset/yahoo_news_articles_preprocessed.csv"

df_news = load_news_csv(CSV_PATH)
docs = build_docs(df_news)

word_vec, word_mat, char_vec, char_mat = build_tfidf_indexes(docs)
gemini_model = get_gemini_model(GOOGLE_API_KEY)

# ★重要：word側の語彙を作る（あなたの全文のバグ修正点）
VOCAB = set(word_vec.get_feature_names_out())

# ============================
# Step6: UI（チャット）
# ============================
st.title("🗨️ ニュース検索チャット（TF-IDF + Gemini RAG）")
st.caption(f"検索設定（固定）：alpha={ALPHA} / TOP_K={TOP_K} / MIN_SCORE={MIN_SCORE}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("ニュースについて質問してください（例：日銀の政策金利について教えて）")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 1) Retrieval
    results, debug = search_docs(
        user_query=user_input,
        docs=docs,
        word_vectorizer=word_vec,
        word_matrix=word_mat,
        char_vectorizer=char_vec,
        char_matrix=char_mat,
        vocab=VOCAB,
        top_k=TOP_K,
        alpha=ALPHA,
        min_score=MIN_SCORE,
    )

    # 学習用：検索に使われたキーワード（word側）だけ表示（不要なら削除OK）
    if debug["q_word"]:
        st.caption(f"🔎 抽出キーワード（word側）: {debug['q_word']}")

    # 2) RAG回答
    if not results:
        gemini_block = (
            "**【Geminiの返答（RAG）】**\n\n"
            "関連する記事が見つからなかったため、根拠に基づく回答ができませんでした。\n"
            "もう少し具体的なキーワードで質問してください（例：日銀 政策金利 据え置き）。\n"
        )
        tfidf_block = "（検索結果：0件）"
        bot_reply = gemini_block + "\n---\n\n" + tfidf_block
    else:
        rag_prompt = build_rag_prompt(
            question=user_input,
            results=results,
            query_keywords=debug["q_word"] or "(抽出なし：char検索のみでヒット)",
        )

        try:
            rag_answer = gemini_ask(gemini_model, rag_prompt) or "(Geminiの返答が空でした)"
            gemini_block = f"**【Geminiの返答（RAG：根拠に基づく回答）】**\n\n{rag_answer}\n"
        except Exception as e:
            gemini_block = f"**【Geminiの返答（RAG）】**\n\nGemini呼び出しでエラー: {e}\n"

        # 3) 参照記事
        lines = [f"**【参照した記事（上位{len(results)}件）】**\n"]
        for rank, r in enumerate(results, start=1):
            lines.append(
                f"{rank}. **{r['title']}**  \n"
                f"　- topic: `{r['topic']}`  \n"
                f"　- score: `{r['score']:.4f}`  \n"
                f"　- url: {r['url']}\n"
            )
        tfidf_block = "\n".join(lines)

        bot_reply = gemini_block + "\n---\n\n" + tfidf_block

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()