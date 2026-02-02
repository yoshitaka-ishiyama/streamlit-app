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
    """
    ニュース記事CSVを読み込み、分析用に前処理したDataFrameを返す。

    役割：
    1. CSVの読み込み
    2. 必須カラムの存在チェック（スキーマ保証）
    3. 欠損値(NaN)を空文字に統一（後続のTF-IDF / 検索処理でのエラー防止）
    4. Streamlitキャッシュにより再読み込み高速化

    Parameters
    ----------
    csv_path : str
        読み込むCSVファイルのパス

    Returns
    -------
    pd.DataFrame
        前処理済みニュースデータ
    """

    # CSV読み込み
    df = pd.read_csv(csv_path)

    # ===== スキーマチェック =====
    # 必須カラムが不足している場合は即エラー（静かに壊れないようにする）
    required_cols = ["topic", "url", "title", "text", "text_tokenized"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}. columns={list(df.columns)}")

    # ===== 欠損値処理 =====
    # TF-IDF / 文字列処理でNaNがあると落ちるため、空文字に統一
    df["topic"] = df["topic"].fillna("")
    df["url"] = df["url"].fillna("")
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["text_tokenized"] = df["text_tokenized"].fillna("")

    return df

def build_docs(df: pd.DataFrame) -> list[dict]:
    """
    ニュースDataFrameを、検索・RAG用のドキュメント構造（list[dict]）に変換する。

    役割：
    1. 各記事を「1ドキュメント」として整理
    2. TF-IDF検索用テキスト（word / char）を準備
    3. Geminiに渡す表示用テキストを生成
    4. 後続処理で扱いやすい辞書形式に統一

    Parameters
    ----------
    df : pd.DataFrame
        前処理済みニュースデータ

    Returns
    -------
    list[dict]
        検索エンジン用ドキュメント配列
    """

    docs: list[dict] = []

    # reset_indexで 0,1,2... の連番IDを振り直す
    for i, row in df.reset_index(drop=True).iterrows():

        # 前後空白除去（検索精度安定化）
        title = str(row["title"]).strip()
        text = str(row["text"]).strip()

        docs.append(
            {
                # 一意なID（検索結果→元記事参照用）
                "doc_id": int(i),

                # メタ情報
                "topic": str(row["topic"]),
                "url": str(row["url"]),
                "title": title,
                "text": text,

                # ===== TF-IDF用 =====

                # 単語ベース検索用（CSV側で分かち書き済み）
                "tfidf_word": str(row["text_tokenized"]),

                # 文字n-gram検索用（自然文・質問文に強い）
                # タイトルも混ぜることでヒット率アップ
                "tfidf_char": f"{title} {text}",

                # ===== Gemini(RAG)用 =====
                # LLMにそのまま渡せる根拠テキスト
                "display_text": f"【タイトル】{title}\n\n【本文】\n{text}",
            }
        )

    return docs

# ============================
# Step3: TF-IDF インデックス作成（word版 / char版）
# ============================
@st.cache_data
def build_tfidf_indexes(docs: list[dict]):
    """
    検索用TF-IDFインデックスを作成する関数。

    役割：
    1. 記事テキストをベクトル化（数値化）
    2. 類似度計算（cosine similarity）可能な形に変換
    3. word検索 + char検索 の2系統を用意して精度向上

    戻り値：
        word_vectorizer : 単語ベース検索用のTF-IDFモデル
        word_matrix     : 単語ベクトル行列（記事×単語）
        char_vectorizer : 文字n-gram検索用TF-IDFモデル
        char_matrix     : 文字ベクトル行列（記事×文字n-gram）
    """

    # =========================
    # word（分かち書き検索）
    # =========================

    # 例: "日銀 政策 金利 据え置き"
    word_corpus = [d["tfidf_word"] for d in docs]

    # 単語スペース区切りをそのまま使う設定
    word_vectorizer = TfidfVectorizer(
        tokenizer=str.split,   # 空白区切り
        preprocessor=None,     # 追加前処理なし
        token_pattern=None,    # sklearn標準分割を無効化
    )

    # 学習 + ベクトル化（ここで「検索インデックス」生成）
    word_matrix = word_vectorizer.fit_transform(word_corpus)

    # =========================
    # char（文字n-gram検索）
    # =========================

    # タイトル + 本文の自然文
    char_corpus = [d["tfidf_char"] for d in docs]

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # 文字n-gram解析（単語境界考慮）
        ngram_range=(2, 4),   # 2〜4文字単位
        min_df=2,            # 2記事以上に出現する語だけ使用（ノイズ除去）
    )

    char_matrix = char_vectorizer.fit_transform(char_corpus)

    return word_vectorizer, word_matrix, char_vectorizer, char_matrix

# ============================
# Step3.5: 質問文をTF-IDF向けに簡易正規化（汎用版）
# - vocab（word側の語彙）に寄せる
# ============================
# ============================
# クエリ正規化（自然文 → TF-IDF検索用キーワード）
# ============================

# 日本語をざっくり「文字種ごとの塊」に分割する正規表現
# 例：
# 「日銀の政策金利について教えて」
# → ["日銀", "の", "政策金利", "について", "教えて"]
_CHUNK_RE = re.compile(r"[一-龥]+|[ぁ-ん]+|[ァ-ン]+|[A-Za-z0-9]+")


def normalize_query_for_tfidf(query: str, vocab: set[str]) -> str:
    """
    自然文の質問を「TF-IDF検索用キーワード列」に変換する関数。

    目的：
    - ユーザーの自然文質問を、CSV(text_tokenized)と同じ語彙形式に揃える
    - 形態素解析なしで軽量に実装
    - 検索に不要な助詞・定型語を除去
    - TF-IDF語彙(vocab)に存在する語のみ残す（誤ヒット防止）

    例：
        入力 : "日銀の政策金利について教えてください"
        出力 : "日銀 政策 金利"

    戻り値：
        スペース区切りキーワード文字列
    """

    # None/空対策
    q = (query or "").strip()
    if not q:
        return ""

    # =========================
    # ① 記号除去
    # =========================
    # 記号をスペースに置換（検索ノイズ防止）
    q = re.sub(r"[^\wぁ-んァ-ン一-龥]+", " ", q)


    # =========================
    # ② ざっくり分割（簡易トークン化）
    # =========================
    # 漢字/ひらがな/カタカナ/英数字の塊ごとに分割
    chunks = _CHUNK_RE.findall(q)


    # =========================
    # ③ ストップワード除去
    # =========================
    # 意味を持たない語（助詞・定型語）を削除
    stop = {
        "について","教えて","ください","とは","です","ます","する","したい",
        "の","が","を","に","へ","と","も","や","から","まで","より",
        "ですか","ますか","ある","あります","いる","います",
        "最近","何","どんな","どう"
    }

    candidates = []
    for c in chunks:
        if c in stop:
            continue
        candidates.append(c)


    # =========================
    # ④ 複合語の分割補助
    # =========================
    # 例: 政策金利 → 政策 + 金利
    expanded = []

    for t in candidates:
        expanded.append(t)

        # vocabに無い長い語だけ分割探索
        if (t not in vocab) and (len(t) >= 4):
            for i in range(2, len(t) - 1):
                a, b = t[:i], t[i:]
                if (a in vocab) and (b in vocab):
                    expanded.extend([a, b])


    # =========================
    # ⑤ vocabにある語だけ残す（最重要）
    # =========================
    # 検索対象語彙以外は削除 → 精度安定
    expanded = [t for t in expanded if t in vocab]


    # =========================
    # ⑥ 重複除去
    # =========================
    seen = set()
    uniq = []
    for t in expanded:
        if t not in seen:
            uniq.append(t)
            seen.add(t)


    # 最終的に「スペース区切り文字列」に変換
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
    """
    ユーザー質問(user_query)に対して、TF-IDF類似検索で上位記事を返す。

    ねらい：
    - word版TF-IDF：CSV側の分かち書き(text_tokenized)を活用（キーワードに強い）
    - char版TF-IDF：自然文や表記ゆれに強い（質問文をそのまま投げられる）
    - 統合スコア：combined = alpha * word + (1-alpha) * char
      ただし、word側のクエリが作れない（語彙に乗らない）場合は、charのみを採用して薄めない

    戻り値：
    - results: 上位記事のリスト（doc_id/title/url/topic/score/display_text）
    - debug:   検索に使ったq_word（word側のキーワード列）やmax_scoreなど
    """
    # 0) 入力チェック：空なら検索しない
    user_query = (user_query or "").strip()
    if not user_query:
        return [], {"q_word": "", "max_score": 0.0}

    # 1) word用のクエリを作る（語彙寄せ）
    #    - 自然文 → vocabに存在する語だけを残した「スペース区切りトークン列」
    #    - これにより、word TF-IDFが意味のある入力を受け取れるようにする
    q_word = normalize_query_for_tfidf(user_query, vocab)

    # 2) char検索（自然文のまま）を常に実施：表記ゆれ/自然文の保険
    qv_char = char_vectorizer.transform([user_query])
    char_scores = cosine_similarity(qv_char, char_matrix).flatten()

    # 3) wordが作れた場合のみ、word検索も実施し統合スコアへ
    if q_word:
        qv_word = word_vectorizer.transform([q_word])
        word_scores = cosine_similarity(qv_word, word_matrix).flatten()

        # 統合スコア（alphaで重み付け）
        combined = alpha * word_scores + (1 - alpha) * char_scores
    else:
        # wordが作れない＝語彙に寄らない自然文
        # → alphaで薄めると精度が落ちるので、charを100%採用
        combined = char_scores

    # 4) 上位K件のインデックスを取得（降順）
    top_idx = np.argsort(combined)[::-1][:top_k]

    # 5) 閾値未満は捨てる（広すぎる質問の誤ヒットを抑制）
    top_idx = [i for i in top_idx if combined[i] >= min_score]

    # 6) 結果整形：docsから必要情報だけを抜き出して返す
    results = []
    for i in top_idx:
        results.append(
            {
                "doc_id": docs[i]["doc_id"],
                "title": docs[i]["title"],
                "url": docs[i]["url"],
                "topic": docs[i]["topic"],
                "score": float(combined[i]),
                "display_text": docs[i]["display_text"],  # Geminiに渡す根拠
            }
        )

    # 7) デバッグ情報（学習・検証用）
    debug = {
        "q_word": q_word,
        "max_score": float(combined.max()) if combined.size else 0.0,
    }

    return results, debug


# ============================
# Step5: RAG（根拠付きプロンプト）
# ============================
def build_rag_prompt(question: str, results: list[dict], query_keywords: str) -> str:
    """
    検索でヒットした記事(results)を「根拠」として、Geminiに渡すプロンプトを作る。

    ねらい：
    - 根拠（記事本文）以外は推測しないように厳密に指示
    - 根拠番号 [根拠1], [根拠2]... を付け、回答の最後に参照番号を出させる
    - display_textは長くなりがちなので、各根拠を最大2000文字に制限（コスト/暴走防止）
    """
    evidence_blocks = []

    # 1) 根拠ブロックを作る（上位K件分）
    for i, r in enumerate(results, start=1):
        evidence = (r.get("display_text") or "")[:2000]  # 長すぎるとLLMが扱いづらいので制限
        evidence_blocks.append(
            f"[根拠{i}] topic={r.get('topic')} url={r.get('url')}\n{evidence}"
        )

    # 2) 根拠を区切り線で結合
    evidence_text = "\n\n---\n\n".join(evidence_blocks)

    # 3) Gemini向けプロンプト（推測禁止を明記）
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
# ここはStreamlit実行時に毎回上から評価されるが、
# @st.cache_data / @st.cache_resource を付けているので内部的には再利用されやすい

CSV_PATH = "dataset/yahoo_news_articles_preprocessed.csv"

# 1) CSV読み込み（必須列チェック・欠損補完）
df_news = load_news_csv(CSV_PATH)

# 2) DataFrame → docs（検索・根拠提示で使う辞書リスト）
docs = build_docs(df_news)

# 3) TF-IDFインデックス作成（word/char）
word_vec, word_mat, char_vec, char_mat = build_tfidf_indexes(docs)

# 4) Geminiモデル準備（APIキーで初期化）
gemini_model = get_gemini_model(GOOGLE_API_KEY)

# ★重要：word側の語彙（vocab）を作る
# normalize_query_for_tfidf() が vocab を参照し、質問文を語彙に寄せたトークン列にする
VOCAB = set(word_vec.get_feature_names_out())


# ============================
# Step6: UI（チャット）
# ============================
# 画面タイトルと固定パラメータ表示
st.title("🗨️ ニュース検索チャット（TF-IDF + Gemini RAG）")
st.caption(f"検索設定（固定）：alpha={ALPHA} / TOP_K={TOP_K} / MIN_SCORE={MIN_SCORE}")

# 会話履歴を保持する（Streamlitはrerunされるので session_state を使う）
if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存の履歴を描画
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 入力欄
user_input = st.chat_input("ニュースについて質問してください（例：日銀の政策金利について教えて）")

if user_input:
    # 1) ユーザーの発言を履歴に追加
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2) Retrieval（検索）
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

    # 3) 学習用：検索に使われたキーワード（word側）を表示（不要なら削除OK）
    if debug.get("q_word"):
        st.caption(f"🔎 抽出キーワード（word側）: {debug['q_word']}")

    # 4) 検索結果が0件なら、RAGはせず「見つからなかった」を返す
    if not results:
        gemini_block = (
            "**【Geminiの返答（RAG）】**\n\n"
            "関連する記事が見つからなかったため、根拠に基づく回答ができませんでした。\n"
            "もう少し具体的なキーワードで質問してください（例：日銀 政策金利 据え置き）。\n"
        )
        tfidf_block = "（検索結果：0件）"
        bot_reply = gemini_block + "\n---\n\n" + tfidf_block

    else:
        # 5) RAGプロンプト生成（根拠を埋め込む）
        rag_prompt = build_rag_prompt(
            question=user_input,
            results=results,
            query_keywords=debug["q_word"] or "(抽出なし：char検索のみでヒット)",
        )

        # 6) Gemini呼び出し（例外で落ちないようtry）
        try:
            rag_answer = gemini_ask(gemini_model, rag_prompt) or "(Geminiの返答が空でした)"
            gemini_block = f"**【Geminiの返答（RAG：根拠に基づく回答）】**\n\n{rag_answer}\n"
        except Exception as e:
            gemini_block = f"**【Geminiの返答（RAG）】**\n\nGemini呼び出しでエラー: {e}\n"

        # 7) 参照記事（透明性のため、タイトル/スコア/URLを出す）
        lines = [f"**【参照した記事（上位{len(results)}件）】**\n"]
        for rank, r in enumerate(results, start=1):
            lines.append(
                f"{rank}. **{r['title']}**  \n"
                f"　- topic: `{r['topic']}`  \n"
                f"　- score: `{r['score']:.4f}`  \n"
                f"　- url: {r['url']}\n"
            )
        tfidf_block = "\n".join(lines)

        # 8) 返答ブロック合成
        bot_reply = gemini_block + "\n---\n\n" + tfidf_block

    # 9) アシスタント返答を履歴に追加し、rerunで画面更新
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()