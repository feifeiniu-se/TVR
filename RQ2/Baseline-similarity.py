import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 如果用 sentenceBERT
from sentence_transformers import SentenceTransformer

# 1. 读取CSV
df = pd.read_csv("Data/pairs.csv", encoding="utf-8")

# 2. 初始化 SentenceBERT 模型（只加载一次）
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # 轻量级常用模型

# 3. 定义函数：计算相似度
def compute_similarity(stakeholder_text, system_text, mode="tfidf"):
    if mode == "tfidf":
        # 用 TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([stakeholder_text, system_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return sim[0, 0]
    elif mode == "sbert":
        # 用 Sentence-BERT
        embeddings = sbert_model.encode([stakeholder_text, system_text])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])
        return sim[0, 0]
    else:
        raise ValueError("mode 必须是 'tfidf' 或 'sbert'")

# 4. 应用到每一行（比如选择 sbert）
df["cosine_similarity_sbert"] = df.apply(
    lambda row: compute_similarity(row["stakeholder"], row["system"], mode="sbert"), axis=1
)

df["cosine_similarity_tfidf"] = df.apply(
    lambda row: compute_similarity(row["stakeholder"], row["system"], mode="tfidf"), axis=1
)

# 5. 保存结果（可选）
df.to_csv("Data/with_similarity_tfidf+sbert.csv", index=False)

