# Это все 3 первых задания, все выводы в таблице, файл прикрепила. Тут есть токен, поэтому осторожно

import pandas as pd
import re
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


# 1 заданиe
df = pd.read_csv("queries.csv", sep=';')
morph = MorphAnalyzer()
cache = {}
df.columns = ["query", "shows", ""]

def lemma(word):
    if word not in cache:
        cache[word] = morph.parse(word)[0].normal_form
    return cache[word]

def normalize(query):
    words = re.findall(r"[а-яёa-z0-9]+", query.lower())
    return " ".join(sorted(lemma(w) for w in words))

df["norm"] = df["query"].apply(normalize)
core = (df.groupby("norm")
          .agg(query=("query", "first"), shows=("shows", "sum"))
          .reset_index())
print(df[1:6])
print(f"было {len(df)}, после склейки {len(core)}")

groups = df.groupby("norm")["query"].apply(list)
pairs = [g for g in groups if len(g) > 1]
for g in pairs:
    print(f"{g[0]}  <->  {g[1]}")


# 2 заданиe
X = TfidfVectorizer().fit_transform(core["norm"])

k = max(2, len(core) // 20)
labels_tfidf = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

score = silhouette_score(X, labels_tfidf, metric="cosine")
print(f"k = {k}, силуэт = {score:.3f}")

core["cluster_tfidf"] = labels_tfidf
for c in core["cluster_tfidf"].drop_duplicates().sample(3, random_state=1):
    print(f"\n--- кластер {c}")
    print(core.loc[core["cluster_tfidf"] == c, "query"].head(5).to_string(index=False))


# 3 заданиe
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode([f"query: {q}" for q in core["norm"]],
                   normalize_embeddings=True, batch_size=64,
                   show_progress_bar=True)

for threshold in (0.10, 0.15, 0.20, 0.25):
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average").fit_predict(emb)
    n = len(set(labels))
    score = silhouette_score(emb, labels, metric="cosine") if 1 < n < len(core) else float("nan")
    print(f"порог {threshold}: кластеров {n}, силуэт {score:.3f}")


# --- кластеризация по эмбеддингам с порогом 0.1 ---
THRESHOLD = 0.1

labels_emb = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=THRESHOLD,
    metric="cosine",
    linkage="average"
).fit_predict(emb)

core["cluster_emb"] = labels_emb

n_clusters = core["cluster_emb"].nunique()
print(f"\ncluster_emb: порог = {THRESHOLD}, кластеров = {n_clusters}")

# силуэт (если кластеров больше 1 и меньше числа точек)
if 1 < n_clusters < len(core):
    score = silhouette_score(emb, labels_emb, metric="cosine")
    print(f"силуэт = {score:.3f}")

# размеры кластеров
print(core["cluster_emb"].value_counts().head(10))

# примеры по кластерам
for c in core["cluster_emb"].drop_duplicates().sample(
        min(3, n_clusters), random_state=1):
    print(f"\n--- cluster_emb = {c}")
    print(core.loc[core["cluster_emb"] == c, "query"]
          .head(5).to_string(index=False))
print(core)
