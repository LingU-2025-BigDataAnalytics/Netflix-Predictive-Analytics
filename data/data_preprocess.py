import re
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =========================
# 1. 固定随机种子，确保切分可复现
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 读取数据
# =========================
file_path = "dsotm_reviews.csv"
df = pd.read_csv(file_path)

print("原始数据形状:", df.shape)
print("原始列名:", df.columns.tolist())

# =========================
# 3. 数据清洗
#    - 保留需要的列
#    - 删除缺失值
#    - Rating 转数值
#    - 去重
#    - 去除空文本
# =========================
df = df[["Review", "Rating"]].copy()
df["Review"] = df["Review"].astype(str)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df.dropna(subset=["Review", "Rating"], inplace=True)
df.drop_duplicates(subset=["Review", "Rating"], inplace=True)

df["Review"] = df["Review"].str.strip()
df = df[df["Review"] != ""].copy()

print("清洗后数据形状:", df.shape)

def rating_to_label(rating):
    return "positive" if rating > 3.0 else "negative"

df["Sentiment"] = df["Rating"].apply(rating_to_label)

print("\n标签分布:")
print(df["Sentiment"].value_counts())

# =========================
# 5. 文本清洗 + 分词
#    这里做英文 baseline，使用正则分词
# =========================
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"http\S+|www\S+", " ", text)          # 去链接
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # 只保留字母和空格
    text = re.sub(r"\s+", " ", text).strip()             # 合并多余空格
    return text

def tokenize_text(text):
    # 简单英文分词：按单词切分
    return re.findall(r"\b[a-z]+\b", text)

df["clean_text"] = df["Review"].apply(clean_text)
df["tokens"] = df["clean_text"].apply(tokenize_text)

# 去掉分词后为空的样本
df = df[df["tokens"].apply(len) > 0].copy()

# 把分词结果重新拼回字符串，方便 TF-IDF 输入
df["processed_text"] = df["tokens"].apply(lambda x: " ".join(x))

print("\n预处理后样本数:", len(df))
print("示例分词结果:")
print(df[["Review", "processed_text", "Sentiment"]].head(3))

# =========================
# 6. 标签编码
# =========================
le = LabelEncoder()
df["label"] = le.fit_transform(df["Sentiment"])

print("\n标签映射:")
for cls_name, cls_id in zip(le.classes_, le.transform(le.classes_)):
    print(f"{cls_name} -> {cls_id}")

# =========================
# 7. 固定切分训练集/测试集
#    stratify 保证类别比例一致
# =========================
X = df["processed_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("\n训练集大小:", len(X_train))
print("测试集大小:", len(X_test))

# =========================
# 8. TF-IDF 特征
# =========================
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF 训练特征形状:", X_train_tfidf.shape)
print("TF-IDF 测试特征形状:", X_test_tfidf.shape)

# =========================
# 9. 逻辑回归 Baseline
# =========================
clf = LogisticRegression(
    random_state=SEED,
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train_tfidf, y_train)

# =========================
# 10. 预测与评估
# =========================
y_pred = clf.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", round(acc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 11. 保存切分结果，确保后续实验完全一致
# =========================
train_df = pd.DataFrame({
    "text": X_train,
    "label": y_train
})

test_df = pd.DataFrame({
    "text": X_test,
    "label": y_test
})

train_df.to_csv("train_fixed_split.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("test_fixed_split.csv", index=False, encoding="utf-8-sig")

print("\n已保存固定切分文件:")
print("train_fixed_split.csv")
print("test_fixed_split.csv")
