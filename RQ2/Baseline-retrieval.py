import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score
)

# 1. 读取数据
df = pd.read_csv("Data/with_similarity.csv")
y_true = df["label"].map({"yes": 1, "no": 0}).values

def evaluate_thresholds(y_true, y_scores, name):
    results = []
    best_macro, best_micro = (-1, None), (-1, None)

    for thresh in np.arange(0, 1, 0.001):
        y_pred = (y_scores >= thresh).astype(int)

        # Accuracy
        acc = accuracy_score(y_true, y_pred)

        # Precision/Recall/F1 per class
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 0], zero_division=0
        )

        # Macro/Micro F1
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        results.append([thresh, acc,
                        precision[0], recall[0], f1[0],   # 正类 yes
                        precision[1], recall[1], f1[1],   # 负类 no
                        macro_f1, micro_f1])

        # 记录最佳
        if macro_f1 > best_macro[0]:
            best_macro = (macro_f1, thresh)
        if micro_f1 > best_micro[0]:
            best_micro = (micro_f1, thresh)

    results_df = pd.DataFrame(results, columns=[
        "threshold", "accuracy",
        "precision_yes", "recall_yes", "f1_yes",
        "precision_no", "recall_no", "f1_no",
        "macro_F1", "micro_F1"
    ])
    results_df.to_csv(f"results_{name}.csv", index=False)

    print(f"\n=== {name} ===")
    print(f"Best Macro-F1: {best_macro[0]:.4f} at threshold={best_macro[1]:.2f}")
    print(f"Best Micro-F1: {best_micro[0]:.4f} at threshold={best_micro[1]:.2f}")

    return results_df


# 2. 分别评估 SBERT 和 TF-IDF
results_sbert = evaluate_thresholds(y_true, df["cosine_similarity_sbert"].values, "sbert")
results_tfidf = evaluate_thresholds(y_true, df["cosine_similarity_tfidf"].values, "tfidf")

# 3. 绘制曲线对比（阈值 vs Macro/Micro F1）
plt.figure(figsize=(10,6))

# SBERT
plt.plot(results_sbert["threshold"], results_sbert["macro_F1"],
         label="SBERT Macro-F1", color="blue", linestyle="-", marker="o", markersize=3)


# TF-IDF
plt.plot(results_tfidf["threshold"], results_tfidf["macro_F1"],
         label="TF-IDF Macro-F1", color="green", linestyle="-", marker="s", markersize=3)


plt.xlabel("Threshold")
plt.ylabel("Macro-F1 Score")
plt.title("Macro-F1 vs Threshold (SBERT vs TF-IDF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("F1_comparison.png", dpi=300)
plt.show()

