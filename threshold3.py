import numpy as np
from sklearn.metrics import roc_curve

errors_good = np.array(
    [
        0.006737,
        0.006828,
        0.007091,
        0.006997,
        0.006610,
        0.006795,
        0.006872,
        0.006725,
        0.006873,
        0.006817,
    ]
)
errors_bad = np.array(
    [
        0.007432,
        0.007455,
        0.008369,
        0.007183,
        0.007653,
        0.008856,
        0.007531,
        0.007624,
        0.007457,
        0.008424,
    ]
)

errors = np.concatenate([errors_good, errors_bad])
labels = np.array([0] * len(errors_good) + [1] * len(errors_bad))

fpr, tpr, thresholds = roc_curve(labels, errors)

print("Threshold\tTPR\t\tFPR")
for thr, t, f in zip(thresholds, tpr, fpr):
    print(f"{thr:.4f}\t\t{t:.4f}\t{f:.4f}")

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"\nBest threshold (Youden J): {best_threshold:.4f}")
