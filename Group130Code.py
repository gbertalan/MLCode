import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo", "-q"]) # installing ucimlrepo

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # for graph legends

from matplotlib.colors import LinearSegmentedColormap # for conf. matrix coloring
from collections import Counter

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold, # to create multiple train/test splits for cross-validation
    GridSearchCV, # for finding best hyperparams
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler # standardizing to mean 0 and standard dev. 1
from sklearn.neighbors import KNeighborsClassifier # sklearn's kNN
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    precision_score, # true positives / (true pos+false pos)
    recall_score, # true positives / (true pos+false neg)
    confusion_matrix,
)
from sklearn.base import BaseEstimator, ClassifierMixin # wrapping custom kNN to work with the sklearn things

from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 42
TEST_SIZE = 0.20
NO_OF_FOLDS = 10 # splitting into 10 folds
NO_OF_REPEATS = 5
FIXED_NEIGHBORS = 5
N_TIMING_REPS = 5 # repeating time measurements, then averaging them

# for consistent style in the plots:
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

# where we save the plots:
os.makedirs("plots", exist_ok=True)

# from-scratch kNN:
def distance_squared(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 + (a[3]-b[3])**2)

def kNN(X_train, y_train, X_test, k=3):
    predicted_labels = []
    for test_point in X_test:
        distances = [distance_squared(test_point, train_point)
                     for train_point in X_train]
        k_indices = sorted(range(len(distances)),
                           key=lambda i: distances[i])[:k]
        labels_of_k_neighbors = [y_train[i] for i in k_indices]
        predicted_labels.append(Counter(labels_of_k_neighbors).most_common(1)[0][0])
    return predicted_labels

# wrapper that works with sklearn so we can use cross_val_score with custom kNN:
class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        return np.array(kNN(self.X_train_, self.y_train_, X, k=self.k))

# loading dataset:
print("Downloading dataset, please wait!")
dataset = fetch_ucirepo(id=267)
print(dataset.metadata.name)

X = dataset.data.features.values
y = dataset.data.targets.values.ravel() # flattened to 1D array

feature_name_array = ["Variance", "Skewness", "Curtosis", "Entropy"]

print(f"{X.shape[0]} samples, {X.shape[1]} features")
print(f"genuine(0)={(y==0).sum()}, forged(1)={(y==1).sum()}\n")

# feature statistics:
print("feature statistics (mean +/- std):")
for i, name in enumerate(feature_name_array):
    m0 = X[y == 0, i].mean();  s0 = X[y == 0, i].std() # genuine
    m1 = X[y == 1, i].mean();  s1 = X[y == 1, i].std() # forged
    print(f" : {name:10s}  Genuine: {m0:+.2f} +/- {s0:.2f}   Forged: {m1:+.2f} +/- {s1:.2f}")
print()

# feature distributions:
distr_plot, axes = plt.subplots(2, 2, figsize=(6.5, 4.0))
distr_plot.subplots_adjust(hspace=0.45, wspace=0.35)

for axis, i, name in zip(axes.flatten(), range(4), feature_name_array):
    d0 = X[y == 0, i]
    d1 = X[y == 1, i]
# min and max for histogram range:
    lo = min(d0.min(), d1.min())
    hi = max(d0.max(), d1.max())

    bins = np.linspace(lo, hi, 35)
    axis.hist(d0, bins=bins, alpha=0.5, color=(0.0, 0.5, 0.0), density=True)
    axis.hist(d1, bins=bins, alpha=0.5, color="tomato",  density=True)
    axis.set_xlabel(name)
    axis.set_ylabel("Density")
# remove borders:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

# custom legend handles:
handles = [
    mpatches.Patch(color=(0.0, 0.5, 0.0), alpha=0.65, label="Genuine (class 0)"),
    mpatches.Patch(color="tomato",  alpha=0.65, label="Forged (class 1)"),
]

distr_plot.legend(handles=handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.02), frameon=False)
distr_plot.savefig("plots/distributions.png", bbox_inches="tight")
plt.show()
print("distributions.png saved to plots folder.\n")

# train/test split:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, # features, labels
    test_size=TEST_SIZE, # 20%
    stratify=y,
    random_state=RANDOM_SEED, # for reproducability
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

# standardizing features:
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train) # (x-mean)/std
X_test_standardized  = scaler.transform(X_test)

# 10 folds, 5 reps -> 50 different train/val splits
cross_validator = RepeatedStratifiedKFold(
    n_splits = NO_OF_FOLDS,
    n_repeats = NO_OF_REPEATS,
    random_state = RANDOM_SEED,
)

# hyperparameter grid search (training set only)

print("Running hyperparameter grid searches.")
print("kNN uses our custom implementation, please wait!")

# kNN, uses custom implementation via the wrapper
knn_param_grid = [1, 3, 5, 7, 9, 11, 15, 21]
best_k = None
best_k_score = -1

for current_k in knn_param_grid:
    temp_classifier = CustomKNN(k=current_k)
    scores = cross_val_score(temp_classifier, X_train_standardized, y_train,
                             cv=cross_validator, scoring="accuracy", n_jobs=1)
    mean_score = scores.mean()
    print(f"    k={current_k:2d}  CV acc = {mean_score*100:.2f}%")
    if mean_score > best_k_score:
        best_k_score = mean_score
        best_k = current_k

print(f"Best k = {best_k}\n")

# LR and DT use sklearn directly
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED),
    param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    cv=cross_validator, scoring="accuracy", n_jobs=-1,
)
lr_grid.fit(X_train_standardized, y_train)
best_C = lr_grid.best_params_["C"]
print(f"LR: best C = {best_C}")

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    param_grid={
        "max_depth":         [3, 5, 7, 10, None],
        "criterion":         ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
    },
    cv=cross_validator, scoring="accuracy", n_jobs=-1,
)
dt_grid.fit(X_train, y_train)
best_dtparams = dt_grid.best_params_
print(f"DT: best params = {best_dtparams}\n")

# training each model on all training data with best hyperparams:

custom_kNN = CustomKNN(k=best_k)
custom_kNN.fit(X_train_standardized, y_train)

reference_kNN = KNeighborsClassifier(n_neighbors=best_k, algorithm="kd_tree")
reference_kNN.fit(X_train_standardized, y_train)

logistic_regression  = LogisticRegression(C=best_C, max_iter=1000,
                             solver="lbfgs", random_state=RANDOM_SEED)
logistic_regression.fit(X_train_standardized, y_train)

decision_tree  = DecisionTreeClassifier(random_state=RANDOM_SEED, **best_dtparams)
decision_tree.fit(X_train, y_train)

print("Computing final CV accuracy with best hyperparameters:")

# computing mean and std of cross-validation accuracy
def cv_stats(clf, Xtr, ytr):
    scores = cross_val_score(clf, Xtr, ytr, cv=cross_validator,
                             scoring="accuracy", n_jobs=1)
    return scores.mean() * 100, scores.std() * 100

acc_custom_m, acc_custom_s = cv_stats(custom_kNN, X_train_standardized, y_train)
print(f"Custom kNN done: {acc_custom_m:.1f} +/- {acc_custom_s:.1f}%")
acc_ref_m,    acc_ref_s    = cv_stats(reference_kNN,    X_train_standardized, y_train)
print(f"Reference kNN done: {acc_ref_m:.1f} +/- {acc_ref_s:.1f}%")
acc_lr_m,     acc_lr_s     = cv_stats(logistic_regression,     X_train_standardized, y_train)
print(f"LR done: {acc_lr_m:.1f} +/- {acc_lr_s:.1f}%")
acc_dt_m,     acc_dt_s     = cv_stats(decision_tree,     X_train,   y_train)
print(f"DT done: {acc_dt_m:.1f} +/- {acc_dt_s:.1f}%\n")

# test-set evaluation:

custom_kNN_prediction = custom_kNN.predict(X_test_standardized)
prec_custom = precision_score(y_test, custom_kNN_prediction) # how many were really forged among those predicted as forged.
rec_custom  = recall_score(y_test, custom_kNN_prediction)
cm_custom   = confusion_matrix(y_test, custom_kNN_prediction)

reference_kNN_prediction = reference_kNN.predict(X_test_standardized)
prec_reference = precision_score(y_test, reference_kNN_prediction)
rec_reference  = recall_score(y_test, reference_kNN_prediction)
cm_reference   = confusion_matrix(y_test, reference_kNN_prediction)

logistic_regression_prediction = logistic_regression.predict(X_test_standardized)
prec_logistic_reg = precision_score(y_test, logistic_regression_prediction)
rec_logistic_reg  = recall_score(y_test, logistic_regression_prediction)
cm_logistic_reg   = confusion_matrix(y_test, logistic_regression_prediction)

decision_tree_prediction = decision_tree.predict(X_test)
prec_dec_tree = precision_score(y_test, decision_tree_prediction)
rec_dec_tree  = recall_score(y_test, decision_tree_prediction)
cm_dec_tree   = confusion_matrix(y_test, decision_tree_prediction)

baseline_acc = (y_test == 0).mean() * 100

print("Classification results:")
rows = [
    (f"Custom kNN (k={best_k})",    acc_custom_m, acc_custom_s, prec_custom, rec_custom),
    (f"Reference kNN (k={best_k})", acc_ref_m,    acc_ref_s,    prec_reference, rec_reference),
    ("Logistic Regression",         acc_lr_m,     acc_lr_s,     prec_logistic_reg, rec_logistic_reg),
    ("Decision Tree",               acc_dt_m,     acc_dt_s,     prec_dec_tree, rec_dec_tree),
]
print(f"{'Classifier':<26} {'CV Acc (%)':>15}  {'Prec':>6}  {'Recall':>6}")
print(f"{'-'*60}")
for name, m, s, p, r in rows:
    print(f"{name:<26} {m:>6.1f} +/- {s:<5.1f}  {p:>6.3f}  {r:>6.3f}")
print(f"{'Majority baseline':<26} {baseline_acc:>6.1f}           "
      f"{'n/a':>6}  {'0.000':>6}")
print()

# our kNN and ref. kNN has to match:
if np.array_equal(custom_kNN_prediction, reference_kNN_prediction):
    print("Custom and reference kNN produce the same predictions.")
else:
    n_diff = (custom_kNN_prediction != reference_kNN_prediction).sum()
    print(f"WARNING: Custom and reference differ on {n_diff} instances.")
print()

# confusion matrices:

# mapping classifier names to their confusion matrices:
cms_dict = {
    f"Custom kNN ($k={best_k}$)":    cm_custom,
    f"Reference kNN ($k={best_k}$)": cm_reference,
    "Logistic Regression":            cm_logistic_reg,
    "Decision Tree":                  cm_dec_tree,
}

# color map:
cmap_colors = LinearSegmentedColormap.from_list("gradient_color", ["#f6fcf2", "#087d5c"])

conf_matrix_plot, axes2 = plt.subplots(2, 2, figsize=(6.5, 4.5))
conf_matrix_plot.subplots_adjust(hspace=0.55, wspace=0.4)

# going through each classifier, calculating accuracy, showing conf. matrix
for axis, (title, cm) in zip(axes2.flatten(), cms_dict.items()):
    
    acc_cm = (cm[0,0] + cm[1,1]) / cm.sum() * 100 # (true neg. + true pos.) / total
    axis.imshow(cm, cmap=cmap_colors, aspect="auto", vmin=0, vmax=cm.max())

    # predictions:
    axis.set_xticks([0,1])
    axis.set_xticklabels(["Pred: Gen.", "Pred: For."], fontsize=7)

    # true values:
    axis.set_yticks([0,1])
    axis.set_yticklabels(["True: Gen.", "True: For."], fontsize=7)

    axis.set_title(f"{title}\nAcc = {acc_cm:.1f}%", fontsize=8, pad=4)
    # writing numbers into cells:
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i,j] > cm.max() * 0.5 else "black" # to keep text color visible
            axis.text(j, i, str(cm[i,j]), ha="center", va="center",
                    fontsize=11, color=col, fontweight="bold")

conf_matrix_plot.savefig("plots/confusion_matrix.png", bbox_inches="tight")
plt.show()
print("confusion_matrix.png saved to plots folder.\n")

# runtime comparison:
train_sizes = [100, 200, 300, 400] # same range for both implementations
rng = np.random.default_rng(RANDOM_SEED)

custom_kNN_times, reference_kNN_times = [], []

print("Measuring prediction times, please wait!")
for n in train_sizes:

    # randomly selecting n samples from the training set:
    indices = rng.choice(X_train_standardized.shape[0], size=n, replace=False)
    Xn  = X_train_standardized[indices]
    yn  = y_train[indices]

    # custom kNN:
    t = []
    for _ in range(N_TIMING_REPS):
        t0 = time.perf_counter() # start time
        kNN(Xn, yn, X_test_standardized, k=FIXED_NEIGHBORS)
        t.append((time.perf_counter() - t0) * 1000) # duration in ms
    custom_kNN_times.append(np.median(t))

    # reference kNN:
    ref_kNN = KNeighborsClassifier(n_neighbors=FIXED_NEIGHBORS, algorithm="kd_tree")
    ref_kNN.fit(Xn, yn)
    t = []
    for _ in range(N_TIMING_REPS):
        t0 = time.perf_counter()
        ref_kNN.predict(X_test_standardized)
        t.append((time.perf_counter() - t0) * 1000)
    reference_kNN_times.append(np.median(t))

    print(f"  n={n:5d}  custom={custom_kNN_times[-1]:.1f} ms  "
          f"kd_tree={reference_kNN_times[-1]:.2f} ms")

speedup = custom_kNN_times[-1] / reference_kNN_times[-1]
print(f"\n  Speedup at n={train_sizes[-1]}: {speedup:.1f}x\n")

# line plot of runtimes:
runtime_comparison_plot, axis = plt.subplots(figsize=(5.5, 3.0))
axis.plot(train_sizes, custom_kNN_times, "o-",  color=(0.0, 0.5, 0.7),
         linewidth=1.8, markersize=4.5, label="Custom kNN (naive search)")
axis.plot(train_sizes, reference_kNN_times,    "o-", color=(0.0, 0.7, 0.5),
         linewidth=1.8, markersize=4.5, label="Reference kNN (KD-tree)")
axis.set_xlabel("Training set size ($n$)")
axis.set_ylabel("Prediction time (ms)")
axis.legend(frameon=False)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
runtime_comparison_plot.savefig("plots/runtime.png", bbox_inches="tight")
plt.show()
print("runtime.png saved to plots folder.\n")

# Summary:
print("Summary:")

print(f"Class balance : genuine {(y==0).sum()} ({(y==0).mean()*100:.1f}%)  "f"forged {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
print(f"Train / test  : {X_train.shape[0]} / {X_test.shape[0]}")
print(f"Baseline acc  : {baseline_acc:.1f}%")

print()
for name, m, s, p, r in rows:
    print(f"{name:<25}  acc={m:.1f}+/-{s:.1f}%  prec={p:.3f}  rec={r:.3f}")
    
print()
print(f"Speedup at n={train_sizes[-1]}: {speedup:.0f}x")
print(f"Custom time at n={train_sizes[-1]}: {custom_kNN_times[-1]:.0f} ms")
print(f"Ref time at n={train_sizes[-1]}: {reference_kNN_times[-1]:.1f} ms")

print()
print(f"Selected k  : {best_k}")
print(f"Selected C  : {best_C}")
print(f"Selected DT : {best_dtparams}")

print()
print("Finished.")
