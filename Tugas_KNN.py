# ==========================================================
# TUGAS IMPLEMENTASI ALGORITMA K-NEAREST NEIGHBORS (KNN)
# Nama  : Tsinta Zahrotul Ahba
# NIM   : 112410100
# Kelas : Machine Learning
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# Konfigurasi SEED sesuai digit terakhir NIM (112410100 -> 0)
SEED = 0
np.random.seed(SEED)

print("--- 1. IDENTITAS & LIBRARY ---")
print("Versi NumPy:", np.__version__)
print("Versi Pandas:", pd.__version__)
print("Versi Scikit-Learn:", sklearn.__version__)

# ==========================================================
# BAGIAN A: Implementasi KNN dari Nol (NumPy) & Studi Kasus
# ==========================================================
class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", p=2, weights="uniform"):
        self.k = k
        self.metric = metric
        self.p = p
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2, axis=-1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b), axis=-1)
        elif self.metric == "minkowski":
            return np.sum(np.abs(a - b) ** self.p, axis=-1) ** (1 / self.p)
        else:
            raise ValueError(f"Metrik '{self.metric}' tidak dikenal.")

    def _predict_single(self, x):
        distances = self._distance(self.X_train, x)
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        if self.weights == "uniform":
            counts = np.bincount(k_nearest_labels)
            return np.argmax(counts)
        elif self.weights == "distance":
            epsilon = 1e-5
            weights = 1.0 / (k_distances ** 2 + epsilon)
            unique_labels = np.unique(k_nearest_labels)
            class_weights = {}
            for label in unique_labels:
                class_weights[label] = np.sum(weights[k_nearest_labels == label])
            best_label = max(class_weights, key=lambda k: (class_weights[k], -k_distances[k_nearest_labels == k].min()))
            return best_label

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x) for x in X])

# Fungsi Normalisasi
def min_max_scaler_fit_transform(X_train, X_test):
    min_val = np.min(X_train, axis=0)
    max_val = np.max(X_train, axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    return (X_train - min_val) / range_val, (X_test - min_val) / range_val

def z_score_scaler_fit_transform(X_train, X_test):
    mean_val = np.mean(X_train, axis=0)
    std_val = np.std(X_train, axis=0)
    std_val[std_val == 0] = 1.0
    return (X_train - mean_val) / std_val, (X_test - mean_val) / std_val

# Verifikasi Studi Kasus Kelulusan
print("\n--- VERIFIKASI DATASET STUDI KASUS ---")
X_kasus = np.array([
    [85, 80], [60, 55], [75, 70], [50, 65],
    [90, 85], [65, 50], [70, 75], [55, 60]
], dtype=float)
y_kasus = np.array([1, 0, 1, 0, 1, 0, 1, 0])
X_uji_kasus = np.array([[68, 64]], dtype=float)

print("Prediksi K=3 (Uniform)  :", KNNClassifier(k=3, weights="uniform").fit(X_kasus, y_kasus).predict(X_uji_kasus)[0])
print("Prediksi K=5 (Uniform)  :", KNNClassifier(k=5, weights="uniform").fit(X_kasus, y_kasus).predict(X_uji_kasus)[0])
print("Prediksi K=5 (Distance) :", KNNClassifier(k=5, weights="distance").fit(X_kasus, y_kasus).predict(X_uji_kasus)[0])

# ==========================================================
# BAGIAN B: Validasi Hasil dengan Scikit-Learn (Dataset Iris)
# ==========================================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

data_iris = load_iris()
X_main, y_main = data_iris.data, data_iris.target

X_tr, X_te, y_tr, y_te = train_test_split(
    X_main, y_main, test_size=0.2, stratify=y_main, random_state=SEED
)

custom_knn = KNNClassifier(k=3, metric="euclidean", weights="uniform").fit(X_tr, y_tr)
sk_knn = SklearnKNN(n_neighbors=3, metric="euclidean", weights="uniform").fit(X_tr, y_tr)

pred_custom = custom_knn.predict(X_te)
pred_sklearn = sk_knn.predict(X_te)
kesamaan = np.mean(pred_custom == pred_sklearn) * 100

print(f"\n--- VALIDASI SCIKIT-LEARN ---")
print(f"Persentase kesamaan prediksi Custom KNN vs Scikit-Learn: {kesamaan:.2f}%")

# ==========================================================
# BAGIAN C: Eksperimen, Grafik & Confusion Matrix
# ==========================================================
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

k_list = [3, 5, 7, 9]
metrics = ["euclidean", "manhattan"]
normalizations = ["none", "min_max", "z_score"]

hasil_eksperimen = []

for norm in normalizations:
    if norm == "none":
        X_tr_n, X_te_n = X_tr.copy(), X_te.copy()
    elif norm == "min_max":
        X_tr_n, X_te_n = min_max_scaler_fit_transform(X_tr, X_te)
    elif norm == "z_score":
        X_tr_n, X_te_n = z_score_scaler_fit_transform(X_tr, X_te)
        
    for met in metrics:
        for k in k_list:
            clf = KNNClassifier(k=k, metric=met, weights="uniform")
            clf.fit(X_tr_n, y_tr)
            preds = clf.predict(X_te_n)
            acc = accuracy_score(y_te, preds)
            hasil_eksperimen.append({
                "Normalisasi": norm,
                "Metrik": met,
                "K": k,
                "Akurasi": acc
            })

df_hasil = pd.DataFrame(hasil_eksperimen)

# Grafik 1: Garis K vs Akurasi
plt.figure(figsize=(8, 4))
for met in metrics:
    subset = df_hasil[(df_hasil["Metrik"] == met) & (df_hasil["Normalisasi"] == "none")]
    plt.plot(subset["K"], subset["Akurasi"], marker="o", linewidth=2, label=f"Metrik: {met}")

plt.title("Pengaruh Nilai K dan Metrik Jarak terhadap Akurasi")
plt.xlabel("Nilai K")
plt.ylabel("Akurasi")
plt.xticks(k_list)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show() # <-- Grafik pengaruh nilai K akan muncul di sini setelah kodenya dijalankan

# Grafik 2: Confusion Matrix
best_row = df_hasil.loc[df_hasil["Akurasi"].idxmax()]
sk_best = SklearnKNN(n_neighbors=int(best_row["K"]), metric=best_row["Metrik"]).fit(X_tr, y_tr)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    sk_best, X_te, y_te, display_labels=data_iris.target_names, cmap="Blues", ax=ax
)
plt.title(f"Confusion Matrix (Iris - K={best_row['K']}, {best_row['Metrik']})")
plt.show() # <-- Grafik Confusion Matrix akan muncul di sini

print("\nClassification Report (Macro):")
print(classification_report(y_te, sk_best.predict(X_te), target_names=data_iris.target_names))
