# -*- coding: utf-8 -*-
"""MachineLearning_Kelompok 105.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rmO1ytk0KbyPWfCvET-dZF9JVXrD28J9

# <center>Project Bassed Assessment Machine Learning </center>

<b>Topik</b>
Klasifikasi pada Dataset Raisin

<b>Kelompok: 105</b>

Anggota:
1. Johan Samser Naibaho - 1301220012
2. Ervan Maulana - 1301223099
3. Tyo Firmansyah Akip - 1301223163

#### Pendahuluan

Dataset ini menyediakan gambar kismis varietas Kecimen dan Besni yang ditanam di Turki. Dataset ini terdiri dari 900 gambar kismis, dengan 450 gambar untuk setiap varietas. Gambar-gambar ini telah diolah terlebih dahulu dan 7 fitur morfologis diekstrak dari setiap gambar. Fitur-fitur ini diklasifikasikan menggunakan tiga teknik machine learning yang berbeda.

Dataset ini digunakan untuk mengembangkan sistem kecerdasan buatan yang dapat membedakan/mengklasifikasikan antara kismis Kecimen dan Besni.

**Informasi Variabel :**

Fitur :
- Area: Luas kismis dalam piksel.
- Perimeter: Keliling kismis dihitung dari jarak batas kismis dengan piksel di sekitarnya.
- MajorAxisLength: Panjang sumbu utama kismis, yaitu garis terpanjang yang dapat digambar pada kismis.
- MinorAxisLength: Panjang sumbu minor kismis, yaitu garis terpendek yang dapat digambar pada kismis.
- Eccentricity: Ukuran ketakbulatan elips yang memiliki momen sama dengan kismis.
- ConvexArea: Jumlah piksel pada kulit cembung terkecil dari bentuk kismis.
- Extent: Rasio luas kismis dengan total piksel pada kotak pembatas.

Target :
- Class : Varietas kismis (Kecimen atau Besni).

## Import Library & Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("/content/Raisin_Dataset.csv")
df

"""##EDA

### Overview
"""

df.info()
df.shape

df.describe().T

"""### Cek data duplikat"""

df.duplicated().sum()

"""###Cek missing values"""

df.isnull().sum()

"""###Encode Target"""

df['Class'] = df['Class'].map({'Kecimen': 1, 'Besni': 0})

"""###Distribusi Target"""

counts = df['Class'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=counts.index, y=counts.values)
plt.xticks([0, 1], ['0', '1'])
plt.title('Class')
plt.xlabel('Class (0 = Besni , 1 = Kecimen)')
plt.ylabel('Count')
plt.show()

counts

"""###Pisah Fitur dan Target"""

X = df.drop(columns=['Class'])
y = df['Class']

"""### Cek Outliers"""

num_plots = len(X.columns)
cols = 10
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))

for idx, column in enumerate(X.columns):
    row_idx = idx // cols
    col_idx = idx % cols
    ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

    sns.boxplot(y=X[column], ax=ax)
    ax.set_title(f"Boxplot of {column}", fontsize=12)
    ax.set_ylabel(column, fontsize=10)

for idx in range(num_plots, rows * cols):
    fig.delaxes(axes.flatten()[idx])

plt.tight_layout()
plt.show()

"""##Preprocessing

###Handle Outliers
"""

def handle_outliers(X):
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        X.loc[X[col] > upper_bound, col] = Q3
        X.loc[X[col] < lower_bound, col] = Q1

    return X

X = handle_outliers(X)

"""####Boxplot setelah handle outlier"""

num_plots = len(X.columns)
cols = 10
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))

for idx, column in enumerate(X.columns):
    row_idx = idx // cols
    col_idx = idx % cols
    ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]

    sns.boxplot(y=X[column], ax=ax)
    ax.set_title(f"Boxplot of {column}", fontsize=12)
    ax.set_ylabel(column, fontsize=10)

for idx in range(num_plots, rows * cols):
    fig.delaxes(axes.flatten()[idx])

plt.tight_layout()
plt.show()

"""### Normalisasi dengan Z-Score"""

mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)

z_scores = (X - mean) / std_dev

X = z_scores

X

"""simpan mean dan std"""

np.save('mean.npy', mean)
np.save('std_dev.npy', std_dev)

"""##Modeling

###Acak Dataset
"""

df = pd.concat([X, y], axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df

"""### Split dataset menjadi training dan testing"""

# (80% untuk train, 20% untuk test)
split_index = int(0.8 * len(df))

train_df = df[:split_index]
test_df = df[split_index:]

X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']

X_train.head()

y_train.head()

X_test.head()

y_test.head()

print(f"Data Training: {len(X_train)}")
print(f"Data Testing: {len(X_test)}")

"""##Base Model

### 1.KNN
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

for k in [3, 5, 7]:
  knn_model = KNeighborsClassifier(n_neighbors=k)

  knn_model.fit(X_train, y_train)

  y_pred = knn_model.predict(X_test)

  print("\nK =", k)
  print("Accuracy:", accuracy_score(y_test, y_pred))
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  # Buat confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  labels = ['Besni', 'Kesimen']

  plt.figure(figsize=(6, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.title('Confusion Matrix')
  plt.tight_layout()
  plt.show()

"""### 2.Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred)

labels = ['Besni', 'Kesimen']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""### 3.Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Buat dan latih model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Prediksi data test
y_pred = nb_model.predict(X_test)

# Evaluasi akurasi dan classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Sesuaikan label sesuai kelas di datasetmu
labels = ['Besni', 'Kesimen']  # Ganti sesuai kelas sebenarnya

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Naive Bayes')
plt.tight_layout()
plt.show()

"""## Stacking"""

from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

# Base models
base_models = [
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('nb', GaussianNB())
]

# Meta model
meta_model = LogisticRegression()

# Stacking ensemble
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking_model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# Skor yang ingin dievaluasi di setiap fold
scoring = ['accuracy', 'precision', 'recall', 'f1']

# Cross-validation
cv_results = cross_validate(
    stacking_model,
    X_train,
    y_train,
    cv=5,
    scoring=scoring
)

# Cetak skor per fold
for i in range(5):
    print(f"Fold {i+1}: "
          f"Accuracy = {cv_results['test_accuracy'][i]:.4f}, "
          f"Precision = {cv_results['test_precision'][i]:.4f}, "
          f"Recall = {cv_results['test_recall'][i]:.4f}, "
          f"F1-Score = {cv_results['test_f1'][i]:.4f}")

# Cetak rata-rata dari semua metrik
print("\nRata-rata:")
print(f"Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision : {cv_results['test_precision'].mean():.4f}")
print(f"Recall    : {cv_results['test_recall'].mean():.4f}")
print(f"F1-Score  : {cv_results['test_f1'].mean():.4f}")

y_pred = stacking_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred)

labels = ['Besni', 'Kesimen']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

"""Simpan Model"""

import joblib
joblib.dump(stacking_model, 'stacking_model.pkl')