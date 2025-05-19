# ライブラリをインポートする
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib as jl
from sklearn.preprocessing import LabelEncoder
from preprocess import load_dataset

# データを読み込む
path = "data/animal_personality_dataset.csv"
X, y = load_dataset(path)

# ラベル（動物名）を数値に変換する
le = LabelEncoder()
y = le.fit_transform(y)

# データを訓練データとテストデータに分ける（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#  GridSearchCV を使って、各分類モデルの最適なパラメータを探索する関数を定義する
def tune_model(model_name, model, params, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"モデル: {model_name}")
    print(f"  最良パラメータ: {grid.best_params_}")
    print(f"  テスト精度: {acc:.4f}")
    print(f"  交差検証平均精度: {grid.best_score_:.4f}")
    print("------")

    return best_model, acc


#  LogisticRegression / KNeighborsClassifier / DecisionTreeClassifier のパラメータを定義する
model_configs = {
    "LogisticRegression":(LogisticRegression(), {"C":[0.01, 0.1, 1, 10]})
    ,"KNeighborsClassifier" : (KNeighborsClassifier(), {"n_neighbors":[1, 3, 5, 7]})
    ,"DecisionTreeClassifier": (DecisionTreeClassifier(), {"max_depth":[3, 5 ,10, None], "min_samples_split":[2, 5, 10]})
}
# 定義したモデルとパラメータの辞書を使って、全てのモデルをチューニングする
results = []
for name, (model, params) in model_configs.items():
    best_model, acc = tune_model(name, model, params, X_train, y_train, X_test, y_test)
    results.append((name, best_model, acc))

# チューニング済みのモデルの中から、最もテスト精度が高かったモデルを選ぶ
best_model_name = ""
best_model = None
best_acc = 0
for name, model, acc in results:
    if acc > best_acc:
        best_model_name = name
        best_model = model
        best_acc = acc


# 最も良かったモデルと encoder を保存する

jl.dump(le, "models/encoder.pkl")
jl.dump(best_model, "models/best_model.pkl")
print("保存されました")