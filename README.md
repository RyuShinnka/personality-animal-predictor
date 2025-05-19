# 🐾 あなたはどの動物に似ている？AI分類アプリ

これは8つの性格特性をもとに、あなたがどの動物タイプに似ているかを分類するAIアプリです。  
機械学習モデル（教師あり学習）を用いて、「ネコ」「イヌ」「パンダ」など7種類の動物タイプの中から最も近いものを予測します。

---

## 📊 特徴量（Features）

以下の8つの性格指標に0〜10のスコアを入力します：

- 活動性  
- 社交性  
- 好奇心  
- 睡眠時間  
- 食べ物へのこだわり  
- 自立性  
- 甘えん坊度  
- ストレス耐性

---

## 🧠 使用技術（Tech Stack）

- Python / scikit-learn  
- GUI: Tkinter（※開発中）  
- データ前処理：pandas  
- モデル保存：joblib  

---

## 🏗️ プロジェクト構成

```
personality-animal-predictor/
├── data/
│   └── animal_personality_dataset.csv
├── models/
│   ├── best_model.pkl
│   └── encoder.pkl
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
├── assets/
│   └── design.png
├── main.py
```

---

## 🚀 実行方法（Usage）

### 1. モデル学習

python src/train.py

### 2. 単体診断（predict.py）


python src/predict.py
# → あなたは「ネコ」タイプです！


---


## 📄 ライセンス

このプロジェクトは学習用・展示用です。
