# ライブラリをインポートする
import pandas as pd

path = "data/animal_personality_dataset.csv"


# 関数名：load_dataset
def load_dataset(path):
    df = pd.read_csv(path)
    X =df.iloc[:, :-1]
    y =df["動物"]
    return X, y

