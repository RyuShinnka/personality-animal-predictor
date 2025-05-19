import joblib as jl

def main():
    # モデルとエンコーダーを読み込む
    le = jl.load("models/encoder.pkl")
    model = jl.load("models/best_model.pkl")

    # 予測したいデータを用意する
    X = [[3, 1, 2, 0, 1, 3, 2, 4]]

    # 予測を実行
    y = model.predict(X)
    y_real = le.inverse_transform(y)

    # 結果を表示
    print(f"あなたは「{y_real[0]}」タイプです！")

# スクリプトとして実行されたときのみ main() を呼ぶ
if __name__ == "__main__":
    main()
