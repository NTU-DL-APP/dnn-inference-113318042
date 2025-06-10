import numpy as np
import json
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# ===== 1. 載入與預處理資料 =====
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ===== 2. 訓練模型（只用 Dense、ReLU、Softmax）=====
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ 測試準確率: {test_acc:.4f}")

# ===== 3. 存成 .h5 =====
model.save("fashion_mnist.h5")

# ===== 4. 匯出模型結構到 .json =====
arch = []
layer_id = 0

for layer in model.layers:
    layer_type = type(layer).__name__
    name = f"{layer_type}_{layer_id}"
    layer_id += 1

    if layer_type == "Flatten":
        arch.append({
            "name": name,
            "type": "Flatten",
            "config": {},
            "weights": []
        })
    elif layer_type == "Dense":
        activation = layer.get_config()["activation"]
        arch.append({
            "name": name,
            "type": "Dense",
            "config": {"activation": activation},
            "weights": [f"W{len(arch)}", f"b{len(arch)}"]
        })

with open("fashion_mnist.json", "w") as f:
    json.dump(arch, f, indent=2)

# ===== 5. 匯出模型權重到 .npz =====
weights = {}
dense_layer_count = 0

for layer in model.layers:
    if isinstance(layer, Dense):
        W, b = layer.get_weights()
        weights[f"W{dense_layer_count}"] = W
        weights[f"b{dense_layer_count}"] = b
        dense_layer_count += 1

np.savez("fashion_mnist.npz", **weights)

print("🎉 模型結構與權重已儲存完成！請將 .json 和 .npz 上傳至 ./model 資料夾。")