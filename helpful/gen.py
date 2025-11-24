import os
import subprocess
import sys

# ==========================================
# 1. ライブラリのインストール
# ==========================================
# !pip install ... の代わりに subprocess を使用
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Colab環境以外で誤って走らないようにするガード（任意）
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("Installing dependencies...")
    install("tf_keras")
    install("tensorflowjs")

import tensorflow as tf
import tf_keras # Legacy Keras

# 環境変数の設定
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 共通の設定
IMG_SHAPE = (224, 224, 3)

# ==========================================
# A. MobileNet V2 の処理
# ==========================================
print("--- Processing MobileNet V2 ---")

input_tensor_mobilenet = tf_keras.Input(shape=IMG_SHAPE)
model_mobilenet = tf_keras.applications.MobileNetV2(
    input_tensor=input_tensor_mobilenet,
    include_top=True,
    weights='imagenet'
)

model_mobilenet.save('mobilenet_v2.h5', save_format='h5')

# !tensorflowjs_converter ... の代わりに os.system を使用
# 注意: コマンドライン引数を1つの文字列として渡します
cmd_mobilenet = (
    "tensorflowjs_converter "
    "--input_format=keras "
    "--output_format=tfjs_layers_model "
    "mobilenet_v2.h5 "
    "tfjs_mobilenet_v2"
)
os.system(cmd_mobilenet)

# ZIP化
# !zip ... の代わり
os.system("zip -r tfjs_mobilenet_v2.zip tfjs_mobilenet_v2")

# ダウンロード（Colab専用機能）
if IN_COLAB:
    from google.colab import files
    files.download('tfjs_mobilenet_v2.zip')


# ==========================================
# B. ResNet50 の処理
# ==========================================
print("--- Processing ResNet50 ---")

input_tensor_resnet = tf_keras.Input(shape=IMG_SHAPE)
model_resnet = tf_keras.applications.ResNet50(
    input_tensor=input_tensor_resnet,
    include_top=True,
    weights='imagenet'
)

model_resnet.save('resnet50.h5', save_format='h5')

cmd_resnet = (
    "tensorflowjs_converter "
    "--input_format=keras "
    "--output_format=tfjs_layers_model "
    "resnet50.h5 "
    "tfjs_resnet50"
)
os.system(cmd_resnet)

os.system("zip -r tfjs_resnet50.zip tfjs_resnet50")

if IN_COLAB:
    files.download('tfjs_resnet50.zip')

print("--- All Done ---")