# 1. ライブラリのインストール
!pip install tf_keras tensorflowjs

import os
import tensorflow as tf
import tf_keras # Legacy Keras
from google.colab import files

# 環境変数の設定
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 共通の設定
IMG_SHAPE = (224, 224, 3)

# ==========================================
# A. MobileNet V2 の処理
# ==========================================
print("--- Processing MobileNet V2 ---")

# モデル構築
input_tensor_mobilenet = tf_keras.Input(shape=IMG_SHAPE)
model_mobilenet = tf_keras.applications.MobileNetV2(
    input_tensor=input_tensor_mobilenet,
    include_top=True,
    weights='imagenet'
)

# 保存と変換
model_mobilenet.save('mobilenet_v2.h5', save_format='h5')

!tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model \
    mobilenet_v2.h5 \
    tfjs_mobilenet_v2

# ZIP化とダウンロード
!zip -r tfjs_mobilenet_v2.zip tfjs_mobilenet_v2
files.download('tfjs_mobilenet_v2.zip')


# ==========================================
# B. ResNet50 の処理
# ==========================================
print("--- Processing ResNet50 ---")

# モデル構築
# ResNet50用に新しいInput Tensorを作成します
input_tensor_resnet = tf_keras.Input(shape=IMG_SHAPE)
model_resnet = tf_keras.applications.ResNet50(
    input_tensor=input_tensor_resnet,
    include_top=True,
    weights='imagenet'
)

# 保存と変換
model_resnet.save('resnet50.h5', save_format='h5')

!tensorflowjs_converter \
    --input_format=keras \
    --output_format=tfjs_layers_model \
    resnet50.h5 \
    tfjs_resnet50

# ZIP化とダウンロード
!zip -r tfjs_resnet50.zip tfjs_resnet50
files.download('tfjs_resnet50.zip')

print("--- All Done ---")