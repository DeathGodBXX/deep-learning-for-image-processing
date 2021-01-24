import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from model import MyModel


img = Image.open('5.jpg')
img = img.resize((28, 28))

img = np.array(img) / 255.0
img = np.expand_dims(img, 0)

model = MyModel()
# 声明模型输入尺寸。
model.build(input_shape=(1, 28, 28, 1))
model.load_weights('weight.h5')
model.summary()

for i in range(3):
    # 对3个通道，每个通道进行预测，看最终是否一致，通道会被压缩，需要再展开
    prediction = model.predict(np.expand_dims(img[:, :, :, i], 3))
    predict_class = np.argmax(np.squeeze(prediction))
    print(predict_class)


























