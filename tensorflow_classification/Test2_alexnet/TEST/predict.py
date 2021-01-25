from model import AlexNet_v1,AlexNet_v2

import numpy as np
import matplotlib.pyplot as plt
import json, os
from PIL import Image


def main():
    img_height, img_width = 224, 224

    image_path = './image/1.jpg'
    assert os.path.exists(image_path), " {} doesn't exist".format(image_path)
    img = Image.open(image_path)

    # plt.imshow(img)
    # plt.show()
    
    json_file = 'class_indices.json'
    assert os.path.exists(json_file), " {} doesn't exist".format(json_file)
    json_file = open(json_file, "r")
    json_content = json.load(json_file)

    weight_file = './save_weight/myAlex_v2.h5'
    # weight_file = './save_weight/myAlex.ckpt'
    assert os.path.exists(weight_file), " {} doesn't exist".format(weight_file)

    model = AlexNet_v1(num_classes=5)
    # model = AlexNet_v2(num_classes=5)
    # model.build(input_shape=(1, img_height, img_width, 3))
    # model.evaluate()
    model.load_weights(weight_file)
    model.summary()

    img = img.resize((img_height, img_width))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    result = model(img)
    # result = model.predict(img)
    prediction = np.squeeze(result)

    predict_class = np.argmax(prediction)
    print(predict_class, type(predict_class))
    print(json_content[str(predict_class)], prediction[predict_class])


if __name__ == '__main__':
    main()













































