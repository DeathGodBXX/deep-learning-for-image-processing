from tensorflow.keras import Model, Sequential, layers, models


# input(None, 224, 224, 3)
# output(None, 227, 227, 3)
# output(None, 55, 55, 48)
# output(None, 27, 27, 48)
# output(None, 27, 27, 128)
# output(None, 13, 13, 128)
# output(None, 13, 13, 192)
# output(None, 13, 13, 192)
# output(None, 13, 13, 128)
# output(None, 6, 6, 128)

# padding == 'valid', (F-W+1)/s向上取整; padding == 'same', F/s


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),
            layers.Conv2D(48, kernel_size=11, strides=4, activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPool2D(pool_size=3, strides=2)

        ])

        self.flatten = layers.Flatten()
        self.classfier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax'),
        ])

    def call(self, images, **kwargs):
        x = self.features(images)
        x = self.flatten(x)
        x = self.classfier(x)
        return x


def AlexNet_v1(height=224, width=224, num_classes=1000):
    input_img = layers.Input(shape=(height, width, 3), dtype='float32')
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_img)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    predict = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_img, outputs=predict)
    return model
