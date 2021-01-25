from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v2, AlexNet_v1
import tensorflow as tf
import json, os


def main():
    # 获取训练集和验证集目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
    image_path = os.path.join(data_root, 'data_set', 'flower_data')
    train_dir = os.path.join(image_path, 'train')
    val_dir = os.path.join(image_path, 'val')

    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(val_dir), "cannot find {}".format(val_dir)

    # 创建用于保存训练权重的目录
    if not os.path.exists('save_weight'):
        os.mkdir('save_weight')

    # 定义图片尺寸,batch,epochs
    img_height = 224
    img_width = 224
    batch = 32
    epochs = 10

    # 训练集和验证集的生成器
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)

    validation_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode="categorical")
    validation_data_gen = validation_image_generator.flow_from_directory(directory=val_dir,
                                                                         batch_size=batch,
                                                                         target_size=(img_height, img_width),
                                                                         class_mode="categorical")
    # 获取训练集个数，类别并把key和val调换位置
    total_train = train_data_gen.n
    class_indices = train_data_gen.class_indices
    total_val = validation_data_gen.n
    print("using {} images for training,{} images for validation.".format(
        total_train, total_val
    ))

    # 写入json文件中
    with open('class_indices.json', 'w') as fp:
        inverse_dict = dict((val, key) for key, val in class_indices.items())
        json.dump(inverse_dict, fp, indent=4)

    # # 打印sample的images,labels
    # sample_training_images, sample_training_labels = next(train_data_gen)
    # def plotImages(images_arr):
    #     plt.figure()
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    #     axes = axes.flatten()
    #     for img, ax in zip(images_arr, axes):
    #         ax.imshow(img)
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    # plotImages(sample_training_images[:5])

    # 实例化模型
    # model = AlexNet_v2(num_classes=5)
    # model.build((batch, img_height, img_width, 3))  # using subclass API
    model = AlexNet_v1(height=img_height, width=img_width, num_classes=5)
    model.summary()

    # # 使用keras高层级api，训练模型
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # model的最后已经调用softmax，这里不再调用softmax
    #     metrics=["accuracy"]
    # )
    #
    # # 保存训练之后的参数
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weight/myAlex_v2.h5',
    #                                                 save_best_only=True,
    #                                                 save_weights_only=True,
    #                                                 monitor='val_loss')]
    #
    # # 使用fit训练模型(tensorflow 2.1 推荐)
    # history = model.fit(x=train_data_gen,
    #                     steps_per_epoch=total_train // batch,
    #                     epochs=epochs,
    #                     validation_data=validation_data_gen,
    #                     validation_steps=total_val // batch,
    #                     callbacks=callbacks)
    #
    # # 画出损失和正确率曲线
    # history_dict = history.history
    # train_loss = history_dict['loss']
    # train_accuracy = history_dict['accuracy']
    # val_loss = history_dict['val_loss']
    # val_accuracy = history_dict['val_accuracy']
    #
    # # figrue1
    # plt.figure()
    # plt.plot(range(epochs), train_loss, label='train_loss')
    # plt.plot(range(epochs), val_loss, label='val_loss')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    #
    # # figure2
    # plt.figure()
    # plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    # plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.show()

    # 使用low-level api train model
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # 定义train函数, train dataset, calculate loss_object, loss and accuracy
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # 定义test函数，test dataset,calculate loss and accuracy,不需要使用优化器和梯度等信息
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_object(labels, predictions)
        test_loss(loss)
        test_accuracy(labels, predictions)

    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step in range(total_train // batch):
            images, labels = next(train_data_gen)
            train_step(images, labels)

        for step in range(total_val // batch):
            images, labels = next(validation_data_gen)
            test_step(images, labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}; Test loss {}, Accuracy {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        if test_loss.result() < best_test_loss:
            model.save_weights('./save_weight/myAlex.ckpt', save_format='tf')


if __name__ == '__main__':
    main()
