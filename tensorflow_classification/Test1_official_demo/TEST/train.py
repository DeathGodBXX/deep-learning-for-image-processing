from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import MyModel
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    mnist = tf.keras.datasets.mnist

    # 下载数据集(范围在0-255的灰度图片和labels, 无channel通道)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 增加channel通道
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # 创建数据生成器，将数据切割并拼装成元组或array,每次随机读入内存中32个grayscale(灰度图片)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # 调用模型
    model = MyModel()

    # 定义交叉熵损失
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # 定义Adam优化器
    optimizer = tf.keras.optimizers.Adam()

    # 计算train损失均值和正确率
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # 计算test损失均值和正确率
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 训练数据,计算loss,梯度求导,反向传播,损失均值和正确率
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            """上下文管理器只监控当前作用域下的梯度"""
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # 测试数据,计算测试loss，正确率
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCH = 5
    for epoch in range(EPOCH):
        """每次epoch开始时都清除历史指标信息"""
        train_loss.reset_states()
        train_accuracy.reset_states()

        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = 'Epoch {},Loss {},Accuracy {};test Loss {}, test Accuracy {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))

    model.save_weights('weight.h5', save_format='h5')
















































if __name__ == '__main__':
    main()


