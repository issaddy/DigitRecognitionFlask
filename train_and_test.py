# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# 超参数设置
batch_size = 200
learning_rate = 0.01
epochs = 40
# 数据集加载和预处理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # 加载数据集
train_images = np.expand_dims(train_images, axis=-1) / 255.0  # 扩展维度并归一化
test_images = np.expand_dims(test_images, axis=-1) / 255.0
train_labels = to_categorical(train_labels, num_classes=10)  # one-hot编码
test_labels = to_categorical(test_labels, num_classes=10)
# shuffle()方法打乱数据集，batch()方法将数据集切分为batch_size大小的数据块
train_db = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(10000).batch(batch_size)


def LeNet5():
    """
    LeNet-5模型
    :return:
    """
    model = tf.keras.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=5, strides=1, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def evaluate(model, loader):
    """
    评估模型在验证集上的准确率
    :param model: 模型
    :param loader: 验证集
    :return:
    """
    correct = tf.keras.metrics.CategoricalAccuracy()
    for x, y in loader:
        logits = model(x, training=False)  # 前向传播
        correct.update_state(y, logits)  # 更新正确预测的样本数
    accuracy = correct.result()  # 计算准确率
    return accuracy.numpy()  # 返回准确率


def plot_curve(data):
    plt.figure()
    plt.plot(range(len(data)), data, color='blue')  # 画出迭代次数与loss的关系
    plt.legend(['value'], loc='upper right')  # 设置图例及其位置
    plt.xlabel('step')  # 设置x轴标签
    plt.ylabel('value')  # 设置y轴标签
    plt.show()


def main():
    model = LeNet5()
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.01)  # 优化器
    criterion = losses.CategoricalCrossentropy()  # 损失函数
    model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])  # 编译模型
    train_loss = []  # 记录训练集loss
    best_acc, best_epoch = 0, 0  # 记录最佳准确率和最佳epoch
    no_improvement_count = 0  # 记录验证集准确率连续没有提升的次数
    early_stopping_patience = 5  # 提前终止训练的次数
    for epoch in range(epochs):  # 训练epochs轮
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)  # 前向传播
                loss_value = criterion(y, logits)  # 计算损失
            grads = tape.gradient(loss_value, model.trainable_variables)  # 计算梯度
            optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新参数
            if step % 100 == 0:
                print('Epoch: {} Step: {} Loss: {:.4f}'.format(epoch, step, loss_value.numpy()))
                train_loss.append(loss_value.numpy())  # 记录训练集loss
        if epoch % 1 == 0:
            val_acc = evaluate(model, test_db)  # 评估模型在验证集上的准确率
            print('Validation Accuracy: {:.4f} Epoch: {}'.format(val_acc, epoch))
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                model.save('best_model.h5')  # 保存模型
                model.save_weights('best_weights.h5')  # 保存模型参数
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            # 如果验证集准确率连续没有提升early_stopping_patience次，则提前终止训练
            if no_improvement_count >= early_stopping_patience:
                print("Early stopping triggered. No improvement in validation accuracy for {} epochs.".format(
                    early_stopping_patience))
                break
    print('Best Accuracy: {:.4f} Best Epoch: {}'.format(best_acc, best_epoch))
    plot_curve(train_loss)


if __name__ == '__main__':
    main()
