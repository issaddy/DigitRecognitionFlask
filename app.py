# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from extract_nums import digital_segmentation
from flask import Flask, request, jsonify, render_template

# 创建一个Flask对象，传递__name__参数，Flask用这个参数决定程序的根目录，以便以后能找到相对于程序根目录的资源文件位置
app = Flask(__name__)


@app.route('/')
def index():
    """
    主页
    :return:
    """
    return render_template("index.html")  # 调用render_template函数，传入html文件参数


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    识别手写数字
    :return:
    """
    image_data = request.files['image'].read()  # 读取图片数据
    nparr = np.frombuffer(image_data, np.uint8)  # 将二进制数据转换为ndarray
    src = cv.imdecode(nparr, cv.IMREAD_COLOR)  # 读取图片
    number_images, min_areas_idx = digital_segmentation(src)  # 数字分割
    model = models.load_model('best_model.h5')  # 加载训练好的模型
    # model = LeNet5()  # 创建模型
    # model.build(input_shape=(None, 28, 28, 1))  # 输入数据的形状
    # model.load_weights('best_weights.h5')  # 加载训练好的模型权重
    print('手写数字为：', end='')
    result = ''
    for i in range(len(number_images)):
        if i == min_areas_idx:
            print('.', end='')
            result += '.'
        else:
            x = cv.imread('./get_num_img/' + str(i) + '.png')
            x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)  # 转换为灰度图
            x = tf.convert_to_tensor(x, dtype=tf.float32)  # 转换为张量
            x = tf.reshape(x, (1, x.shape[0], x.shape[1], 1))  # 调整形状来符合模型输入要求
            x = tf.image.resize(x, (28, 28), method=tf.image.ResizeMethod.AREA)  # 使用INTER_AREA方法调整大小
            logits = model.call(x)  # 调用模型
            pred = tf.argmax(logits, axis=1)  # 代表在[b ， 10]中0-9的一个最大值的索引
            pred = pred.numpy()  # 转换为numpy数组
            print(pred[0], end='')  # 输出预测结果
            result += str(pred[0])
    # cv.waitKey(0)  # 等待用户操作
    # cv.destroyAllWindows()  # 销毁所有窗口
    return jsonify({'result': result})


if __name__ == "__main__":
    """
    项目启动入口，
    运行后要点击控制台中的链接（http://127.0.0.1:5000）访问
    """
    # app.run()
    app.run(host='0.0.0.0', port=5001)
