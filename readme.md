# 手写数字识别项目

该项目是一个基于深度学习和图像处理技术的手写数字识别系统。通过该系统，用户可以上传手写数字图像，系统将自动对图像进行数字分割，并使用训练好的模型对分割出的单个数字进行识别。最终，系统将识别结果返回给用户。

## 功能特点

- 提供一个Web界面，方便用户上传手写数字图像进行识别。
- 使用LeNet-5模型和MNIST数据集（训练集6万张、测试集1万张）进行手写数字分类训练。
- 利用图像处理技术实现数字分割，通过几何计算和轮廓分析等技术，准确地将数字从输入图像中分割出来。
- 实现了一个简单而有效的手写数字识别系统，准确率较高。

## 环境要求

- Python 3.7
- TensorFlow 2.6
- OpenCV 4.6
- Flask 2.2

## 安装

1. 克隆该仓库：`git clonehttps://github.com/issaddy/DigitRecognitionFlask.git`
2. 安装所需依赖：`pip install -r requirements.txt`
3. 训练模型并保存权重：`python train_and_test.py`
4. 启动Web应用：`python app.py`
5. 在Web浏览器中访问 `http://localhost:5000` 来使用应用程序。

## 使用

1. 在Web浏览器中打开应用程序。
2. 点击 "上传图像" 按钮选择要识别的手写数字图像。
3. 上传完成后，应用程序将显示分割后的数字以及识别出的数字值。
4. 可以重复此过程以识别其他图像，或点击 "重置" 按钮重新开始。

## 项目结构

```
├── app.py            # Flask应用入口文件
├── extract_nums.py   # 数字分割模块
├── train_and_test.py # 训练模型模块
├── best_model.h5     # 训练后保存的最佳模型
├── best_weights.h5   # 训练后保存的最佳模型权重
├── readme.md         # 项目说明
├── requirements.txt  # 项目依赖
├── test_img          # 自己手写数字用来测试的图片目录
├── get_num_img       # 分割出的数字保存目录
├── static            # 静态资源目录
└── templates         # HTML模板目录
    └── index.html    # 主页模板
```

