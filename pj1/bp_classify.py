import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


# 数据处理的类
class DataProcessor:
    def __init__(self):
        self.x_all = np.empty((0, 784))
        self.y_all = np.empty((0, 12))
        self.x_learn = np.empty((0, 784))
        self.x_test = np.empty((0, 784))
        self.y_arrays = [np.eye(12)[i] for i in range(12)]
        self.y_learn = np.empty((0, 12))
        self.y_test = np.empty((0, 12))

    def traverse_folder(self, folder_name, num):
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                img_array = np.array(img).flatten()
                self.x_all = np.vstack((self.x_all, img_array))
                self.y_all = np.vstack((self.y_all, self.y_arrays[num]))

    def get_data(self):
        for i in range(12):
            folder_name = './train/' + str(i + 1)
            self.traverse_folder(folder_name, i)
        self.x_all, self.y_all = shuffle(self.x_all, self.y_all)
        train_ratio = 0.75
        split_point = int(len(self.x_all) * train_ratio)
        self.x_learn = self.x_all[:split_point]
        self.y_learn = self.y_all[:split_point]
        self.x_test = self.x_all[split_point:]
        self.y_test = self.y_all[split_point:]


data_processor = DataProcessor()
data_processor.get_data()
x_learn = data_processor.x_learn
x_test = data_processor.x_test
y_learn = data_processor.y_learn
y_test = data_processor.y_test

actual_labels = np.argmax(y_test, axis=1)


# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止指数过大，减去最大值
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class HiddenLayer:
    def __init__(self, input, output):
        self.input_size = input
        self.output_size = output
        self.w = np.random.normal(0.0, pow(self.output_size, -0.5), (self.input_size, self.output_size))
        self.b = np.zeros((1, self.output_size))

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        self.inputs = inputs
        hidden_input = np.dot(self.inputs, self.w) + self.b
        return sigmoid(hidden_input)

    def backward(self, outputs, input_loss):
        self.dw = np.dot(outputs.T, input_loss)
        self.db = np.sum(input_loss, axis=0, keepdims=True)
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db
        return np.dot(input_loss, self.w.T) * outputs * (1 - outputs)


np.random.seed(50)
# 定义神经网络参数
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 64
output_size = 12
learning_rate = 0.005
epochs = 5000

batch_size = 32

epoch_array = []
accuracy_array = []

h1 = HiddenLayer(input_size, hidden_size1)
h2 = HiddenLayer(hidden_size1, hidden_size2)
h3 = HiddenLayer(hidden_size2, hidden_size3)
h4 = HiddenLayer(hidden_size3, output_size)


def test_network():
    global w1, w2, w3, w4, b1, b2, b3, b4, actual_labels
    # 测试神经网络
    x = h1.forward(x_test)
    x = h2.forward(x)
    x = h3.forward(x)
    predicted_output_test = h4.forward(x)

    # 使用 Softmax 处理输出
    predicted_output_test = softmax(predicted_output_test)
    # 计算预测类别
    predicted_labels = np.argmax(predicted_output_test, axis=1)
    actual_labels = np.argmax(y_test, axis=1)

    # 计算准确率
    accuracy = np.mean(predicted_labels == actual_labels)
    accuracy_array.append(accuracy * 100)
    print(f"准确率: {accuracy * 100:.2f}%")


# 训练神经网络
for epoch in range(epochs):
    for i in range(0, len(x_learn), batch_size):
        x_batch = x_learn[i:i + batch_size]
        y_batch = y_learn[i:i + batch_size]
        # 前向传播
        x1 = h1.forward(x_batch)
        x2 = h2.forward(x1)
        x3 = h3.forward(x2)
        predicted_output = h4.forward(x3)

        # 计算交叉熵损失
        predicted_output = np.clip(predicted_output, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_batch * np.log(predicted_output))

        # 反向传播
        d_loss = (predicted_output - y_batch) / len(y_batch)

        y1 = h4.backward(x3, d_loss)
        y2 = h3.backward(x2, y1)
        y3 = h2.backward(x1, y2)
        y4 = h1.backward(x_batch, y3)
        if epoch % 10 == 0 and i == 0:
            print(epoch, ':', loss)
            epoch_array.append(epoch)
            test_network()

plt.figure(figsize=(8, 6))
plt.scatter(epoch_array, accuracy_array, color='blue')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
