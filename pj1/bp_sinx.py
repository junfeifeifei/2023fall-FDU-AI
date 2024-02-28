import numpy as np
import matplotlib.pyplot as plt

# 创建数据
x = np.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)
y = np.sin(x)
np.set_printoptions(threshold=np.inf)

# 定义每层的节点数和学习率等信息
input_size = 1
hidden_size1 = 128
hidden_size2 = 32
output_size = 1
learning_rate = 0.0001
epochs = 50000


# Tanh激活函数
def tanh(x):
    return np.tanh(x)


# 定义隐藏层
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
        return tanh(hidden_input)

    def backward(self, outputs, input_loss):
        self.dw = np.dot(outputs.T, input_loss)
        self.db = np.sum(input_loss, axis=0, keepdims=True)
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db
        return np.dot(input_loss, self.w.T) * (1 - outputs ** 2)  # Derivative of tanh


# 创建隐藏层实例
h1 = HiddenLayer(input_size, hidden_size1)
h2 = HiddenLayer(hidden_size1, hidden_size2)
h3 = HiddenLayer(hidden_size2, output_size)

epoch_array = []
accuracy_array = []
# 训练网络
for epoch in range(epochs):
    # 先正向传播
    x1 = h1.forward(x.reshape(-1, 1))
    x2 = h2.forward(x1)
    predicted_output = h3.forward(x2)

    loss = 0.5 * np.mean(np.square(predicted_output - y))
    error = np.mean(np.absolute(predicted_output - y))
    epoch_array.append(epoch)
    accuracy_array.append(1 - error)
    if epoch % 1000 == 0:
        print(epoch, ':', 'loss:', loss, 'Accuracy:', error)

    # 然后进行反向传播
    d_loss = predicted_output - y.reshape(-1, 1)
    y1 = h3.backward(x2, d_loss)
    y2 = h2.backward(x1, y1)
    y3 = h1.backward(x, y2)

# 利用我们训练好的模型来进行输出
x1 = h1.forward(x.reshape(-1, 1))
x2 = h2.forward(x1)
predicted_ans = h3.forward(x2)

average_error = np.mean(np.absolute(predicted_ans - y))
print('误差为:', average_error)
# 画图
plt.figure(figsize=(8, 6))
plt.title('fit')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.plot(x, y, label='sin(x)')
plt.plot(x, predicted_ans, label='Fitted Curve', linestyle='dashed')
plt.legend()
plt.show()
plt.figure(1)
plt.title('accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(epoch_array, accuracy_array)
plt.show()
