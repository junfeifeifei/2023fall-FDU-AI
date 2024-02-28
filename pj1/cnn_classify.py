# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import os
import matplotlib.pyplot as plt


# 定义一个LeNet5的神经网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        # 定义全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 数据处理类
class DataProcessor:
    def __init__(self):
        # 初始化数据数组
        self.x_all = np.empty((0, 784))
        self.y_all = np.empty((0, 12))
        self.x_learn = np.empty((0, 784))
        self.x_test = np.empty((0, 784))
        # 创建标签数组
        self.y_arrays = [np.eye(12)[i] for i in range(12)]
        self.y_learn = np.empty((0, 12))
        self.y_test = np.empty((0, 12))

    # 遍历文件夹并加载数据
    def traverse_folder(self, folder_name, num):
        for root, dirs, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                img = Image.open(file_path)
                img_array = np.array(img).flatten()
                self.x_all = np.vstack((self.x_all, img_array))
                self.y_all = np.vstack((self.y_all, self.y_arrays[num]))

    # 获取并整理数据
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


# 创建数据处理实例
data_processor = DataProcessor()
data_processor.get_data()
x_learn = data_processor.x_learn
x_test = data_processor.x_test
y_learn = data_processor.y_learn
y_test = data_processor.y_test

# 设置超参数
learning_rate = 0.001
batch_size = 64
num_epochs = 500

# 创建用于训练和测试的数据加载器
x_learn_tensor = torch.from_numpy(x_learn).view(-1, 1, 28, 28).float()
y_learn_tensor = torch.from_numpy(y_learn).float()
x_test_tensor = torch.from_numpy(x_test).view(-1, 1, 28, 28).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(x_learn_tensor, y_learn_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型和优化器
model = LeNet5()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 初始化准确度列表
accuracy_values = []

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, torch.max(batch_y, 1)[1])
        loss.backward()
        optimizer.step()

    # 在测试集上计算准确度
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == torch.max(batch_y, 1)[1]).sum().item()
        test_accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Test Accuracy: {test_accuracy:.2f}%')

    # 将测试准确度添加到列表中
    accuracy_values.append(test_accuracy)

# 绘制准确度与迭代次数的曲线
plt.plot(range(1, num_epochs + 1), accuracy_values)
plt.xlabel('Epoch')
plt.ylabel('Test accuracy (%)')
plt.title('Test accuracy vs. Epoch')
plt.grid(True)
plt.show()
