# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim   #包含各种优化算法
import torchvision     #PyTorch 的视觉库
import torchvision.transforms as transforms  #用于对图像进行预处理和转换模块
from torch.utils.data import DataLoader  #加载数据集
import os
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=12, stride=6),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(len(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.eval()  # 将模型切换为评估模式
    correct = 0
    total = 0
    # with torch.no_grad():  # 在测试过程中不需要计算梯度
    #     for inputs, labels in test_loader:
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)
    #         print(predicted, labels)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # accuracy = correct / total
    # print(f'Accuracy on test set: {accuracy:.4f}')

def train_model1(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(len(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# 创建一个空列表来分别存储图片序列，标签序列
# image_sequence = []
# label= []
# for i in range(5):
#     # 定义图片文件夹路径
#     folder_path = "trace/"+str(i+1)+"/"
#
#     # 获取文件夹中的所有文件
#     image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
#
#     # 按照文件名排序
#     image_files.sort()
#     # 加载并添加图片到序列中
#     for file in image_files:
#         img_path = os.path.join(folder_path, file)
#         img = Image.open(img_path)
#         image_sequence.append(img)
#         label.append(i)
# #print(len(image_sequence))
# #print(len(label))
#     # 现在 image_sequence 就是包含所有图片的序列
#
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.ToTensor(),
# ])
#
# # 训练数据集
# train_dataset = CustomDataset(image_sequence, label, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)#每个批次包含46个样本，shuffle=True,每个epoch开始，对数据进行随机打乱，增加模型的泛化能力。
# #print(len(train_dataset))
#
# # 测试数据集
# test_dataset = CustomDataset(image_sequence, label, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
# # 定义模型、损失函数和优化器
# model = CNN(num_classes=5)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# # 训练模型
# num_epochs = 1000
# train_model(model, train_loader, criterion, optimizer, num_epochs)
# torch.save(model, 'stateNN.pth')

#以下为输出部分

def stateNN(image_data):
    device = torch.device('cuda:0')
    model = torch.load('stateNN.pth')
    model = model.to(device)
    model.eval()
    # preprocess = transforms.Compose([
    #     transforms.Resize((300, 300))
    # ])
    input_tensor = image_data.image[0]
    input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)

    output = model(input_batch)
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
    return predicted

def getModel(image_sequence, label_sequence):
    transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    ])
    model = CNN(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_dataset = CustomDataset(image_sequence, label_sequence, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    num_epochs = 1000
    train_model1(model, train_loader, criterion, optimizer, num_epochs)
    return model



