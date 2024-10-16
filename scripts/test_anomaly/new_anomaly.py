import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# 定义数据增强和转换
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),  # 增加随机水平翻转
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),  # 增加颜色抖动
        transforms.RandomRotation(15),  # 增加随机旋转
        transforms.ToTensor(),
    ]
)


# 定义数据集
class ImageDataset(Dataset):
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.image_paths.append(img_path)
                self.labels.append(label)
            else:
                print(f"Skipped {img_path} (not an image)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义自编码器模型
class ConvAutoencoder(nn.Module):
    def __init__(self, dropout_prob):
        super(ConvAutoencoder, self).__init__()
        self.enc1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc5 = nn.Conv2d(
            256, 512, kernel_size=4, stride=2, padding=1
        )  # 增加编码层
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # 增加解码层

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x), 0.2)
        x = F.leaky_relu(self.enc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.enc3(x), 0.2)
        x = F.leaky_relu(self.enc4(x), 0.2)
        x = F.leaky_relu(self.enc5(x), 0.2)  # 新增层
        x = F.leaky_relu(self.dec1(x), 0.2)
        x = F.leaky_relu(self.dec2(x), 0.2)
        x = F.leaky_relu(self.dec3(x), 0.2)
        x = F.leaky_relu(self.dec4(x), 0.2)
        x = torch.sigmoid(self.dec5(x))  # 新增层
        return x


class MutationModule:
    def __init__(self, params, normal_dir, anomaly_dir, device, transform):
        self.params = params
        self.normal_dir = normal_dir
        self.anomaly_dir = anomaly_dir
        self.device = device
        self.autoencoder = ConvAutoencoder(params["dropout_prob"]).to(self.device)
        self.kde = None
        self.transform = transform
        self.reconstruction_errors = None  # 添加属性

    def add_to_buffer(self, img):
        img.save(
            f"self.normal_dir/{len([f for f in os.listdir(normal_dir)])}.bmp",
            format="BMP",
        )

    def train(self, num_epochs):
        normal_dataset = ImageDataset(
            self.normal_dir, label=0, transform=self.transform
        )
        train_loader = DataLoader(
            normal_dataset, batch_size=params["batch_size"], shuffle=True
        )

        # 初始化模型和优化器
        autoencoder = ConvAutoencoder(params["dropout_prob"]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=self.params["lr"])

        # 训练模型
        for epoch in range(num_epochs):
            autoencoder.train()
            running_loss = 0.0
            for images, _ in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                outputs = autoencoder(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # 计算正常数据的重构误差，并使用核密度估计拟合分布
            autoencoder.eval()
            reconstruction_errors = []
            with torch.no_grad():
                for images, _ in train_loader:
                    images = images.to(device)
                    outputs = autoencoder(images)
                    loss = torch.mean((outputs - images) ** 2, dim=[1, 2, 3])
                    print(loss)
                    reconstruction_errors.extend(loss.cpu().numpy())

            # 使用核密度估计拟合分布
            self.reconstruction_errors = np.array(reconstruction_errors).reshape(
                -1, 1
            )  # 保存重构误差
            self.kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(
                self.reconstruction_errors
            )

    def predict(self, img):
        # 在异常数据上进行检测
        with torch.no_grad():
            img = img.to(device)
            outputs = self.autoencoder(img)
            loss = torch.mean((outputs - img) ** 2, dim=[1, 2, 3])
            anomaly_score = loss.cpu().numpy().reshape(-1, 1)

        # 计算异常概率
        log_density = self.kde.score_samples(anomaly_score)
        anomaly_probability = np.exp(log_density)

        return anomaly_score, anomaly_probability


# 设置超参数
params = {"lr": 0.001, "batch_size": 16, "dropout_prob": 0.5}

normal_dir = "/home/sporeking/Codes/RLABL1/scripts/test_anomaly/dataset/0"
anomaly_dir = "/home/sporeking/Codes/RLABL1/scripts/test_anomaly/dataset/1"

# 初始化数据集和数据加载器
normal_dataset = ImageDataset(normal_dir, label=0, transform=transform)
anomaly_dataset = ImageDataset(anomaly_dir, label=1, transform=transform)

if len(normal_dataset) > 0 and len(anomaly_dataset) > 0:
    train_loader = DataLoader(
        normal_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=False)

    # 初始化模型和优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mutation_module = MutationModule(params, normal_dir, anomaly_dir, device, transform)
    mutation_module.train(num_epochs=10)

    # 在异常数据上进行检测
    anomaly_scores = []
    anomaly_probabilities = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            score, prob = mutation_module.predict(images)
            anomaly_scores.append(score[0])
            anomaly_probabilities.append(prob[0])

    # 输出每个样本的异常概率
    for idx, (score, prob) in enumerate(zip(anomaly_scores, anomaly_probabilities)):
        print(
            f"Image {idx + 1}: Reconstruction Error = {score[0]:.4f}, Anomaly Probability = {prob:.4f}"
        )

    train_loader = DataLoader(
        normal_dataset, batch_size=1, shuffle=True
    )
    normal_scores = []
    normal_probabilities = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            score, prob = mutation_module.predict(images)
            normal_scores.append(score[0])
            normal_probabilities.append(prob[0])

    for idx, (score, prob) in enumerate(zip(normal_scores, normal_probabilities)):
        print(
            f"Normal Image {idx + 1}: Reconstruction Error = {score[0]:.4f}, Anomaly Probability = {prob:.4f}"
        )

    # 可视化重构误差的分布
    plt.figure(figsize=(10, 5))
    plt.hist(
        mutation_module.reconstruction_errors,
        bins=50,
        alpha=0.6,
        color="g",
        label="Normal Data",
    )
    plt.hist(
        [s[0] for s in anomaly_scores],
        bins=50,
        alpha=0.6,
        color="r",
        label="Anomaly Data",
    )
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    output_dir = "./output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "reconstruction_error_distribution.png"))

else:
    print("No images found in one or both datasets.")
