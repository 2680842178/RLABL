import os
import pickle
from PIL import Image
import cv2
import torch
from collections import deque
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 定义数据增强和转换
# not used
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# 设置超参数
params = {"lr": 0.001, "batch_size": 16, "dropout_prob": 0.5}


# 定义数据集
class ImageBuffer:
    def __init__(self, max_size, shape):  # shape is a tuple, for example: (256, 256, 3)
        self.max_size = max_size
        # self.buffer = np.ndarray(shape=(max_size, *shape))
        self.buffer = deque(maxlen=max_size)

    def add_image(self, image):
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
        image = cv2.resize(image, (256, 256))
        self.buffer.append(image)

    def reset(self):
        self.buffer = []

    def get_images(self, batch_size):
        assert len(self.buffer) >= batch_size
        indices = torch.randperm(len(self.buffer))[:batch_size]
        return np.array([self.buffer[i] for i in indices])

    def __len__(self):
        return len(self.buffer)


# 定义自编码器模型
class ConvAutoencoder(nn.Module):
    def __init__(self, dropout_prob):
        super(ConvAutoencoder, self).__init__()
        self.enc1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x), 0.2)
        x = F.leaky_relu(self.enc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.enc3(x), 0.2)
        x = F.leaky_relu(self.enc4(x), 0.2)
        x = F.leaky_relu(self.dec1(x), 0.2)
        x = F.leaky_relu(self.dec2(x), 0.2)
        x = F.leaky_relu(self.dec3(x), 0.2)
        x = torch.sigmoid(self.dec4(x))
        return x


class MutationModule:
    def __init__(
        self, params=params, batch_size=16, device="cuda", transform=transform
    ):
        self.params = params
        self.device = device
        self.autoencoder = ConvAutoencoder(params["dropout_prob"]).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.autoencoder.parameters(), lr=params["lr"]
        )
        self.kde = None
        self.train_batch_size = batch_size
        self.transform = transform
        self.buffer = ImageBuffer(max_size=256, shape=(256, 256, 3))

    def add_to_buffer(self, img):
        self.buffer.add_image(img)

    def add_batch_to_buffer(self, batch_images):
        if batch_images[0] == self.buffer.max_size:
            self.buffer.reset()
            self.buffer.buffer = batch_images

    def train(self, num_epochs):
        # 训练模型
        for epoch in range(num_epochs):
            self.autoencoder.train()
            running_loss = 0.0
            for _ in range(256 // self.train_batch_size):
                batch_images = self.buffer.get_images(self.train_batch_size)
                # plt.imshow(batch_images[0])
                # plt.show()
                # batch_images = (
                #     torch.tensor(batch_images).to(self.device).permute(0, 3, 1, 2)
                # )
                # batch_images = transforms.ToTensor()(batch_images).to(self.device)
                batch_images = batch_images.astype(np.float32)
                batch_images /= 255.0
                batch_images = (
                    torch.tensor(batch_images).to(self.device).permute(0, 3, 1, 2)
                )

                self.optimizer.zero_grad()
                outputs = self.autoencoder(batch_images)
                loss = self.criterion(outputs, batch_images)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch_images.size(0)
            epoch_loss = running_loss / len(self.buffer)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            # 计算正常数据的重构误差，并拟合正态分布
            self.autoencoder.eval()
            reconstruction_errors = []
            with torch.no_grad():
                for _ in range(256 // self.train_batch_size):
                    batch_images = self.buffer.get_images(self.train_batch_size)
                    # batch_images = (
                    #     torch.tensor(batch_images).to(self.device).permute(0, 3, 1, 2)
                    # )
                    batch_images = batch_images.astype(np.float32)
                    batch_images /= 255.0
                    batch_images = (
                        torch.tensor(batch_images).to(self.device).permute(0, 3, 1, 2)
                    )
                    outputs = self.autoencoder(batch_images)
                    loss = torch.mean((outputs - batch_images) ** 2, dim=[1, 2, 3])
                    reconstruction_errors.extend(loss.cpu().numpy())

            # 拟合正态分布
            # self.mu, self.std = norm.fit(reconstruction_errors)
            self.reconstruction_errors = np.array(reconstruction_errors).reshape(-1, 1)
            self.kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(
                self.reconstruction_errors
            )

        

    def predict(self, img):
        # 在异常数据上进行检测
        with torch.no_grad():
            # print(img.shape)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32)
            img /= 255.0
            img = (
                torch.tensor(img).to(self.device).permute(2, 0, 1).unsqueeze(0)
            )
            outputs = self.autoencoder(img)
            loss = torch.mean((outputs - img) ** 2, dim=[1, 2, 3])
            anomaly_score = loss.cpu().numpy()

        log_density = self.kde.score_samples(anomaly_score.reshape(-1, 1))
        anomaly_probability = np.exp(log_density)

        return anomaly_score, anomaly_probability

    @classmethod
    def load_model(
        cls,
        model_path,
        params=params,
        batch_size=16,
        device="cuda",
        transform=transform,
    ):
        new_instance = cls(
            params=params, batch_size=batch_size, device=device, transform=transform
        )
        new_instance.autoencoder.load_state_dict(torch.load(model_path))
        kde_path = model_path + "_kde"
        with open(kde_path, "rb") as f:
            new_instance.kde = pickle.load(f)
            print(new_instance.kde)
        return new_instance

    def save_model(self, model_path):
        torch.save(self.autoencoder.state_dict(), model_path)
        kde_path = model_path + "_kde"
        with open(kde_path, "wb") as f:
            pickle.dump(self.kde, f)
        print("Mutation Module saved to ", model_path)
