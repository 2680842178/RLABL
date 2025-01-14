import time
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from abc import abstractmethod
from .process import contrast_ssim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader, Dataset

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
def train_vae(vae, dataloader, optimizer, epochs, device):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(images)

            # 计算损失
            recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


class AnomalyDetector:
    def __init__(self):
        pass

    @abstractmethod
    def add_normal_samples(self, image) -> bool:
        pass

    @abstractmethod
    def detect_anomaly(self, image) -> bool:
        pass

    @classmethod
    def load_model(cls, model_path):
        pass

    def save_model(self, model_path):
        pass

class BoundaryDetectorSSIM(AnomalyDetector):
    def __init__(self, saved_images_folder):
        super().__init__()
        self.saved_images_folder = saved_images_folder
        if not os.path.exists(self.saved_images_folder):
            os.makedirs(self.saved_images_folder)
        self.saved_images = self.load_saved_images()
        self.contrast_value = 0.5

    def load_saved_images(self):
        saved_images = []
        for filename in os.listdir(self.saved_images_folder):
            img = cv2.imread(os.path.join(self.saved_images_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                saved_images.append((filename, img, hist))
        return saved_images

    def add_normal_samples(self, image):
        if self.is_known_image(image, add_to_buffer=True):
            return False
        return True

    def contrast(self, img1, img2, return_bool=False):
        if not return_bool:
            return contrast_ssim(img1, img2)
        else:
            return contrast_ssim(img1, img2) > self.contrast_value

    def preprocess_RGBimage(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray_image[y:y + h, x:x + w]
            processed.append(roi)
        
        return gray_image, processed

    def is_known_roi(self, roi, add_to_buffer=False):
        is_anomaly = False
        # processed_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        # processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()

        similar = 0
        # print(len(self.saved_images))
        for saved_image_name, saved_image, saved_hist in self.saved_images:
            # saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
            # correlation = cv2.compareHist(processed_hist, saved_hist, cv2.HISTCMP_CORREL)
            # similar = max(similar, abs(correlation))
            similar = max(similar, contrast_ssim(roi, saved_image))
            # print(similar)
            if similar > 0.6:
                break
        if similar < 0.6 and similar >= 0:
            is_anomaly = True
            if add_to_buffer:
                # print("Anomaly detected, saving image")
                self.add_new_image(roi, is_processed=True)

        return is_anomaly

    def is_known_image(self, processed, add_to_buffer=False):
        # _, processed = self.preprocess_RGBimage(image)
        is_anomaly = False
        for processed_image in processed:
            processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
            processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()

            similar = 0
            # print(len(self.saved_images))
            for saved_image_name, saved_image, saved_hist in self.saved_images:
                # saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
                # correlation = cv2.compareHist(processed_hist, saved_hist, cv2.HISTCMP_CORREL)
                # similar = max(similar, abs(correlation))
                similar = max(similar, contrast_ssim(processed_image, saved_image)) 
                # print(similar)
                if similar > 0.7:
                    break
            if similar < 0.7 and similar >= 0:
                is_anomaly = True
                if add_to_buffer:
                    # print("Anomaly detected, saving image")
                    self.add_new_image(processed_image, is_processed=True)
            # else:
            #     print("No anomaly detected, dont save.")
            # elif similar >= 1:
            #     return True
        if is_anomaly:
            return False
        return True

    def add_new_image(self, image, is_processed=False):
        if not is_processed:
            processed_image, _ = self.preprocess_RGBimage(image)
        else:
            processed_image = image
        saved_image_count = len(self.saved_images)
        new_filename = f"{saved_image_count}.bmp"
        save_path = os.path.join(self.saved_images_folder, new_filename)
        cv2.imwrite(save_path, processed_image)
        processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()
        self.saved_images.append((new_filename, processed_image, processed_hist))
        print(f"Image saved: {new_filename}")

    def detect_anomaly(self, image):
        return not self.is_known_image(image, add_to_buffer=False)



class BoundaryDetector(AnomalyDetector):
    def __init__(self, saved_images_folder):
        super().__init__()
        self.saved_images_folder = saved_images_folder
        if not os.path.exists(self.saved_images_folder):
            os.makedirs(self.saved_images_folder)
        self.saved_images = self.load_saved_images()
        self.contrast_value = 0.99999

    def load_saved_images(self):
        saved_images = []
        for filename in os.listdir(self.saved_images_folder):
            img = cv2.imread(os.path.join(self.saved_images_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                saved_images.append((filename, img, hist))
        return saved_images

    def add_normal_samples(self, image):
        if self.is_known_image(image, add_to_buffer=True):
            return False
        return True

    def contrast(self, img1, img2, return_bool=False):
        if not return_bool:
            return contrast_ssim(img1, img2)
        else:
            return contrast_ssim(img1, img2) > self.contrast_value

    def preprocess_RGBimage(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray_image[y:y + h, x:x + w]
            processed.append(roi)
        
        return gray_image, processed

    def is_known_roi(self, roi, add_to_buffer=False):
        is_anomaly = False
        processed_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()

        similar = 0
        # print(len(self.saved_images))
        for saved_image_name, saved_image, saved_hist in self.saved_images:
            saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
            correlation = cv2.compareHist(processed_hist, saved_hist, cv2.HISTCMP_CORREL)
            similar = max(similar, abs(correlation))
            # similar = max(similar, contrast_ssim(roi, saved_image))
            # print(similar)
            if similar > 0.99999:
                break
        if similar < 0.99999 and similar >= 0:
            is_anomaly = True
            if add_to_buffer:
                # print("Anomaly detected, saving image")
                self.add_new_image(roi, is_processed=True)

        return is_anomaly

    def is_known_image(self, processed, add_to_buffer=False):
        # _, processed = self.preprocess_RGBimage(image)
        is_anomaly = False
        for processed_image in processed:
            processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
            processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()

            similar = 0
            # print(len(self.saved_images))
            for saved_image_name, saved_image, saved_hist in self.saved_images:
                saved_hist = cv2.normalize(saved_hist, saved_hist).flatten()
                correlation = cv2.compareHist(processed_hist, saved_hist, cv2.HISTCMP_CORREL)
                similar = max(similar, abs(correlation))
                # similar = max(similar, contrast_ssim(processed_image, saved_image)) 
                # print(similar)
                if similar > 0.99999:
                    break
            if similar < 0.99999 and similar >= 0:
                is_anomaly = True
                if add_to_buffer:
                    # print("Anomaly detected, saving image")
                    self.add_new_image(processed_image, is_processed=True)
            # else:
            #     print("No anomaly detected, dont save.")
            # elif similar >= 1:
            #     return True
        if is_anomaly:
            return False
        return True

    def add_new_image(self, image, is_processed=False):
        if not is_processed:
            processed_image, _ = self.preprocess_RGBimage(image)
        else:
            processed_image = image
        saved_image_count = len(self.saved_images)
        new_filename = f"{saved_image_count}.bmp"
        save_path = os.path.join(self.saved_images_folder, new_filename)
        cv2.imwrite(save_path, processed_image)
        processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        processed_hist = cv2.normalize(processed_hist, processed_hist).flatten()
        self.saved_images.append((new_filename, processed_image, processed_hist))
        print(f"Image saved: {new_filename}")

    def detect_anomaly(self, image):
        return not self.is_known_image(image, add_to_buffer=False)

class ImageDataset(Dataset):
    def __init__(self, is_list=True, image_list=None, image_dir=None, target_size=(32, 32)):
        if is_list == True:
            self.images = image_list
        else:
            self.image_dir = image_dir
            self.file_names = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
            self.images = []
            for file_name in self.file_names:
                image_path = os.path.join(image_dir, file_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.images.append(image)
        self.processed_images = []
        for image in self.images:
            processed_image = self.preprocess_image(image, target_size)
            self.processed_images.append(processed_image)

        self.target_size = target_size

        self.processed_images = np.array(self.processed_images, dtype=np.float32) / 255.0  # 归一化
        self.processed_images = np.expand_dims(self.processed_images, axis=1)  # 添加通道维度

    def preprocess_image(self, image, target_size=(32, 32)):
        _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
        image = contour_image

        # 图像缩放和填充
        h, w = image.shape
        if h > target_size[0] or w > target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            h, w = target_size
        padded_image = np.zeros(target_size, dtype=np.uint8)
        start_y = (target_size[0] - h) // 2
        start_x = (target_size[1] - w) // 2
        padded_image[start_y:start_y + h, start_x:start_x + w] = image
        return padded_image

    def __len__(self):
        return len(self.processed_images)

    def __getitem__(self, idx):
        return self.processed_images[idx]

class ClusterAnomalyDetector:
    def __init__(self):
        # self.kmeans = None
        self.contrast_value = 0.5
        self.inited = False

        self.latent_dim = 16
        self.clu_num = 2
        self.batch_size = 16
        self.epochs = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae = VAE(self.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters())

        self.scaler = StandardScaler()
        self.iforest = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)

    def add_samples(self, image_list):
        self.inited = True
        self.dataset = ImageDataset(is_list=True, image_list=image_list)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self._train_vae(image_list)
        self.vae.eval()
        images = torch.tensor(self.dataset.processed_images).to(self.device)
        with torch.no_grad():
            mu, _ = self.vae.encode(images)
            z = mu.cpu().numpy()

        original_sizes = np.array([self.dataset.images[i].shape for i in range(len(self.dataset.images))])
        original_sizes = original_sizes.astype(np.float32)

        # # 合并特征
        features = np.hstack((z, original_sizes))
        features = self.scaler.fit_transform(features)
        
        self.iforest.fit(features)
        y_pred = self.iforest.decision_function(features)
        anomaly_image = self.dataset.images[np.argmin(y_pred)]
        self.min_anomaly_decision = np.min(y_pred)
        
        # for i in range(len(self.dataset.images)):
        #     cv2.imwrite(f"taxi_mutations/taxi-mutation/{y_pred[i]}-{time.time()}.bmp", self.dataset.images[i]) 
        return anomaly_image
        
        # KMeans聚类
        # kmeans = KMeans(n_clusters=self.clu_num, init="k-means++", random_state=0)
        # labels = kmeans.fit_predict(features)
        # print(labels)
        # unique_labels, count = np.unique(labels, return_counts=True)
        # self.anomaly_class = unique_labels[np.argmin(count)]
        
        # anomaly_features = features[labels == self.anomaly_class]
        # centroid = np.mean(anomaly_features, axis=0)
        # distances = np.linalg.norm(features - centroid, axis=1)
        # representative_idx = np.argmin(distances)
        # return self.dataset.images[representative_idx]
        
        
    def _train_vae(self, images):
        self.vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for images in self.dataloader:
                images = images.to(self.device)
                self.optimizer.zero_grad()
                reconstructed, mu, logvar = self.vae(images)

                # 计算损失
                recon_loss = nn.functional.mse_loss(reconstructed, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

    def preprocess_image(self, image, target_size=(32, 32)):
        _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
        image = contour_image

        # 图像缩放和填充
        h, w = image.shape
        if h > target_size[0] or w > target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            h, w = target_size
        padded_image = np.zeros(target_size, dtype=np.uint8)
        start_y = (target_size[0] - h) // 2
        start_x = (target_size[1] - w) // 2
        padded_image[start_y:start_y + h, start_x:start_x + w] = image
        return padded_image
    
    # def contrast(self, img1, img2):
    #     feature1 = self.extract_features(img1)
    #     feature2 = self.extract_features(img2)
    #     # lebel1 = self.kmeans.predict([feature1])[0]
    #     # lebel2 = self.kmeans.predict([feature2])[0]
    #     # similarity = 1 if lebel1 == lebel2 else 0
    #     return similarity
    def contrast(self, img1, img2, return_bool=False):
        if not return_bool:
            return contrast_ssim(img1, img2)
        else:
            return contrast_ssim(img1, img2) > self.contrast_value

    def extract_features(self, image):# 初始化异常检测器，输入一个图像列表，内部处理并初始化KMeans模型。
        image_shape = image.shape
        image_shape = np.array(image_shape, dtype=np.float32)
        image_shape = np.array([image_shape])
        image_shape = image_shape.astype(np.float32)
        # original_sizes = np.array([self.dataset.images[i].shape for i in range(len(self.dataset.images))])
        # original_sizes = original_sizes.astype(np.float32)
        processed_image = self.preprocess_image(image)
        processed_image = [processed_image]
        processed_image = np.array(processed_image, dtype=np.float32) / 255.0
        # processed_image = processed_image.astype(np.float32) / 255.0
        processed_image = np.expand_dims(processed_image, axis=1)
        # features = self.vae.encode(torch.tensor(processed_image).to(self.device)).cpu().numpy()
        with torch.no_grad():
            mu, _ = self.vae.encode(torch.tensor(processed_image).to(self.device))
            features = mu.cpu().numpy()
        features = np.hstack((features, image_shape))
        features = self.scaler.transform(features)
        return features

    def is_known_roi(self, roi, add_to_buffer=False):
        if not self.inited:
            return False
        feature = self.extract_features(roi)
        # label = self.kmeans.predict([feature])[0]
        y_pred = self.iforest.predict([feature])
        # return y_pred < 0
        print("y_pred:", y_pred, "minimum:", self.min_anomaly_decision)
        return y_pred <= self.min_anomaly_decision

    def detect_anomaly(self, image):#提取图像的分辨率特征
        feature = self.extract_resolution_features(image)
        label = self.kmeans.predict([feature])[0]  # 预测图像的类别
        # center = self.kmeans.cluster_centers_[label]  # 获得该类别的聚类中心
        # distance = np.linalg.norm(feature - center)  # 计算图像特征与聚类中心的欧氏距离
        
        # 归一化距离，最大距离对应异常概率为1
        # max_distance = np.max([np.linalg.norm(f - center) for f in self.features])
        # anomaly_probability = distance / max_distance
        return float(label == self.anomaly_class)

    def compare_similarity(self, image1, image2):# 识别输入图像是否为异常
        feature1 = self.extract_resolution_features(image1)
        feature2 = self.extract_resolution_features(image2)
        
        # 计算两个特征向量的欧氏距离
        distance = np.linalg.norm(feature1 - feature2)
        similarity = 1 / (1 + distance)  # 使用距离的倒数来表示相似度
        return similarity

if __name__ == "__main__":
    folder_path = "test_split/dataset/0"

    files = os.listdir(folder_path)
    images = [f for f in files]
    images.sort()
    saved_images_folder = 'buffer'
    processor = BoundaryDetector(saved_images_folder)

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = plt.imread(image_path)

        processor.add_normal_samples(image)

    test_images_folder = 'test_split/dataset/1'
    test_files = os.listdir(test_images_folder)
    test_images = [f for f in test_files]
    test_images.sort()
    for image_name in test_images:
        image_path = os.path.join(test_images_folder, image_name)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = plt.imread(image_path)

        if processor.detect_anomaly(image):
            print(f"Anomaly detected in {image_name}")
        else:
            print(f"No anomaly detected in {image_name}")
