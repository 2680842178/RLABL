import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.cluster import DBSCAN
import shutil
folder_path = './taxi-mutation'
# if os.path.exists(folder_path):
#     print(f"Folder exists: {folder_path}")
# else:
#     print(f"Folder does not exist: {folder_path}")
# 定义VAE模型
class VAE(Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(32, 32, 1)),
            layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)  # 包含均值和log(方差)
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(8 * 8 * 64, activation='relu'),
            layers.Reshape((8, 8, 64)),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        batch_size = tf.shape(mean)[0]  # 动态获取 batch size
        latent_dim = mean.shape[1]     # 获取 latent dim
        eps = tf.random.normal(shape=(batch_size, latent_dim))  # 生成正态分布
        return mean + tf.exp(0.5 * log_var) * eps

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        return reconstructed

# 数据预处理
image_dir = "taxi-mutation"
latent_dim = 16
images = []

# for file_name in os.listdir(image_dir):
#     if file_name.endswith(".bmp"):
#         image_path = os.path.join(image_dir, file_name)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         image = cv2.resize(image, (64, 64))  # 调整图像大小
#         images.append(image)

images = []  # 用于存储处理后的图像
target_size = (32, 32)  # 目标图像大小

for file_name in os.listdir(image_dir):
    if file_name.endswith(".bmp"):
        image_path = os.path.join(image_dir, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        

        _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)  # 二值化
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建一个空白图像来绘制轮廓
        contour_image = np.zeros_like(image, dtype=np.uint8)
        cv2.drawContours(contour_image, contours, -1, 255, thickness=1)  # 将轮廓绘制到空白图像
        
        image = contour_image
    
        # 获取原始图像尺寸
        h, w = image.shape
        
        # 如果图像超过目标大小，缩放到目标大小
        if h > target_size[0] or w > target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            h, w = target_size  # 更新尺寸信息为缩放后的
        
        # 创建一个目标大小的黑色背景
        padded_image = np.zeros(target_size, dtype=np.uint8)
        
        # 计算放置原始图像的起始位置
        start_y = (target_size[0] - h) // 2
        start_x = (target_size[1] - w) // 2
        
        # 将图像放入黑色背景中
        padded_image[start_y:start_y + h, start_x:start_x + w] = image
        
        images.append(padded_image)

images = np.array(images, dtype=np.float32) / 255.0  # 归一化
images = np.expand_dims(images, axis=-1)  # 添加通道维度

# 训练VAE
vae = VAE(latent_dim)
vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
vae.fit(images, images, epochs=30, batch_size=16)

# 使用VAE降维
mean, log_var = vae.encode(images)
z = vae.reparameterize(mean, log_var)

# 获取原始图像的尺寸作为附加特征
original_sizes = np.array([cv2.imread(os.path.join(image_dir, file_name), cv2.IMREAD_GRAYSCALE).shape for file_name in os.listdir(image_dir) if file_name.endswith(".bmp")])
original_sizes = original_sizes.astype(np.float32)

# 将长宽信息与VAE降维后的特征合并
features = np.hstack((z, original_sizes))  # 将VAE的降维特征与长宽信息连接起来

# 标准化特征
scaler = StandardScaler()
features = scaler.fit_transform(features)

# KMeans聚类
clu_num = 2
kmeans = KMeans(n_clusters=clu_num, random_state=0)
labels = kmeans.fit_predict(features)


# # Kmaens聚类
# kmeans = KMeans(n_clusters=clu_num, random_state=0)
# labels = kmeans.fit_predict(z)

# #DBS
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# labels = dbscan.fit_predict(z)

# 输出结果
for i, label in enumerate(labels):
    print(f"Image {i} -> Cluster {label}")

# 创建分类文件夹
output_dir = "clustered_images"
cluster_dirs = [os.path.join(output_dir, f"cluster_{i}") for i in range(clu_num)]

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 为每个聚类创建子文件夹
for cluster_dir in cluster_dirs:
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

# 保存图像到对应的聚类文件夹
for i, (label, image) in enumerate(zip(labels, images)):
    # 恢复图像的像素值范围（0-255）
    image = (image * 255).astype(np.uint8)
    # 转换为2D图像（去掉单通道维度）
    image = np.squeeze(image, axis=-1)
    # 保存到对应的文件夹
    output_path = os.path.join(cluster_dirs[label], f"image_{i}.bmp")
    cv2.imwrite(output_path, image)

print(f"Images have been saved to {output_dir} by cluster.")