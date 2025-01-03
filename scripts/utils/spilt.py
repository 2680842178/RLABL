import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from PIL import Image
import shutil


class AnomalyDetector:
    def __init__(self, image_list):
        self.image_list = image_list
        self.features = [self.extract_resolution_features(image) for image in image_list]  # 提取图像特征
        self.kmeans = KMeans(n_clusters=2, random_state=42)  # 使用KMeans进行聚类
        self.kmeans.fit(self.features)  # 对特征进行拟合

    def extract_resolution_features(self, image):# 初始化异常检测器，输入一个图像列表，内部处理并初始化KMeans模型。
        height, width = image.shape
        return np.array([width, height])

    def detect_anomaly(self, image):#提取图像的分辨率特征
        feature = self.extract_resolution_features(image)
        label = self.kmeans.predict([feature])[0]  # 预测图像的类别
        center = self.kmeans.cluster_centers_[label]  # 获得该类别的聚类中心
        distance = np.linalg.norm(feature - center)  # 计算图像特征与聚类中心的欧氏距离
        
        # 归一化距离，最大距离对应异常概率为1
        max_distance = np.max([np.linalg.norm(f - center) for f in self.features])
        anomaly_probability = distance / max_distance
        return anomaly_probability

    def compare_similarity(self, image1, image2):# 识别输入图像是否为异常
        feature1 = self.extract_resolution_features(image1)
        feature2 = self.extract_resolution_features(image2)
        
        # 计算两个特征向量的欧氏距离
        distance = np.linalg.norm(feature1 - feature2)
        similarity = 1 / (1 + distance)  # 使用距离的倒数来表示相似度
        return similarity


