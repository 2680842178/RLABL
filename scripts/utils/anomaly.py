import argparse
import yaml
import time
import copy
import datetime
import torch_ac
import tensorboardX
from torchvision import transforms
import sys
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import gymnasium as gym
from typing import Optional, Callable, List
from abc import abstractmethod
from .process import contrast_ssim


import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

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


class ClusterAnomalyDetector:
    def __init__(self):
        self.kmeans = None
        self.contrast_value = 0.5

    def add_samples(self, image_list):
        self.image_list = image_list
        for image in image_list:
            cv2.imwrite(f"taxi_mutations/{time.time()}.bmp", image)
        self.features = [self.extract_resolution_features(image) for image in image_list]  # 提取图像特征
        self.kmeans = KMeans(n_clusters=2, random_state=42)  # 使用KMeans进行聚类
        self.kmeans.fit(self.features)  # 对特征进行拟合
        labels = self.kmeans.labels_
        label_counts = np.bincount(labels)
        print("labels:", labels)
        self.anomaly_class = np.argmin(label_counts)
        for i, label in enumerate(labels):
            if label == self.anomaly_class:
                return image_list[i]
        return None

    def contrast(self, img1, img2):
        feature1 = self.extract_resolution_features(img1)
        feature2 = self.extract_resolution_features(img2)
        lebel1 = self.kmeans.predict([feature1])[0]
        lebel2 = self.kmeans.predict([feature2])[0]
        similarity = 1 if lebel1 == lebel2 else 0
        return similarity

    def extract_resolution_features(self, image):# 初始化异常检测器，输入一个图像列表，内部处理并初始化KMeans模型。
        height, width = image.shape
        return np.array([width, height])

    def is_known_roi(self, roi, add_to_buffer=False):
        if self.kmeans is None:
            return False
        feature = self.extract_resolution_features(roi)
        label = self.kmeans.predict([feature])[0]
        return label == self.anomaly_class

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