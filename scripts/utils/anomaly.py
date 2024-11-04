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
import gymnasium as gym
from typing import Optional, Callable, List
from abc import abstractmethod


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

class BoundaryDetector(AnomalyDetector):
    def __init__(self, saved_images_folder):
        super().__init__()
        self.saved_images_folder = saved_images_folder
        if not os.path.exists(self.saved_images_folder):
            os.makedirs(self.saved_images_folder)
        self.saved_images = self.load_saved_images()

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

    def is_known_image(self, image, add_to_buffer=False):
        _, processed = self.preprocess_RGBimage(image)
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
                # print(similar)
                if similar > 0.9999999:
                    break
            if similar < 0.9999999 and similar >= 0:
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

# 使用示例