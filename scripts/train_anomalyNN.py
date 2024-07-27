import os
from tqdm import tqdm
from model import ACModel, CNN
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt 

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

class ImageDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None, label=None):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        self.transform = transform
        self.lebel = label

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        # img_name = self.img_path[idx]
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = plt.imread(img_item_path)
        #print(img.shape)
        label = int(self.label_dir) 
        label = torch.as_tensor(label, dtype=torch.long)
        if self.transform:
            img = self.transform(img)
            # print(img.shape)
            # img = img.permute(2, 0, 1)
            # print(img.shape)
        # print(label)
        ##plt.show()
        return img, label

class ConvertRGBAtoRGB(object):
    def __call__(self, img):
        return img.convert('RGB')
root_dir = "../test/dataset1"
dir0 = "0"
dir1 = "1"
transform = transforms.Compose([
    #$ ConvertRGBAtoRGB(), 
    transforms.ToTensor()
])
dataset0 = ImageDataset(root_dir, dir0, transform = transform)
dataset1 = ImageDataset(root_dir, dir1, transform = transform)
dataset = dataset0 + dataset1 
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size   
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])    

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)
device = "cuda" if torch.cuda.is_available() else "cpu"


anomalyNN = CNN(2)
anomalyNN.to(device)

num_epochs = 25
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(anomalyNN.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1) 

writer = SummaryWriter("logs")    

def train_model(num_epochs: int,
          model,
          criterion,
          optimizer,
          scheduler,
          train_loader,
          test_loader,
          device,
          writer):
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}") 
        running_loss = 0.0
        total = 0.0
        correct = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)   
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        torch.save(model, "../test/model/test_{}.pth".format(epoch))
        print(f'Epoch[{epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("train_auc", epoch_accuracy, epoch)

        model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(test_loader.dataset)
            epoch_accuracy = 100 * correct / total
            print(f'Test: Loss: {epoch_loss:.4f}, test Accuracy: {epoch_accuracy:.2f}%')
            writer.add_scalar("test_loss", epoch_loss, epoch)
            writer.add_scalar("test_auc", epoch_accuracy, epoch)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the network on the test images: {100 * correct / total}%")

train_model(num_epochs, anomalyNN, criterion, optimizer, scheduler, train_loader, test_loader, device, writer)
# anomalyNN = torch.load("../test/model/test_24.pth")
# # test_model(anomalyNN, test_loader, device)
# dir0 = "../test/dataset1/0"
# dir1 = "../test/dataset1/1"
# num = 0
# for image in os.listdir(dir1):
#     num += 1
#     img = plt.imread(os.path.join(dir1, image))
#     img = transforms.ToTensor()(img)
#     img = img.unsqueeze(0)
#     img = img.to(device)
#     output = anomalyNN(img)
#     print(output)
#     _, predicted = torch.max(output, 1)
#     print(f"Predicted: {predicted.item()}")
# print(num)