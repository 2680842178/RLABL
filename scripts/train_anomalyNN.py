import os
from model import ACModel, CNN
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torchvision import datasets

dataset_path = "../test/anomaly_datasets/"
dataset = datasets.ImageFolder(dataset_path)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

anomalyNN = CNN(2)
device = torch.device("cuda")
anomalyNN.to(device)

criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(anomalyNN.parameters(), lr = 0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = anomalyNN(images)
        loss = criteria(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {i + 1}, Loss: {loss.item()}")
        
print("Finished Training")
torch.save(anomalyNN.state_dict(), "StateNN/anomalyNN_0716.pth")


