# DL- Developing a Neural Network Classification Model using Transfer Learning

### STEP 5: 

Define loss function (CrossEntropyLoss) and optimizer (Adam). Train the model and plot the loss curve.



### STEP 6: 


Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.




## PROGRAM

### Name: BHAGATHKRISHNA A

### Register Number: 212223230029

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

from pathlib import Path
import zipfile

zip_path = Path("chip_data.zip")
data_path = Path("data")
dataset_path = data_path / "dataset"
if not dataset_path.exists():
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_path)

print(f"Dataset extracted to: {dataset_path.resolve()}")


# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()


# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")
