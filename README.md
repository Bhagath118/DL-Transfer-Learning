# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

Develop an image classification model using transfer learning with VGG19 architecture for the given Chip data provided along the reposistory.



## Neural Network Model



## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.



### STEP 2: 

Load training and testing datasets using ImageFolder.



### STEP 3: 

Visualize sample images from the dataset.



### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.



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

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset
print(f"Total number of training samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1, label = test_dataset[0]
print(f"Shape of the first image : {first_image.shape}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from torchvision.models import VGG19_Weights
model = models.vgg19(weights=VGG19_Weights.DEFAULT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
in_features=model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
def train_model(model, train_loader,test_loader,num_epochs=10):
# Evaluate the model
test_model(model,test_loader)


def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Get the predicted class index
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1042" height="450" alt="image" src="https://github.com/user-attachments/assets/19957437-8dd4-43f0-af6f-d3abfebafbf1" />


<img width="862" height="683" alt="image" src="https://github.com/user-attachments/assets/8f07cf9f-ebb3-48df-9253-20fdf12c1b1e" />


## Confusion Matrix

<img width="798" height="680" alt="image" src="https://github.com/user-attachments/assets/abfe0265-e823-4ae3-a70c-317410dd801b" />


## Classification Report

<img width="1020" height="423" alt="image" src="https://github.com/user-attachments/assets/b764a8ca-4aee-4c00-a53d-2a7def058b84" />


### New Sample Data Prediction

<img width="694" height="810" alt="image" src="https://github.com/user-attachments/assets/9c92071e-7ad3-48d1-b378-b4ebbf753764" />


<img width="700" height="826" alt="image" src="https://github.com/user-attachments/assets/6f9d6a09-ca42-47b6-81ca-a32fb8cf3e3a" />


## RESULT

Developing a Neural Network Classification Model using Transfer Learning was Successfully built
