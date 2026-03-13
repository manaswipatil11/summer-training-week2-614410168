import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import wandb

from rfmid_dataset import RFMiDDataset
from focal_loss import FocalLoss

# -----------------------
# Hyperparameters
# -----------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = 224
TRAIN_SPLIT = 0.8
NUM_CLASSES = 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# WandB init
# -----------------------
wandb.init(project="rfmid_cnn_training")

# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------
# Dataset
# -----------------------
dataset = RFMiDDataset(
    csv_file='Retinal-disease-classification/labels.csv',
    img_dir='Retinal-disease-classification/images',
    transform=transform
)

train_size = int(TRAIN_SPLIT * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------
# Model
# -----------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# -----------------------
# Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------
# Training loop
# -----------------------
for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # -----------------------
    # Validation
    # -----------------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    wandb.log({
        "train_loss": avg_train_loss,
        "val_accuracy": val_acc
    })

# -----------------------
# Save model
# -----------------------
torch.save(model.state_dict(), "rfmid_resnet18.pt")
print("Training finished.")
