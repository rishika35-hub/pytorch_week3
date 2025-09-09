import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from custom_resnet import ResNet18
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch.nn.functional as F

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 128
learning_rate = 0.1
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CIFAR-10 Dataset
# -----------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = train_dataset.classes

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = ResNet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# -----------------------------
# Training
# -----------------------------
train_losses, val_losses = [], []
train_acc, val_acc = [], []
all_preds, all_labels = [], []
correct_images, correct_labels, correct_preds = [], [], []
incorrect_images, incorrect_labels, incorrect_preds = [], [], []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_losses.append(running_loss / total)
    train_acc.append(100. * correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            if epoch == num_epochs - 1:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                for i in range(len(labels)):
                    if predicted[i] == labels[i]:
                        if len(correct_images) < 16:
                            correct_images.append(images[i].cpu())
                            correct_labels.append(labels[i].cpu().item())
                            correct_preds.append(predicted[i].cpu().item())
                    else:
                        if len(incorrect_images) < 16:
                            incorrect_images.append(images[i].cpu())
                            incorrect_labels.append(labels[i].cpu().item())
                            incorrect_preds.append(predicted[i].cpu().item())
    
    val_losses.append(val_loss / val_total)
    val_acc.append(100. * val_correct / val_total)
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.2f}% "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc[-1]:.2f}%")

print("Training complete!")
os.makedirs("runs/cls", exist_ok=True)
torch.save(model.state_dict(), "runs/cls/resnet18_cifar10.pth")

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = None
        self.activations = None
        self.gradients = None

        def save_activation(module, input, output):
            self.activations = output

        def save_gradient(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        for name, module in self.model.named_modules():
            if name == target_layer_name:
                self.target_layer = module
                self.target_layer.register_forward_hook(save_activation)
                self.target_layer.register_backward_hook(save_gradient)
                break
        
        if self.target_layer is None:
            raise RuntimeError(f"Target layer '{target_layer_name}' not found.")

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = F.one_hot(class_idx, num_classes=output.size(1)).float().to(x.device)
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze(1)

def visualize_gradcam(model, image, label, pred, target_layer, filename):
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam(image.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])
    img = image.permute(1, 2, 0).numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original\nL:{classes[label]}, P:{classes[pred]}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"runs/cls/{filename}")
    plt.close()

print("Generating Grad-CAM heatmaps...")
target_layer = 'layer4.1.conv2'
if correct_images:
    for i in range(min(4, len(correct_images))):
        img = correct_images[i]
        lbl = correct_labels[i]
        pred = correct_preds[i]
        visualize_gradcam(model, img, lbl, pred, target_layer, f"gradcam_correct_{i}.png")
if incorrect_images:
    for i in range(min(4, len(incorrect_images))):
        img = incorrect_images[i]
        lbl = incorrect_labels[i]
        pred = incorrect_preds[i]
        visualize_gradcam(model, img, lbl, pred, target_layer, f"gradcam_incorrect_{i}.png")
        
# -----------------------------
# Save Plots
# -----------------------------
print("Saving training curves, confusion matrix, and prediction grids...")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.legend()
plt.title("ResNet-18 CIFAR-10 Training Curves")
plt.savefig("runs/cls/curves_cls.png")
plt.close()

cm = confusion_matrix(all_labels, all_preds, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Normalized Confusion Matrix")
plt.savefig("runs/cls/confusion_matrix.png")
plt.close()

def plot_images(images, labels, preds, filename):
    plt.figure(figsize=(8,8))
    for i in range(len(images)):
        plt.subplot(4,4,i+1)
        img = images[i].permute(1,2,0).numpy()
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        img = (img * std) + mean
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"L:{classes[labels[i]]}\nP:{classes[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_images(correct_images, correct_labels, correct_preds, "runs/cls/preds_grid.png")
plot_images(incorrect_images, incorrect_labels, incorrect_preds, "runs/cls/miscls_grid.png")

print("All ResNet-18 tasks complete! Visuals saved in runs/cls/")