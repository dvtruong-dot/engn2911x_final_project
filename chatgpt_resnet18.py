import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np

# Configurations
device = torch.device("cpu")
BATCH_SIZE = 16  # smaller for CPU speed
EPOCHS = 10
LR = 1e-4

# ---------------------------
# Datasets
# ---------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------
# Model
# ---------------------------
print("Loading model...")
model = resnet18(weights="IMAGENET1K_V1")  # or weights=None for faster CPU-only testing
model.fc = nn.Linear(512, 10)

# Freeze all except layer4 + fc
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ---------------------------
# Training
# ---------------------------
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss/len(trainloader):.4f}")

# ---------------------------
# Evaluation
# ---------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
acc = 100 * correct / total
print(f"Test Accuracy: {acc:.2f}%")

# ---------------------------
# Save model
# ---------------------------
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Saved model: resnet18_cifar10.pth")

# ---------------------------
# Export to ONNX
# ---------------------------
dummy = torch.randn(1,3,224,224, device=device)
torch.onnx.export(
    model,
    dummy,
    "resnet18_cifar10.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("Saved ONNX: resnet18_cifar10.onnx")

# ---------------------------
# Save golden inputs/outputs
# ---------------------------
model.eval()
with torch.no_grad():
    example_images, _ = next(iter(testloader))
    example_images = example_images[:16].to(device)
    golden_out = model(example_images).cpu().numpy()
    np.save("golden_inputs.npy", example_images.cpu().numpy())
    np.save("golden_outputs.npy", golden_out)
print("Saved golden inputs/outputs")
