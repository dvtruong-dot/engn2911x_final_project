from torchvision import datasets, transforms, models
import torch

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 10)
model.eval()

images, labels = next(iter(loader))
outputs = model(images)  # should now run without hanging
print(outputs.shape)
