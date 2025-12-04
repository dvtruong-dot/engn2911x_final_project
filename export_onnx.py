import torch
from torchvision import models


model = models.resnet18(num_classes=10) 
model.eval()  


model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location='cpu'))  


dummy_input = torch.randn(1, 3, 224, 224)  

torch.onnx.export(
    model,
    dummy_input,
    "resnet18_trained.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None 
)

