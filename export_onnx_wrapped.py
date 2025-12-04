import torch
import torch.nn as nn
import torchvision.models as models

class WrappedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    
    def forward(self, x):
        return self.layer(x)
    

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)   # adjust to match your trained model
model.load_state_dict(torch.load("resnet18_cifar10.pth", map_location="cpu"))
model.eval()

wrapped = WrappedLayer(model.conv1)

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    wrapped,
    dummy,
    "conv1.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13
)