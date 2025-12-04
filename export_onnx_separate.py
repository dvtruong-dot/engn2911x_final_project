import torch
import torch.nn as nn
import torchvision.models as models

# -----------------------
# Load your trained model
# -----------------------
# IMPORTANT: modify the state_dict path below  
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)   # adjust to match your trained model
model.load_state_dict(torch.load("../resnet18_cifar10.pth", map_location="cpu"))
model.eval()


# Helper to export onnx
def export_onnx(layer, input_shape, filename):
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        layer,
        dummy,
        filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )
    print(f"Exported {filename}")


# Extract layers
# conv1
conv1 = model.conv1
export_onnx(conv1, (1, 3, 224, 224), "conv1.onnx")

# layer1 block 0
layer1_0_conv1 = model.layer1[0].conv1
layer1_0_conv2 = model.layer1[0].conv2

export_onnx(layer1_0_conv1, (1, 64, 56, 56), "layer1_0_conv1.onnx")
export_onnx(layer1_0_conv2, (1, 64, 56, 56), "layer1_0_conv2.onnx")

# layer1 block 1
layer1_1_conv1 = model.layer1[1].conv1
layer1_1_conv2 = model.layer1[1].conv2

export_onnx(layer1_1_conv1, (1, 64, 56, 56), "layer1_1_conv1.onnx")
export_onnx(layer1_1_conv2, (1, 64, 56, 56), "layer1_1_conv2.onnx")


# layer2 (note: first block has downsample = /2)


# layer2 block 0
layer2_0_conv1 = model.layer2[0].conv1
layer2_0_conv2 = model.layer2[0].conv2

# After layer1, activation size = 56×56, layer2.0.conv1 has stride=2 → input is 64 channels
export_onnx(layer2_0_conv1, (1, 64, 56, 56), "layer2_0_conv1.onnx")
export_onnx(layer2_0_conv2, (1, 128, 28, 28), "layer2_0_conv2.onnx")

# layer2 block 1
layer2_1_conv1 = model.layer2[1].conv1
layer2_1_conv2 = model.layer2[1].conv2

export_onnx(layer2_1_conv1, (1, 128, 28, 28), "layer2_1_conv1.onnx")
export_onnx(layer2_1_conv2, (1, 128, 28, 28), "layer2_1_conv2.onnx")
