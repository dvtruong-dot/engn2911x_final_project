import hls4ml
import onnx

onnx_model = onnx.load("resnet18_cifar10.onnx")

# Generate initial config
config = hls4ml.utils.config_from_onnx_model(onnx_model)

print(config)
