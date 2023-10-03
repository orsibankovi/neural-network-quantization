import model_load
import torch
import transformers
import quant_int8
import nnef_tools

if __name__ == "__main__":
    #object detection
    yolov5m = model_load.load_models('ultralytics/yolov5', 'yolov5m')
    yolov5m.eval()
    #yolos_tiny =  transformers.YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    #yolos_tiny.eval()

    #image classification
    #efficientnet = transformers.EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")
    #segformer = transformers.SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    #efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    #efficientnet.eval()

    quant_yolov5m = quant_int8.torch_quantize(yolov5m)
    dummy_input = torch.randn(1, 3, 640, 640)
    onnx_yolov5m = torch.onnx.export(quant_yolov5m, dummy_input, 'yolov5m.onnx')
    for name, param in quant_yolov5m.named_parameters():
        print(name, param.size())
    
