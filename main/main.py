import torch
import transformers
import torchvision
import quant_int8
import qat
import imagenet_preprocess as preprocess
import object_detection_test
import coco_preprocess
import imagenet_test
import ptq
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(model, name, testset, dev, batch_size):
    test = imagenet_test.ImageNetTest(batch_size)
    print(name, 'FP32')
    dev = torch.device("cpu")
    model = model.to(dev)
    fp32 = test.test(model, testset, dev)
    model_int8 = quant_int8.quantize_model(model, testset, dev)
    print(name, 'INT8')
    int8 =  test.test(model_int8, testset, dev)
    ptq_ = ptq.PTQ(model, testset, dev)
    bias_corr = ptq_.quantize_model()
    print(name, 'PTQ INT8')
    ptq_int8 = test.test(bias_corr, testset, dev)
    
    return fp32, int8, ptq_int8

if __name__ == "__main__":
    #image classification (IMAGENET)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    
    testset = preprocess.GetDataset()
    with open('./imagenet_classes.txt', 'w') as f:
    
        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        fp32, int8, ptq_int8 = evaluate_model(alexnet, 'AlexNet', testset, dev, 1)
        f.write('AlexNet FP32: ' + str(fp32) + '\n')
        f.write('AlexNet INT8: ' + str(int8) + '\n')
        f.write('AlexNet PTQ INT8: ' + str(ptq_int8) + '\n')
        
        efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        fp32, int8, ptq_int8 = evaluate_model(efficientnet, 'EfficientNet', testset, dev, 1)
        f.write('EfficientNet FP32: ' + str(fp32) + '\n')
        f.write('EfficientNet INT8: ' + str(int8) + '\n')
        f.write('EfficientNet PTQ INT8: ' + str(ptq_int8) + '\n')
        
        mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)
        fp32, int8, ptq_int8 = evaluate_model(mobilenet_v2, 'MobileNetV2', testset, dev, 1)
        f.write('MobileNetV2 FP32: ' + str(fp32) + '\n')
        f.write('MobileNetV2 INT8: ' + str(int8) + '\n')
        f.write('MobileNetV2 PTQ INT8: ' + str(ptq_int8) + '\n')
        
        shufflenet = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
        fp32, int8, ptq_int8 = evaluate_model(shufflenet, 'ShuffleNet', testset, dev, 1)
        f.write('ShuffleNet FP32: ' + str(fp32) + '\n')
        f.write('ShuffleNet INT8: ' + str(int8) + '\n')
        f.write('ShuffleNet PTQ INT8: ' + str(ptq_int8) + '\n')

        maxvit_t = torch.hub.load('pytorch/vision', 'maxvit_t', pretrained=True)
        fp32, int8, ptq_int8 = evaluate_model(maxvit_t, 'MaxViT', testset, dev, 1)
        f.write('MaxViT FP32: ' + str(fp32) + '\n')
        f.write('MaxViT INT8: ' + str(int8) + '\n')
        f.write('MaxViT PTQ INT8: ' + str(ptq_int8) + '\n')