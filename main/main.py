import model_load
import torch
import transformers
import quant_int8
import imagenet_preprocess as preprocess
import object_detection_test
import coco_preprocess
import test

if __name__ == "__main__":
    #image classification (IMAGENET-1K)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device("cpu")
    '''
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    efficientnet.eval()
    
    testset = preprocess.GetDataset()
    
    print('fp32')
    test_fp32 = test.Test(dev, 1, testset, efficientnet)
    #test_fp32.run()
    
    efficientnet_int8 = quant_int8.quantize_model(efficientnet)
    efficientnet_int8.eval()
    print('int8')
    test_int8 = test.Test(dev, 1, testset, efficientnet_int8)
    test_int8.run()
    
    for name, param in efficientnet.named_parameters():
        print(name)
        
    for name, param in efficientnet_int8.named_parameters():
        print(name)
    '''
    #segformer = transformers.SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    
    #object detection
    yolov5s = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolov5s.eval()
        
    # COCO adathalmaz betöltése
    dataDir = 'C:/Users/orsolya.bankovi/Documents/uni/deepL/deepLearning_project/main/COCO'  # Útvonal a COCO adathalmazhoz
  # Útvonal a COCO adathalmazhoz
    dataType = 'val2017'  # Vagy 'train2017' vagy 'test2017' attól függően, hogy melyik adathalmazt szeretné használni

    # Inicializáljuk a COCO adathalmazat
    coco_dataset = coco_preprocess.GetCOCODataset(dataDir, dataType)
    '''
    # Hívjuk meg a teszt osztályt a fent inicializált paraméterekkel
    tester = object_detection_test.ObjectDetectionTest(dev=dev, batch_size=1, testset=coco_dataset, net=yolov5s)
    tester.run()
    
    

    onnx_yolov5m = torch.onnx.export(quant_yolov5m, dummy_input, 'yolov5m.onnx')
    for name, param in quant_yolov5m.named_parameters():
        print(name, param.data)
        
    for name, param in yolov5m.named_parameters():
        print(name, param.data)
        
    yolos_tiny =  transformers.YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    yolos_tiny.eval()
    
    for name, param in yolos_tiny.named_parameters():
        print(name, param.data)
        
    for name, param in yolos_tiny.named_parameters():
        print(name, param.data)
    '''
    
    yolos_tiny =  transformers.YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    yolos_tiny.eval()
    
    tester1 = object_detection_test.ObjectDetectionTest(dev=dev, batch_size=1, testset=coco_dataset, net=yolos_tiny)
    tester1.run()
    
