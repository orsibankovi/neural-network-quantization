import torch
import numpy as np

class ObjectDetectionTest():
    def __init__(self, dev, batch_size, testset, net):
        super(ObjectDetectionTest, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
        
    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width<0) or (height <0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+(1e-10))
        return iou

    def evaluate_bounding_boxes(self, pred_boxes, true_boxes, threshold=0.0):
        correct = 0
        max_ious = []
        for i in range(pred_boxes.shape[0]):
            max_iou = 0
            for j in range(true_boxes.shape[0]):
                iou = self.iou(pred_boxes[i, :4], true_boxes[j, :4])
                if iou > max_iou:
                    max_iou = iou
            if max_iou > threshold:
                correct += 1
            max_ious.append(max_iou)
                  
        return max_ious, correct
        
    def test_yolov5(self):
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        self.net.eval()  # Set the model to evaluation mode
        avg_ious = []
        length = 0
        correct = 0
        all_ious = []
        with torch.no_grad():
            for k, data in enumerate(test_loader):
                img_path, img, targets = data
                if self.dev.type == 'cuda':
                    targets = [{k: v.to(self.dev) for k, v in t.items()} for t in targets]
                output = self.net(img)
                predicted_boxes = output.xyxy[0]
                #print(output.pandas().xyxy[0])  # Pandas DataFrame
                true_boxes = targets['boxes']
                if len(true_boxes) == 0:
                    continue
                
                xmin = true_boxes[:, :, 0]
                ymin = true_boxes[:, :, 1]
                xmax = true_boxes[:, :, 0] + true_boxes[:, :, 2]
                ymax = true_boxes[:, :, 1] + true_boxes[:, :, 3]
                true_boxes = torch.stack((xmin, ymin, xmax, ymax), 2)
                
                true_boxes = torch.squeeze(true_boxes, dim=0)

                #predicted_boxes = torch.squeeze(predicted_boxes, dim=0)
                ious, score = self.evaluate_bounding_boxes(predicted_boxes, true_boxes)
                correct += score
                avg_ious.append(ious)
                length += len(predicted_boxes) if len(predicted_boxes) < len(true_boxes) else len(true_boxes)
                for i in ious:
                    all_ious.append(i)
                if k % 500 == 0:
                    print('Average IoU: ', np.average(np.array(all_ious)))
        print('Accuracy: ', correct/length)
        print('Average IoU: ', np.average(np.array(all_ious)))
        
    def test_yolos(self):
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        self.net.eval()  # Set the model to evaluation mode
        avg_ious = []
        length = 0
        correct = 0
        all_ious = []
        with torch.no_grad():
            for k, data in enumerate(test_loader):
                _, img, targets = data
                if self.dev.type == 'cuda':
                    targets = [{k: v.to(self.dev) for k, v in t.items()} for t in targets]
                output = self.net(img)
            
                predicted_boxes = output.pred_boxes
                true_boxes = targets['boxes'].data
                x0_norm = torch.squeeze((true_boxes[:, :, 0] / img.shape[-2]), dim=0)
                x1_norm = torch.squeeze((true_boxes[:, :, 2] / img.shape[-2]), dim=0)
                y0_norm = torch.squeeze(true_boxes[:, :, 1] / img.shape[-1], dim=0)
                y1_norm = torch.squeeze(true_boxes[:, :, 3] / img.shape[-1], dim=0)

                true_boxes = torch.stack((x0_norm, y0_norm, x1_norm, y1_norm), 1)
                if len(true_boxes) == 0:
                    continue
                
                xmin = true_boxes[:, :, 0]
                ymin = true_boxes[:, :, 1]
                xmax = true_boxes[:, :, 0] + true_boxes[:, :, 2]
                ymax = true_boxes[:, :, 1] + true_boxes[:, :, 3]
                true_boxes = torch.stack((xmin, ymin, xmax, ymax), 2)
                
                true_boxes = torch.squeeze(true_boxes, dim=0)

                #predicted_boxes = torch.squeeze(predicted_boxes, dim=0)
                ious, score = self.evaluate_bounding_boxes(predicted_boxes, true_boxes)
                correct += score
                avg_ious.append(ious)
                length += len(predicted_boxes) if len(predicted_boxes) < len(true_boxes) else len(true_boxes)
                for i in ious:
                    all_ious.append(i)
                if k % 500 == 0:
                    print('Average IoU: ', np.average(np.array(all_ious)))
        print('Accuracy: ', correct/length)
        print('Average IoU: ', np.average(np.array(all_ious)))
        
    def collate_fn(self, batch):
        return tuple(zip(*batch))
