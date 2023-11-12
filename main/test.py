import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Test():
    def __init__(self, dev, batch_size, testset, net):
        super(Test, self).__init__()
        self.dev = dev
        self.batch_size = batch_size
        self.testset = testset
        self.net = net
    
    def detect(img, model):
        # propagate through the model
        outputs = model(img.cuda().unsqueeze(0))

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        probas = probas[keep]
        boxes = outputs['pred_boxes'][0, keep]

        bboxes_scaled = boxes.cpu()
        
        return probas, bboxes_scaled
        
    def test(self):
        test_loader = torch.utils.data.DataLoader(self.testset, self.batch_size, shuffle=False)
        self.net.train(False)
        count = 0
        correct = 0

        with torch.no_grad():
            for _, data, target in test_loader:
                if self.dev.type == 'cuda':
                    data = data.to(self.dev)
                    target = target.to(self.dev)
                data = data.float()
                output = self.net(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred = probabilities.argmax(dim=1, keepdim=True)
                if pred == target:
                    correct += 1
                count += 1
        print('Accuracy: ' + str(correct/count))
        
    def run(self):
        self.test()