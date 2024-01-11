import torch
import torch.nn as nn
from torch.quantization import quantize_fx
import copy
from torch.ao.quantization import QConfigMapping

class QuantizationAwareTraining():
    def __init__(self, model, trainset, dev, batch_size):
        super(QuantizationAwareTraining, self).__init__()
        self.model = model
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        total_size = len(trainset)
        split1_size = int(0.2 * total_size)
        _trainset = torch.utils.data.Subset(trainset, range(split1_size))
        self.train_loader = torch.utils.data.DataLoader(_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        _testset = torch.utils.data.Subset(trainset, range(split1_size, total_size))
        self.test_loader = torch.utils.data.DataLoader(_testset, batch_size=1, shuffle=False, num_workers=0)

    def train(self, net):
        net = net.to(self.dev)
        best = 1000
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        for epoch in range(5):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs = inputs.to(self.dev)
                labels = labels.to(self.dev)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if best > running_loss/i:
                best = running_loss/i
                save = copy.deepcopy(net)
                best_epoch = epoch
        print('Finished Training')
        print('Best Epoch: ' + str(best_epoch))
        print('Best Loss: ' + str(best))
        return save

    def quantize_model(self, model, calibdata):
        model = model.to(self.dev)
        torch.no_grad()
        m = copy.deepcopy(model)
        m.eval()
        
        example_inputs = torch.unsqueeze(calibdata.input_images[0], dim=0)
        
        qconfig = torch.quantization.get_default_qat_qconfig("fbgemm", 1)
        qconfig_mapping = QConfigMapping().set_activations(qconfig).set_weights(qconfig)

        
        m = quantize_fx.fuse_fx(m)
        model_prepared = quantize_fx.prepare_qat_fx(m, qconfig_mapping, example_inputs)

        with torch.inference_mode():
            for i in range(10):
                x = torch.unsqueeze(calibdata.input_images[i], dim=0).to(self.dev)
                model_prepared(x)
                    
        trained_model = self.train(model_prepared)
        trained_model = trained_model.to(torch.device('cpu'))
        model_quantized = quantize_fx.convert_fx(trained_model)
        model_quantized.train(False)
        model_quantized.eval()
        self.test(model_quantized)
        return model_quantized
    
    def test(self, net):
        net = net.to(torch.device('cpu'))
        net.train(False)
        count = 0
        correct = 0


        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(torch.device('cpu'))
                target = target.to(torch.device('cpu'))
                output = net(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred = probabilities.argmax(dim=1, keepdim=True)
                if pred == target:
                    correct += 1
                count += 1
        print('Accuracy: ' + str(correct/count))