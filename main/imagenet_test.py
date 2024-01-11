import torch

class ImageNetTest():
    def __init__(self, batch_size):
        super(ImageNetTest, self).__init__()
        self.batch_size = batch_size
        
    def test(self, net, testset, dev):
        test_loader = torch.utils.data.DataLoader(testset, self.batch_size, shuffle=False)
        net = net.to(dev)
        net.train(False)
        count = 0
        correct = 0


        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(dev)
                target = target.to(dev)
                output = net(data)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred = probabilities.argmax(dim=1, keepdim=True)
                if pred == target:
                    correct += 1
                count += 1
        print('Accuracy: ' + str(correct/count))
        return correct/count