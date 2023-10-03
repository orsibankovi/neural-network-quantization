import torch
import transformers

def load_models(name, model):
    net = torch.hub.load(name, model, pretrained=True)
    net.eval()
    return net