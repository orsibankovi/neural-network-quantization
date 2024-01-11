import torch
from torch.quantization import quantize_fx
from torch.ao.quantization import QuantStub, DeQuantStub, QConfigMapping
import copy

def quantize_model(model, calibdata, dev):
    torch.no_grad()
    m = copy.deepcopy(model).to(dev)
    m.eval()

    example_inputs = torch.unsqueeze(calibdata.input_images[0], dim=0).to(dev)
    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    model_prepared = quantize_fx.prepare_fx(m, qconfig_mapping, example_inputs)

    with torch.inference_mode():
        for i in range(5):
            x = torch.unsqueeze(calibdata.input_images[i], dim=0).to(dev)
            model_prepared(x)
            
    model_quantized = quantize_fx.convert_fx(model_prepared)

    return model_quantized