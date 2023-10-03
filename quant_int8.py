import torch

def torch_quantize(model, backend='x86'):
    model.eval()
    #scripted_model = torch.jit.script(model)
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    return quantized_model