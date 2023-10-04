import torch
from torch import Tensor

def quantize_tensor(tensor: Tensor, num_bits=8)->Tensor:
    symmetric = True
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    symmetric = True if (min_value > 0 or max_value < 0) else True
        
    if symmetric:
        r_max = max(abs(min_value), abs(max_value))
        scale = r_max / (2 ** (num_bits - 1) - 1)
        quantized_tensor = torch.round(tensor / scale)
        quantized_tensor = torch.clip(quantized_tensor, -2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1)
    else: # Asymmetric
        scale = (max_value - min_value) / (2 ** num_bits - 1)
        quantized_tensor = torch.round((tensor - min_value) / scale)
        quantized_tensor = torch.clip(quantized_tensor, 0, 2 ** num_bits - 1)
    print(torch.min(quantized_tensor*scale)-min_value)   
    print(torch.max(quantized_tensor*scale)-max_value)    
    print(torch.mean(quantized_tensor*scale)-tensor.mean())
    return quantized_tensor, scale


def asymmetric_quantize_tensor(tensor, min_value, max_value, num_bits=8):
    # Compute the scale factor based on the given min and max values
    scale = (max_value - min_value) / (2 ** num_bits - 1)
    clipped_tensor = torch.clamp(tensor, min_value, max_value)
    quantized_tensor = torch.round((clipped_tensor - min_value) / scale)
    quantized_tensor = quantized_tensor * scale + min_value
    return quantized_tensor

def quantize_model(model, num_bits=8):
    quantized_model = model
    for name, param in quantized_model.named_parameters():
        print(name)
        if 'weight' in name:  # Quantize weights
            quantized_weight, scale = quantize_tensor(param.data.cpu(), num_bits)
            param.data = quantized_weight.to(param.device)
        if 'bias' in name:  # Don't quantize biases
            continue
    return quantized_model