import torch
from torch import Tensor

import torch

def quantize_tensor(tensor, num_bits, symmetric):
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    signed = min_value < 0
        
    if signed: # Symmetric
        if symmetric:
            r_max = max(abs(min_value), abs(max_value))
            scale = r_max / (2 ** (num_bits - 1) - 1)
            quantized_tensor = torch.round(tensor / scale)
            quantized_tensor = torch.clip(quantized_tensor, -2 ** (num_bits - 1), 2 ** (num_bits - 1) - 1)
            dequantized_tensor = quantized_tensor * scale
        else: # Asymmetric
            scale = (max_value - min_value) / (2 ** num_bits - 1)
            quantized_tensor = torch.round((tensor - min_value) / scale)
            quantized_tensor = torch.clip(quantized_tensor, 0, 2 ** num_bits - 1)  # Adjust the clip range
            dequantized_tensor = quantized_tensor * scale + min_value
    else: # Unsigned
        scale = max_value / (2 ** num_bits - 1)
        quantized_tensor = torch.round(tensor / scale)
        quantized_tensor = torch.clip(quantized_tensor, 0, 2 ** num_bits - 1)
        dequantized_tensor = quantized_tensor * scale
    return dequantized_tensor


def quantize_model(model, num_bits=8):
    quantized_model = model
    for name, param in quantized_model.named_parameters():
        if 'weight' or 'bias' in name:  # Quantize weights
            data = param.data.cpu()
            quantized_weight= quantize_tensor(data, num_bits=8, symmetric=False)
            param.data = quantized_weight.to(param.device)
    return quantized_model