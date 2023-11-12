import torch

def quantize_tensor(tensor, num_mantissa_bits=3, num_exponent_bits=4):
    # Scaling the tensor to fit into the FP8 range
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    scale = max(abs(max_value), abs(min_value))
    scaled_tensor = tensor / scale

    # Convert the scaled tensor to FP8 format
    fp8_tensor = torch.quantize_per_tensor(scaled_tensor, scale, 0, torch.quint8)

    return fp8_tensor

def quantize_model(model):
    quantized_model = model
    for name, param in quantized_model.named_parameters():
        if 'weight' in name or 'bias' in name:  # Quantize weights and biases
            data = param.data.cpu()
            quantized_weight = quantize_tensor(data)
            param.data = quantized_weight.to(param.device)
    return quantized_model
