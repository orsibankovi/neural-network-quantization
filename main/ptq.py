import copy
import torch
from torch.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from quant_int8 import quantize_model

class PTQ():
    def __init__(self, model, calibdata, dev):
        super(PTQ, self).__init__()
        self.model = model
        self.calibdata = calibdata
        self.dev = dev
        self.hooks_names = []
        self.hook_values = []
        self.quant_hook_values = []

        self.target_modules = {'Conv2d', 'Linear'}

    def get_hooks(self, model, target_modules, quantized=False):
        """Get hooks for given model and target modules."""
        if quantized:
            self.quant_hook_values.clear()
        else:    
            self.hook_values.clear()
        m = copy.deepcopy(model)

        def hook_fn(module, input, output):
            if quantized:
                self.quant_hook_values.append(torch.dequantize(output.detach().cpu()))
            else:
                self.hook_values.append(output.detach().cpu().float())

        self.hooks_names.clear()
        hooks = []
        for module in m.named_modules():
            if module[1].__class__.__name__ in target_modules:
                self.hooks_names.append(module[0])
                hook = module[1].register_forward_hook(hook_fn)
                hooks.append(hook)

        with torch.inference_mode():
            x = torch.stack(self.calibdata.input_images[5:10], dim=0).to(self.dev)
            m(x)

        for hook in hooks:
            hook.remove()

    def get_module(self, model, name):
        """Given name of submodule, this function grabs the submodule from given model."""
        return dict(model.named_modules())[name]

    def parent_child_names(self, name):
        """Split full name of submodule into parent submodule's full name and submodule's name."""
        split_name = name.rsplit('.', 1)
        if len(split_name) == 1:
            return '', split_name[0]
        else:
            return split_name[0], split_name[1]

    def get_param(self, module, attr):
        param = getattr(module, attr, None)
        if callable(param):
            return param()
        else:
            return param

    def get_quantized_model(self, model):
        torch.no_grad()
        m = copy.deepcopy(model)
        m.eval()

        example_inputs = torch.unsqueeze(self.calibdata.input_images[0], dim=0)
        qconfig = torch.quantization.get_default_qconfig("fbgemm")
        qconfig_mapping = QConfigMapping().set_global(qconfig)

        m = quantize_fx.fuse_fx(m)
        m = torch.quantization.QuantWrapper(m)
        model_prepared = torch.quantization.prepare(m, qconfig_mapping)

        with torch.inference_mode():
            for i in range(5):
                x = torch.unsqueeze(self.calibdata.input_images[i], dim=0).to(self.dev)
                model_prepared(x)

        model_quantized = torch.quantization.convert(model_prepared)

        self.get_hooks(model_quantized, target_modules=self.target_modules,
                       quantized=True)

        return model_quantized


    def bias_correction(self, model, target_modules):
        i=0
        for _, submodule in model.named_modules():
            if submodule.__class__.__name__ in target_modules:
                bias = self.get_param(submodule, 'bias')
                if bias is not None:
                    self.get_quantized_model(model)
                    f = self.quant_hook_values[i]
                    g = self.hook_values[i]
                    eps = torch.mean(f - g)

                    #print("eps data:", eps)
                    bias.data[::] -= eps
                i += 1

        return model

    def quantize_model(self):
        torch.no_grad()
        m = copy.deepcopy(self.model).to(self.dev)
        m.eval()

        self.get_hooks(self.model, target_modules=self.target_modules)

        model_bias_corr = self.bias_correction(m, target_modules=self.target_modules)
        model_bias_corr.eval()

        model_quantized_bias_corr = quantize_model(model_bias_corr, self.calibdata, self.dev)

        return model_quantized_bias_corr