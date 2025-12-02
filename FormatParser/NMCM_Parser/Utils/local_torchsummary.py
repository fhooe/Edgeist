# code coppied from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                summary[m_key]["weights"] = {}
                summary[m_key]["weights"]["data"] = module.weight.data
                summary[m_key]["weights"]["amount"] = module.weight.numel()
                summary[m_key]["weights"]["dtype_size"] = module.weight.element_size()
                summary[m_key]["weights"]["dtype"] = module.weight.dtype
                summary[m_key]["weights"]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                summary[m_key]["bias"] = {}
                summary[m_key]["bias"]["data"] = module.bias.data
                summary[m_key]["bias"]["amount"] = module.bias.numel()
                summary[m_key]["bias"]["dtype_size"] = module.bias.element_size()
                summary[m_key]["bias"]["dtype"] = module.bias.dtype
                summary[m_key]["bias"]["trainable"] = module.bias.requires_grad


            # running_mean and running_var for BatchNorm layers
            if hasattr(module, "running_mean"):
                summary[m_key]["running_mean"] = module.running_mean.detach().cpu().numpy().tolist()
            if hasattr(module, "running_var"):
                summary[m_key]["running_var"] = module.running_var.detach().cpu().numpy().tolist()

            if hasattr(module, "dilation"):
                summary[m_key]["dilation"] = module.dilation
            if hasattr(module, "groups"):
                summary[m_key]["groups"] = module.groups
            if hasattr(module, "kernel_size"):
                summary[m_key]["kernel_size"] = module.kernel_size
            if hasattr(module, "padding"):
                summary[m_key]["padding"] = module.padding
            if hasattr(module, "stride"):
                summary[m_key]["stride"] = module.stride

            # Dropout rate for Dropout layers
            if hasattr(module, "p"):
                summary[m_key]["DropoutRate"] = module.p

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # remove model it self
    summary.popitem()

    # return summary
    return summary