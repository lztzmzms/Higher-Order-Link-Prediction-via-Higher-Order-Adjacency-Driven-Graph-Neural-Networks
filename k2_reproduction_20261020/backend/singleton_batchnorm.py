                                                                                         
import torch
from torch.nn import functional as F


def install():
    original = torch.nn.BatchNorm1d.forward
    stats = {'enabled': True, 'fallback_calls': 0, 'kind': 'singleton_training_running_statistics'}

    def forward(layer, value):
        if layer.training and value.dim() in (2, 3) and value.size(1) and value.numel() // value.size(1) == 1:
            if layer.running_mean is None or layer.running_var is None:
                raise RuntimeError('Singleton BatchNorm repair requires tracked running statistics')
            stats['fallback_calls'] += 1
            return F.batch_norm(value, layer.running_mean, layer.running_var,
                                layer.weight, layer.bias, training=False, momentum=0., eps=layer.eps)
        return original(layer, value)

    torch.nn.BatchNorm1d.forward = forward

    def restore():
        torch.nn.BatchNorm1d.forward = original

    return stats, restore
