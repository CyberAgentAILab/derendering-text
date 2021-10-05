import numpy as np
import torch


def torch_to_numpy(arr: torch.Tensor, is_cuda: bool = True) -> np.ndarray:
    if is_cuda:
        return arr.data.cpu().numpy()
    else:
        return arr.data.numpy()


def arr_to_cuda(arr: np.ndarray, is_cuda: bool = True,
                dev: torch.device = None) -> torch.Tensor:
    if is_cuda:
        return torch.from_numpy(arr).to(dev)
    else:
        return torch.from_numpy(arr)
