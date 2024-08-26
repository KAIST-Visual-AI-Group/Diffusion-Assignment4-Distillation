from typing import Optional
from PIL import Image 
import numpy as np 
import torch


def pil_to_torch(pil_img):
    _np_img = np.array(pil_img).astype(np.float32) / 255.0
    _torch_img = torch.from_numpy(_np_img).permute(2, 0, 1).unsqueeze(0)
    return _torch_img

def torch_to_pil(tensor):
    if tensor.dim() == 4:
        _b, *_ = tensor.shape
        if _b == 1:
            tensor = tensor.squeeze(0)
        else:
            tensor = tensor[0, ...]
    
    tensor = tensor.permute(1, 2, 0)
    np_tensor = tensor.detach().cpu().numpy()
    np_tensor = (np_tensor * 255.0).astype(np.uint8)
    pil_tensor = Image.fromarray(np_tensor)
    return pil_tensor