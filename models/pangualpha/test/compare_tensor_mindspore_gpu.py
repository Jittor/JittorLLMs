import torch

def compare(a):
    a = a.cpu()
    import numpy as np
    b = np.load('/userhome/tmp/tmp.npy')
    b = torch.Tensor(b).view_as(a)
    res = torch.max(torch.abs(torch.sub(a,b)))
    return res
