import torch
import numpy as np
import random

SEED = 42

def set_seed(SEED = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

'''def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device'''


