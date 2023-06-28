import gc
import torch

def clear():
    gc.collect()

    torch.cuda.empty_cache()