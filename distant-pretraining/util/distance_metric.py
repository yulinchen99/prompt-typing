# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

def js_div(p, q):
    kldiv = nn.KLDivLoss(reduction='none')
    log_mean = ((p+q) / 2).log()
    sim = (kldiv(log_mean, p)+kldiv(log_mean, q)) / 2
    sim = sim.sum(1)
    return sim