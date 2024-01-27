import torch


model_state = torch.rand(1, 14, 800)
param = torch.rand(1, 2, 600)

min_shape = [min(param.shape[i], model_state.shape[i]) for i in range(len(param.shape))]
idxs = torch.meshgrid(*[torch.arange(s) for s in min_shape])
model_state[tuple(idxs)].copy_(param[tuple(idxs)])    

