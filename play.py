import torch
from torch.utils.data import DataLoader
from itertools import chain

# Define your generator function
def generator():
    for i in range(10):
        yield i

# Create a generator dataloader
generator_dataloader = DataLoader(generator(), batch_size=2)

# Create a normal dataloader
normal_dataloader = DataLoader([10, 20, 30, 40, 50], batch_size=2)

# Combine the two dataloaders
combined_dataloader = chain(generator_dataloader, normal_dataloader)

# Iterate over the combined dataloader
for batch in combined_dataloader:
    print(batch)
