import torch

fr = torch.range(0, 9)
fr2 = torch.range(0, 19)
grid_fr = torch.meshgrid(fr, fr, fr2, fr2)

print(grid_fr)

