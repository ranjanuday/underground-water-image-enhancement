from model import UIE_Net
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch

input_tensor = torch.randn(1, 3, 256, 256)  # batch_size=1, 3 channels, 256x256
model = UIE_Net()


# Model architecture
print(model)
with open('model.txt', 'w') as f:
    f.write(repr(model))
    
# FLOPs calculation
flops = FlopCountAnalysis(model, input_tensor)
print(flop_count_table(flops))

total_flops = flops.total()
print("Total FLOPs:", total_flops)

# Parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total params:", total_params)
print("Trainable params:", trainable_params)


# Save to file
with open('flops.txt', 'w') as f:
    f.write(f"Total FLOPs: {total_flops}\n")
    f.write(f"Total No. of Parameters: {total_params}\n")
    f.write(f"Total No. of Trainable Parameters: {trainable_params}\n")
    f.write(flop_count_table(flops))