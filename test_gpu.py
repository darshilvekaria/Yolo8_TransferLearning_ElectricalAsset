import torch
import torchvision
from torchvision.ops import nms

# Example inputs for NMS
boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.85, 0.95])

# Check if CUDA is available, then move tensors to the GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    boxes = boxes.to(device)
    scores = scores.to(device)
    print("Running NMS on GPU...")
else:
    device = torch.device("cpu")
    print("Running NMS on CPU...")

# Run NMS
keep = nms(boxes, scores, 0.5)

# Print results
print(f"Indices of boxes kept: {keep}")
