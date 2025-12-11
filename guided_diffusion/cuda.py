# 1. 检查 CUDA 可用性
import torch

print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(torch.version.cuda)

if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name()}")