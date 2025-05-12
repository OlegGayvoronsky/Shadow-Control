import torch
import tensorflow as tf

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
