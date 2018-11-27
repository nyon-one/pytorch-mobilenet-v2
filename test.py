from MobileNetV2 import MobileNetV2
import torch
net = MobileNetV2(n_class=1000)
state_dict = torch.load('mobilenet_v2.pth.tar', map_location='cpu') # add map_location='cpu' if no gpu
net.load_state_dict(state_dict)

print(dir(net))