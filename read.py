import torch
checkpoint = torch.load('/home/zwt/wd/DaChuang/epoch-135.pth')
for k,v in checkpoint['state_dict'].items():
    print(k)