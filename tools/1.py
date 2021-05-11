import torch

"""def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1]) #(x2-x1)*(y2-y1)

box1 = torch.tensor([[1,1,3,3],[3,3,5,5]])
box2 = torch.tensor([[2,2,3,3],[4,4,5,5],[4,4,6,6]])

area1 = box_area(box1.T)
area2 = box_area(box2.T)

print(area1)
print(area2)

inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
print(inter)
#iou = inter / (area1[:, None] + area2 - inter)
print(area1[:, None] + area2 - inter)"""

tcls_tensor = torch.tensor([1,1,2,3,6,4,5,4])
for cls in torch.unique(tcls_tensor):                    
    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
    print(ti)
