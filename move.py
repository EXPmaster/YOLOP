import os
from shutil import copy
from tqdm import tqdm
folder_list1 = "/data2/zwt/bdd/bdd100k/images/100k/train"
folder_list3 = "/data2/zwt/bdd/bdd100k/labels/100k/train"
folder_list2 = "/home/zwt/wd/DaChuang/inference/images"

img_list1 = os.listdir(folder_list1)
label_list1 = os.listdir(folder_list3)
print(len(img_list1))
print(len(label_list1))
"""if not os.path.exists(folder_list2):
    os.makedirs(folder_list2)
for img in tqdm(img_list1[:1000],total = 1000):
    from_path = os.path.join(folder_list1,img)
    to_path = os.path.join(folder_list2,img)
    copy(from_path,to_path)"""