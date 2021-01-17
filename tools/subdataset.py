import os
import shutil

old_root_path = "/workspace/wh/projects/DaChuang/bdd"
old_folders=["/bdd_seg_gt","/images","/labels/100k"]
new_root_path = "/workspace/wh/projects/DaChuang/subbdd"
folders = ["/bdd_seg_gt","/images","/labels"]
sub_folders = ["/train","/val"]

train_list=os.listdir(old_root_path+old_folders[0]+sub_folders[0])[:192]
val_list= os.listdir(old_root_path+old_folders[0]+sub_folders[1])[:192]

for folder in folders:
        for sub_folder in sub_folders:
                if not os.path.exists(new_root_path+folder+sub_folder):
                        os.makedirs(new_root_path+folder+sub_folder)

for file_train,file_val in zip(train_list,val_list):
        src = os.path.join(old_root_path+old_folders[0]+sub_folders[0],file_train)
        dst = os.path.join(new_root_path+folders[0]+sub_folders[0],file_train)
        shutil.copyfile(src, dst)
        src = os.path.join(old_root_path+old_folders[0]+sub_folders[1],file_val)
        dst = os.path.join(new_root_path+folders[0]+sub_folders[1],file_val)
        shutil.copyfile(src, dst)

for file_train,file_val in zip(train_list,val_list):
        file_train=file_train.replace(".png",".jpg")
        file_val=file_val.replace(".png",".jpg")
        src = os.path.join(old_root_path+old_folders[1]+sub_folders[0],file_train)
        dst = os.path.join(new_root_path+folders[1]+sub_folders[0],file_train)
        shutil.copyfile(src, dst)
        src = os.path.join(old_root_path+old_folders[1]+sub_folders[1],file_val)
        dst = os.path.join(new_root_path+folders[1]+sub_folders[1],file_val)
        shutil.copyfile(src, dst)

for file_train,file_val in zip(train_list,val_list):
        file_train=file_train.replace(".png",".json")
        file_val=file_val.replace(".png",".json")
        src = os.path.join(old_root_path+old_folders[2]+sub_folders[0],file_train)
        dst = os.path.join(new_root_path+folders[2]+sub_folders[0],file_train)
        shutil.copyfile(src, dst)
        src = os.path.join(old_root_path+old_folders[2]+sub_folders[1],file_val)
        dst = os.path.join(new_root_path+folders[2]+sub_folders[1],file_val)
        shutil.copyfile(src, dst)
