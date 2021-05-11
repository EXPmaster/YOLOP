import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages
from lib.core.general import non_max_suppression, scale_coords
from visualization import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_components_analysis, fitlane, fillHole
from tqdm import tqdm
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

"""tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((720, 1280)),
                transforms.ToTensor()
            ]
        )"""

def tf(img,h,w):
    img = transforms.ToPILImage()(img)
    img = transforms.Resize((h, w))(img)
    img = transforms.ToTensor()(img)
    return img

def detect(cfg,opt):

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    #device = torch.device('cpu')
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(opt.source, img_size=opt.img_size)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print(colors)

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        img = transform(img).to(device)
        #print(img.shape)
        #print(img_det.shape)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        #print("model inference time: " +str(t2-t1))

        inf_out,train_out = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)
        t4 = time_synchronized()
        #print("non max suppression: "+str(t4-t3))
        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        #img_seg = img_det.copy()
        save_path = str(opt.save_dir +'/'+ Path(path).name)

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # print(da_seg_out.shape)
        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        # da_seg_mask = fillHole(da_seg_mask)
        #da_seg_mask_post = connect_components_analysis(da_seg_mask)
        """labels = da_seg_mask_post[1]
        stats = da_seg_mask_post[2]
        for index, stat in enumerate(stats):
            # print(stat[4])
            if stat[4] <= 400:
                # print(stat[4])
                idx = np.where(labels == index)
                da_seg_mask[idx] = 0"""

        # img_det = show_seg_result(img_det, da_seg_mask, _, _)

        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        ll_seg_mask = fitlane(ll_seg_mask)
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                #plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        else:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            # print(img_det.shape)
            vid_writer.write(img_det)

        #save result for Decision making system
        txt_folder =  str(opt.save_dir +'/txt/')
        da_binary_folder = str(opt.save_dir +'/da_binary_img/')
        ll_binary_folder = str(opt.save_dir +'/ll_binary_img/')
        raw_folder = str(opt.save_dir +'/raw_img/')
        if not os.path.exists(txt_folder):
            os.makedirs(txt_folder)
        if not os.path.exists(da_binary_folder):
            os.makedirs(da_binary_folder)
        if not os.path.exists(ll_binary_folder):
            os.makedirs(ll_binary_folder)
        if not os.path.exists(raw_folder):
            os.makedirs(raw_folder)

        txt_path = str(opt.save_dir +'/txt/'+ Path(path).name.split(".")[0]+ str(i)+".txt")
        da_binary_img_path = str(opt.save_dir +'/da_binary_img/'+ Path(path).name.split(".")[0]+ str(i)+".png")
        ll_binary_img_path = str(opt.save_dir +'/ll_binary_img/'+ Path(path).name.split(".")[0]+ str(i)+".png")
        raw_img_path= str(opt.save_dir +'/raw_img/'+ Path(path).name.split(".")[0]+ str(i)+".jpg")
        with open(txt_path, 'w') as f:
            if len(det):
                for *xyxy,conf,cls in reversed(det): 
                    f.write(str(int(xyxy[0].cpu().numpy()))+" "+str(int(xyxy[1].cpu().numpy()))+" "+str(int(xyxy[2].cpu().numpy()))+" "+str(int(xyxy[3].cpu().numpy()))+" "+str(conf.cpu().numpy())+" "+str(int(cls.cpu().numpy())))
                    f.write('\n')
        imageio.imwrite(da_binary_img_path, da_seg_mask)
        imageio.imwrite(ll_binary_img_path, ll_seg_mask)
        cv2.imwrite(raw_img_path,img_det)
        
    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/home/zwt/wd/DaChuang/epoch-68.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='/data2/zwt/origin/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/prbmap', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
