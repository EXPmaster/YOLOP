import time
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from visualization import plot_img_and_mask,plot_one_box
import torch
from threading import Thread
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import json
import random
import cv2
import os
from torch.cuda import amp


def train(config, train_loader, model, criterion, optimizer, scaler, epoch,
          writer_dict, logger, device, rank=-1):
    """
    train for one epoch

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
    output[1] len:1, [2,256,256]
    target(2,)
    target[0] [1,n,5]
    target[1] [2,256,256]
    Returns:
    None

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (input, target, _, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        if not config.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, model)

        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))

            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1



def validate(epoch,config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, logger=None, device='cpu', rank=-1):
    """
    validata

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None
    save_dir = output_dir + os.path.sep + 'visualization'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print(save_dir)
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE] #imgsz is multiple of max_stride
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = True
    is_coco = False #is coco dataset
    save_conf=False # save auto-label confidences
    verbose=False
    save_hybrid=False
    log_imgs,wandb = min(16,100), None
    tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((720, 1280)),
                transforms.ToTensor()
            ]
        )

    nc = 13
    iouv = torch.linspace(0.5,0.95,10).to(device)     #iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen =  0 
    confusion_matrix = ConfusionMatrix(nc=model.nc) #detector confusion matrix
    metric = SegmentationMetric(2) #segment confusion matrix
    metric.reset()    

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()
    acc_seg = AverageMeter()
    meanAcc_seg = AverageMeter()
    mIoU_seg = AverageMeter()
    FWIoU_seg = AverageMeter()

    # switch to train mode
    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, target, paths, shapes) in enumerate(val_loader):
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            t = time_synchronized()
            det_out, seg_out= model(img)
            inf_out,train_out = det_out
            t_inf += time_synchronized() - t

            #segment evaluation
            _,predict=torch.max(seg_out, 1)
            _,gt=torch.max(target[1], 1)
            predict = predict[:,56:200,:]
            gt = gt[:,56:200,:]
            metric.addBatch(predict.cpu(), gt.cpu())
            acc = metric.pixelAccuracy()
            meanAcc = metric.meanPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

            acc_seg.update(acc,img.size(0))
            meanAcc_seg.update(meanAcc,img.size(0))
            mIoU_seg.update(mIoU,img.size(0))
            FWIoU_seg.update(FWIoU,img.size(0))


        
            if training:
                total_loss, head_losses = criterion((train_out,seg_out), target, model)   #Compute loss
                losses.update(total_loss.item(), img.size(0))

            #NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6, labels=lb)
            #output = non_max_suppression(inf_out, conf_thres=0.001, iou_thres=0.6)
            #output = non_max_suppression(inf_out, conf_thres=config.TEST.NMS_CONF_THRES, iou_thres=config.TEST.NMS_IOU_THRES)
            t_nms += time_synchronized() - t

            #可视化
            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        img_path = Path(paths[i])
                        img_test = Image.open(img_path)
                        seg_mask = predict[i].float()
                        seg_mask = tf(seg_mask.cpu())
                        seg_mask = seg_mask.squeeze().cpu().numpy()
                        seg_mask = seg_mask > 0.5
                        # print(seg_mask.shape)
                        plot_img_and_mask(img_test, seg_mask, i,epoch,save_dir)

                        img_det = cv2.imread(paths[i])
                        img_gt = img_det.copy()
                        det = output[i]
                        if len(det):
                            det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(output[i]):
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_pred.png".format(epoch,i),img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        # print(labels)
                        labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                        if len(labels):
                            labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                        for cls,x1,y1,x2,y2 in labels:
                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1,y1,x2,y2)
                            plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_gt.png".format(epoch,i),img_gt)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if config.TEST.SAVE_JSON:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):   #用于去重，图片有多少类
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, target[0], paths, f, names), daemon=True).start()
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions
            #Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=config.TEST.PLOTS, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    #print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    segment_result = (acc_seg.avg,meanAcc_seg.avg,mIoU_seg.avg,FWIoU_seg.avg)

    #print segmet_result
    return segment_result,(mp, mr, map50, map, losses.avg), maps, t
        


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0