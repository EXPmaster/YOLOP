## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask, index):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig('/home/zwt/DaChuang/visualization/{}.png'.format(index))


if __name__ == "__main__":
    pass
# def plot():
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU) if not cfg.DEBUG \
#         else select_device(logger, 'cpu')

#     if args.local_rank != -1:
#         assert torch.cuda.device_count() > args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device('cuda', args.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
#     model = get_net(cfg).to(device)
#     model_file = '/home/zwt/DaChuang/weights/epoch--2.pth'
#     checkpoint = torch.load(model_file)
#     model.load_state_dict(checkpoint['state_dict'])
#     if rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
#     if rank != -1:
#         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)