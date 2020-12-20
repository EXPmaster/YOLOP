import time


def train(config, train_loader, model, criterion, optimizer, epoch,
          writer_dict, logger, rank=-1):
    """
    train for one epoch

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:

    Returns:
    None
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    start = time.time()
    for i, (input, target, meta) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        outputs = model(input)
        target = target.cuda(non_blocking=True)
        
        if isinstance(outputs, list):
            total_loss, head_losses = criterion(outputs[0], target)
            for output in outputs[1:]:
                total_loss += criterion(output, target)
        else:
            output = outputs
            total_loss, head_losses = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
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


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
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
    pass


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