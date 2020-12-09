import torch.nn as nn


class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, losses, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - lambdas: (list)各个loss的权重
        """
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.ModuleList(losses)
        self.lambdas = lambdas

    def forward(self, head_fields, head_targets):
        """
        Inputs:
        - head_fields: (list)各个head输出的数据
        - head_targets: (list)各个head对应的gt

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (list) 包含所有loss[loss1, loss2, ...]

        More:
        我感觉head_losses 也可以做成字典, 比如'classification':loss1
        看哪个比较方便
        """
        head_losses = [ll
                        for l, f, t in zip(self.losses, head_fields, head_targets)
                        for ll in l(f, t)]

        assert len(self.lambdas) == len(head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, head_losses

def get_loss(cfg):
    """
    get MultiHeadLoss

    Inputs:
    -cfg: configuration use the loss_name part or 
          function part(like regression classification)

    Returns:
    -loss: (MultiHeadLoss)

    More:
    通过loss的种类调用各种loss,组成一个list
    用这个list调用MultiHeadLoss类
    """
    loss = ...
    return loss

# example
# class L1_Loss(nn.Module)