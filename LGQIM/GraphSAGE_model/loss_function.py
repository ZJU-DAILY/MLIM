import torch
import torch.nn as nn
import torch.nn.functional as F

class ListMLELoss(nn.Module):
    def __init__(self, topk=None):
        super(ListMLELoss, self).__init__()
        self.topk = topk

    def forward(self, preds, labels):
        """
        preds: shape (batch_size, n)
        labels: shape (batch_size, n)
        """

        batch_size, n = preds.size()

        # sort labels descending to get true ranking
        _, sorted_indices = torch.sort(labels, dim=1, descending=True)
        if self.topk is not None:
            sorted_indices = sorted_indices[:, :self.topk]

        # gather predictions according to true ranking
        preds_sorted = torch.gather(preds, dim=1, index=sorted_indices)

        # compute ListMLE loss
        loss = 0.0
        for i in range(preds_sorted.size(0)):  # batch loop
            s = preds_sorted[i]
            log_cumsum = torch.logcumsumexp(s, dim=0)
            loss += torch.sum(log_cumsum - s)

        return loss / batch_size

class CombinedListMLE_MSE_Loss(nn.Module):
    def __init__(self, alpha=0.5, topk=None):
        """
        alpha: weight for ListMLE loss (between 0 and 1)
        topk: if not None, only consider top-k items in ListMLE
        """
        super(CombinedListMLE_MSE_Loss, self).__init__()
        self.alpha = alpha
        self.listmle = ListMLELoss(topk=topk)
        self.mse = nn.MSELoss()

    def forward(self, preds, labels):
        """
        preds, labels: shape (batch_size, n)
        """
        listmle_loss = self.listmle(preds, labels)
        mse_loss = self.mse(preds, labels)
        return self.alpha * listmle_loss + (1 - self.alpha) * mse_loss
