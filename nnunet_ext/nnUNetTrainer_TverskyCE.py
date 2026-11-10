"""Tversky + CE trainer (500 epochs).

RECONSTRUCTED 2026-08-27. The original file this symlink pointed to was lost, which
left a dangling symlink inside the nnunetv2 package; nnU-Net imports every module
under training/nnUNetTrainer when resolving a trainer name, so *all* inference with
this environment crashed with ModuleNotFoundError. This restores a working class of
the same name. Architecture (the only thing inference depends on) is unchanged;
the loss is a faithful re-implementation, so re-training would not be bit-identical
to the checkpoints already in nnUNet_trained_models.
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1


class TverskyLoss(SoftDiceLoss):
    """Dice generalised with asymmetric FP/FN weights (alpha=FP, beta=FN)."""

    def __init__(self, alpha=0.3, beta=0.7, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta = alpha, beta

    def forward(self, x, y, loss_mask=None):
        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = list(range(2, x.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        if not self.do_bg:
            tp, fp, fn = tp[:, 1:], fp[:, 1:], fn[:, 1:]
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return -tversky.mean()


class Tversky_and_CE_loss(DC_and_CE_loss):
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1, weight_tversky=1,
                 ignore_label=None):
        super().__init__(tversky_kwargs, ce_kwargs, weight_ce, weight_tversky,
                         ignore_label=ignore_label, dice_class=SoftDiceLoss)
        self.dc = TverskyLoss(apply_nonlin=softmax_helper_dim1, **tversky_kwargs)


class nnUNetTrainer_TverskyCE(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500

    def _build_loss(self):
        loss = Tversky_and_CE_loss(
            {'alpha': 0.3, 'beta': 0.7, 'batch_dice': self.configuration_manager.batch_dice,
             'do_bg': False, 'smooth': 1e-5},
            {}, weight_ce=1, weight_tversky=1,
            ignore_label=self.label_manager.ignore_label)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            import numpy as np
            dr = self.configuration_manager.pool_op_kernel_sizes
            w = np.array([1 / (2 ** i) for i in range(len(dr))])
            w[-1] = 0
            w = w / w.sum()
            loss = DeepSupervisionWrapper(loss, w)
        return loss
