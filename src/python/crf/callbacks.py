"""
Shared Lightning callbacks for the CRF training scripts (train_haploid.py,
train_diploid.py). Extracted so both entry points can reuse them without
importing each other's (heavier) training module.
"""

import torch
import pytorch_lightning as pl


class EMACallback(pl.Callback):
    """Exponential moving average of the weights. The EMA is swapped into the model
    for every validation pass (so the monitored val metric — what ModelCheckpoint
    selects on — and the saved checkpoint both reflect the averaged weights) and
    swapped back out when training resumes. Targets late-epoch oscillation directly:
    even when the raw weights spike out of the basin, their moving average stays in
    it, so the averaged decode is far steadier than any single late-epoch snapshot."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = None
        self._backup = None

    def on_train_start(self, trainer, pl_module):
        self.shadow = {k: v.detach().clone().float()
                       for k, v in pl_module.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *a):
        if self.shadow is None:
            return
        d = self.decay
        for k, v in pl_module.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module):
        if self.shadow is None or self._backup is not None:
            return                                          # skip pre-train sanity val
        self._backup = {k: v.detach().clone()
                        for k, v in pl_module.state_dict().items() if k in self.shadow}
        for k, v in pl_module.state_dict().items():
            if k in self.shadow:
                v.copy_(self.shadow[k].to(v.dtype))

    @torch.no_grad()
    def on_train_batch_start(self, trainer, pl_module, *a):
        if self._backup is None:                            # restore raw weights once
            return
        sd = pl_module.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None
