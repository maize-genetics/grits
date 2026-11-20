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
        self._resume = None
        self._module = None

    def state_dict(self):
        # A checkpoint written during validation holds the EMA weights as the model weights,
        # while the optimizer moments belong to the raw weights. Keep the raw weights and the
        # shadow here so --resume continues from the raw weights the optimizer state matches.
        if self.shadow is None:
            return {}
        raw = self._backup
        if raw is None:
            raw = {k: v for k, v in self._module.state_dict().items() if k in self.shadow}
        return {"shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
                "raw": {k: v.detach().cpu() for k, v in raw.items()}}

    def load_state_dict(self, state_dict):
        self._resume = state_dict or None

    @torch.no_grad()
    def on_train_start(self, trainer, pl_module):
        self._module = pl_module
        if self._resume:
            sd = pl_module.state_dict()
            for k, v in self._resume["raw"].items():
                sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
            self.shadow = {k: v.to(sd[k].device).float() for k, v in self._resume["shadow"].items()}
            self._resume = None
            return
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
