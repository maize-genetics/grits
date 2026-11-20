import torch
import torch.nn as nn

from python.crf.callbacks import EMACallback


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(3))


def test_ema_state_roundtrip_restores_raw_weights_and_shadow():
    m = _M()
    cb = EMACallback(decay=0.5)
    cb.on_train_start(None, m)
    with torch.no_grad():
        m.w.fill_(4.0)
    cb.on_train_batch_end(None, m)                     # shadow = 2.0
    cb.on_validation_start(None, m)                    # EMA swapped in: model = 2.0, raw backed up
    assert torch.allclose(m.w, torch.full((3,), 2.0))
    state = cb.state_dict()                            # checkpoint written now
    assert torch.allclose(state["raw"]["w"], torch.full((3,), 4.0))
    assert torch.allclose(state["shadow"]["w"], torch.full((3,), 2.0))

    m2 = _M()
    with torch.no_grad():
        m2.w.fill_(2.0)                                # what --resume loads as model weights (EMA)
    cb2 = EMACallback(decay=0.5)
    cb2.load_state_dict(state)
    cb2.on_train_start(None, m2)
    assert torch.allclose(m2.w, torch.full((3,), 4.0))  # raw weights, matching the optimizer state
    assert torch.allclose(cb2.shadow["w"], torch.full((3,), 2.0))


def test_ema_state_outside_validation_uses_current_weights():
    m = _M()
    cb = EMACallback()
    assert cb.state_dict() == {}
    cb.on_train_start(None, m)
    with torch.no_grad():
        m.w.fill_(1.0)
    assert torch.allclose(cb.state_dict()["raw"]["w"], torch.ones(3))
