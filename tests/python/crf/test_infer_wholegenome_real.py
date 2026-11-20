import types

import numpy as np
import pandas as pd
import torch

from python.crf import infer_wholegenome_real as iwr

NAMES = ["A", "B", "C"]


def test_k_target_to_name_null_state():
    assert iwr.k_target_to_name(0, None, NAMES) == "A"
    assert iwr.k_target_to_name(1, 1, NAMES) == "C"
    assert iwr.k_target_to_name(3, None, NAMES) is None


def test_wholechrom_bed_skips_null_and_uses_markers(tmp_path):
    # pair index p -> (pi[p], pj[p]); pair 2 is the null state (founder index 3)
    model = types.SimpleNamespace(pi=torch.tensor([0, 1, 3]), pj=torch.tensor([0, 1, 3]))
    pred = torch.tensor([0, 0, 2, 0, 1])
    bp = np.array([0, 256, 512, 768, 1024])
    iwr.write_wholechrom_bed("S", {"chr1": (pred, bp)}, model, NAMES, None, tmp_path)
    got = pd.read_csv(tmp_path / "S_chr1_imputed.bed", sep="\t").values.tolist()
    assert got == [["chr1", 0, 257, "A", "A"], ["chr1", 768, 896, "A", "A"],
                   ["chr1", 896, 1025, "B", "B"]]


def test_encode_runs_on_cpu():
    class M(torch.nn.Module):
        def forward(self, X, hs, eb):
            return X, None, X[..., :1]
    feats = torch.randn(10, 4)
    em, cc = iwr._encode_real(M(), feats, [0, 2], 4, None, "cpu", 8, 0.0)
    assert em.dtype == torch.float32 and em.shape == (2, 4, 4)
