import numpy as np
import torch

from python.crf.crf_kernels import _dcrf_nll
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, _dcrf_nll_tied


def _model(K=5):
    return GRITSCRFDiploidIndel(num_parents=K, d_model=16, n_heads=2, n_layers=1, tie_aware_loss=True)


def test_tied_nll_equals_standard_when_only_label_allowed():
    torch.manual_seed(0)
    m = _model()
    B, T = 3, 7
    emis = torch.randn(B, T, m.P)
    c = torch.rand(B, T)
    tags = torch.randint(0, m.P, (B, T))
    allowed = torch.zeros(B, T, m.P, dtype=torch.bool)
    allowed.scatter_(2, tags.unsqueeze(-1), True)
    a = _dcrf_nll(emis, c, m.nsw_pair, m.stay_bonus, tags)
    b = _dcrf_nll_tied(emis, c, m.nsw_pair, m.stay_bonus, allowed)
    assert torch.allclose(a, b, atol=1e-4)
    everything = torch.ones(B, T, m.P, dtype=torch.bool)
    assert abs(float(_dcrf_nll_tied(emis, c, m.nsw_pair, m.stay_bonus, everything))) < 1e-4


def test_allowed_pairs_follow_lineage():
    m = _model(K=5)
    lin = torch.tensor([[[0, 0, 1, 2, 2]]])            # founders 0,1 share a lineage; 3,4 share another
    h1, h2 = torch.tensor([[0]]), torch.tensor([[3]])
    al = m._allowed_pairs(h1, h2, lin)[0, 0]
    got = {(int(m.pi[p]), int(m.pj[p])) for p in torch.nonzero(al).ravel()}
    assert got == {(0, 3), (0, 4), (1, 3), (1, 4)}
    # null label is equivalent only to itself
    al = m._allowed_pairs(torch.tensor([[5]]), torch.tensor([[2]]), lin)[0, 0]
    got = {(int(m.pi[p]), int(m.pj[p])) for p in torch.nonzero(al).ravel()}
    assert got == {(2, 5)}


def test_mixture_pair_emission():
    import math
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=5, d_model=16, n_heads=2, n_layers=1, pair_emission="mixture").eval()
    ms = GRITSCRFDiploidIndel(num_parents=5, d_model=16, n_heads=2, n_layers=1).eval()
    ms.load_state_dict(m.state_dict())
    X = torch.stack([torch.randint(-1, 2, (2, 6, 5)), torch.randint(0, 100, (2, 6, 5))], -1).float()
    with torch.no_grad():
        ep, _, _ = m(X)
        es, _, _ = ms(X)
    homo = (m.pi == m.pj)
    # homozygous states: mixture = e_i, sum = 2 e_i
    assert torch.allclose(ep[..., homo] * 2, es[..., homo], atol=1e-5)
    i, j = int(m.pi[~homo][0]), int(m.pj[~homo][0])
    p = int(torch.nonzero(~homo)[0])
    ei = (es[..., (m.pi == i) & (m.pj == i)].squeeze(-1)) / 2
    ej = (es[..., (m.pi == j) & (m.pj == j)].squeeze(-1)) / 2
    assert torch.allclose(ep[..., p], torch.logaddexp(ei, ej) - math.log(2.0), atol=1e-5)


def test_likelihood_emission_uses_only_input_gate():
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=5, d_model=16, n_heads=2, n_layers=1,
                             emission="likelihood", pair_emission="mixture").eval()
    X = torch.stack([torch.randint(-1, 2, (2, 6, 5)), torch.randint(0, 100, (2, 6, 5))], -1).float()
    with torch.no_grad():
        ep, g, _ = m(X)
    homo = m.pi == m.pj
    for f in range(5):
        p = int(torch.nonzero(homo & (m.pi == f))[0])
        want = g * m.tern_loglik[(X[..., f, 0] + 1).long()]
        assert torch.allclose(ep[..., p], want, atol=1e-5)
    # two founders with identical ternary columns get identical emissions (no founder identity)
    X2 = X.clone(); X2[..., 1, :] = X2[..., 0, :]
    with torch.no_grad():
        ep2, _, _ = m(X2)
    p0 = int(torch.nonzero(homo & (m.pi == 0))[0]); p1 = int(torch.nonzero(homo & (m.pi == 1))[0])
    assert torch.allclose(ep2[..., p0], ep2[..., p1], atol=1e-6)


def test_train_homo_scale_marks_inbreds_zero():
    from python.crf.train_diploid_indel import IndelDiploidAffinityDataset
    K, T, G = 5, 8, 2
    d = np.zeros((4, T, 3 * K + 2), np.int8)
    d[:, :, 0] = 1
    d[:2, :, K] = 1; d[:2, :, K + 1] = 1          # individual 0: homozygous
    d[2:, :, K] = 1; d[2:, :, K + 1] = 3          # individual 1: hybrid
    ds = IndelDiploidAffinityDataset(d, K, G, train_homo_scale=True)
    assert float(ds[0]["homo_scale"]) == 0.0 and float(ds[3]["homo_scale"]) == 1.0
    assert "homo_scale" not in IndelDiploidAffinityDataset(d, K, G)[0]


def test_likelihood_dist_emission_bands():
    torch.manual_seed(0)
    m = GRITSCRFDiploidIndel(num_parents=4, d_model=16, n_heads=2, n_layers=1,
                             emission="likelihood_dist", pair_emission="mixture").eval()
    with torch.no_grad():
        m.tern_dist_loglik.copy_(torch.arange(27, dtype=torch.float32).view(3, 9))
    codes = torch.tensor([0., 20., 50., 60., 75., 85., 100., 120.])        # bands 0..7
    X = torch.zeros(1, 8, 4, 2)
    X[0, :, 0, 0] = 1; X[0, :, 0, 1] = codes                               # founder 0: match, all bands
    X[0, :, 1, 0] = -1; X[0, :, 1, 1] = -1                                  # founder 1: deleted, no anchor
    with torch.no_grad():
        ep, g, _ = m(X)
    homo = m.pi == m.pj
    p0 = int(torch.nonzero(homo & (m.pi == 0))[0]); p1 = int(torch.nonzero(homo & (m.pi == 1))[0])
    assert torch.allclose(ep[0, :, p0], g[0] * torch.arange(18, 26, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(ep[0, :, p1], g[0] * 8.0, atol=1e-5)
