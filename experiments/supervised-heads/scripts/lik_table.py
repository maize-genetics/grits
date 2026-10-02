"""(A) simulator-derived likelihood table: l[state, band] = log P(state, band | founder is the
row's source) - log P(state, band | founder is not the source), over CLEAN rows (prov_is_clean)
of the TRAIN split of the simulated set (--rows all: every labelled row). 'Source' = founders sharing the ancestral lineage (at
the row) of the true founder of the haplotype the row was read from (row kind 0/2/4 -> hap1,
1/3/5 -> hap2). Bands as GRITSCRFDiploidIndel.dist_band_edges (8 = no anchor). Also the
band-free 3-state table. Add-one smoothing."""
import argparse, json
import numpy as np, torch
from python.crf.train_diploid_indel import individual_split_rows
from python.crf.simulate_alleles import prov_is_clean, PROV_KIND_MASK
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--windows-per-depth", type=int, default=6000)
ap.add_argument("--n-depths", type=int, default=4)
ap.add_argument("--rows", choices=["clean", "all"], default="clean",
                help="clean: provenance-clean rows only (the source founder then always matches, so a "
                     "mismatch is scored near-impossible); all: every labelled row incl. corrupted and "
                     "off-site ones -- the emission then sees reads as they come and the gate is only "
                     "the soft multiplier on top")
a = ap.parse_args()
K, G, T = 25, 117, 512
data = np.load(a.data, mmap_mode="r"); lin = np.load(a.data[:-4] + ".lin.npy", mmap_mode="r")
prov = np.load(a.data[:-4] + ".prov.npy", mmap_mode="r")
tr, _, _ = individual_split_rows(len(data) // G, G, .1, .1, 0)
edges = torch.tensor([0.5, 40.5, 56.5, 72.5, 80.5, 88.5, 104.5])
rng = np.random.default_rng(0)
per = len(data) // a.n_depths
tables = {}
tot_src = np.zeros((3, 9)); tot_non = np.zeros((3, 9))
for dpt in range(a.n_depths):
    rows = tr[(tr >= dpt * per) & (tr < (dpt + 1) * per)]
    rows = np.sort(rng.choice(rows, a.windows_per_depth, replace=False))
    src_c = np.zeros((3, 9)); non_c = np.zeros((3, 9))
    for s in range(0, len(rows), 500):
        r = rows[s:s + 500]
        X = np.asarray(data[r]); L = np.asarray(lin[r]).astype(np.int64); P = np.asarray(prov[r]).astype(np.int64)
        tern = X[..., :K].astype(np.int64); dist = torch.tensor(X[..., K + 2:2 * K + 2].astype(np.float32))
        band = torch.bucketize(dist, edges).numpy(); band[dist.numpy() < 0] = 8
        clean = prov_is_clean(P) if a.rows == "clean" else P >= 0
        hap2 = (P & PROV_KIND_MASK) % 2 == 1
        h = np.where(hap2, X[..., K + 1], X[..., K]).astype(np.int64)
        clean &= h >= 0
        hs = np.clip(h, 0, K - 1)
        lsrc = np.take_along_axis(L, hs[..., None], -1)
        is_src = (L == lsrc) & clean[..., None]
        is_non = (L != lsrc) & clean[..., None]
        st = np.clip(tern, -1, 1) + 1
        np.add.at(src_c, (st[is_src], band[is_src]), 1)
        np.add.at(non_c, (st[is_non], band[is_non]), 1)
    tot_src += src_c; tot_non += non_c
    ps = (src_c + 1) / (src_c + 1).sum(); pn = (non_c + 1) / (non_c + 1).sum()
    tables[f"depth{dpt}"] = (np.log(ps) - np.log(pn)).round(3).tolist()
ps = (tot_src + 1) / (tot_src + 1).sum(); pn = (tot_non + 1) / (tot_non + 1).sum()
lr_dist = np.log(ps) - np.log(pn)
# (state, band) cells never observed for either class carry no simulator evidence: 0 (neutral),
# not the smoothing artifact log((1/Ns)/(1/Nn))
lr_dist[(tot_src == 0) & (tot_non == 0)] = 0.0
s3 = tot_src.sum(1) + 1; n3 = tot_non.sum(1) + 1
lr3 = np.log(s3 / s3.sum()) - np.log(n3 / n3.sum())
res = {"tern_dist_loglik": lr_dist.tolist(), "tern_loglik": lr3.tolist(),
       "per_depth_tern_dist_loglik": tables,
       "counts_src": tot_src.tolist(), "counts_non": tot_non.tolist(),
       "rows": "states -1/0/1 (DEL/DIV/MATCH) x bands 0|1-40|41-56|57-72|73-80|81-88|89-104|>=105|none"}
json.dump(res, open(a.out, "w"), indent=1)
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print("log-LR [state x band]:\n", lr_dist, "\n3-state:", lr3)
print("P(state|src) marginal", (tot_src.sum(1) / tot_src.sum()).round(4), " P(state|non)", (tot_non.sum(1) / tot_non.sum()).round(4))
