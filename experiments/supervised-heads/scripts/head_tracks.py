"""Atlas-style genome tracks for supervised-heads checkpoints on real 0.1x rows (Ia453, A188xEP1,
B73xOh43): gate g, switch probability p = sigmoid(-c_head), het h, decoded switches and
SNP+RefCall error, binned to 1 Mb along the concatenated autosomes -> tracks_<sample>.npz (plot with plot_tracks.py).
Usage: head_tracks.py <device> <outdir> name=ckpt [name=ckpt ...]"""
import contextlib, io, sys
from pathlib import Path
WT = str(Path(__file__).resolve().parents[3] / "src")
sys.path.insert(0, WT)
from python.crf.train_diploid_indel import GRITSCRFDiploidIndel, infer_real_founder_pairs  # noqa: E402  (first: module cache)
import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, "/home/zrm22/.claude/jobs/ec1f8138/tmp/viz")
import model_internals as mi  # noqa: E402
import window_truth as wt  # noqa: E402

K, T, BIN = 25, 512, 1_000_000
dev, outd = sys.argv[1], Path(sys.argv[2])
outd.mkdir(parents=True, exist_ok=True)
models = dict(a.split("=", 1) for a in sys.argv[3:])
fss, hae = mi.fss, mi.hae
panel = fss.Panel(); panel.fa = np.asarray(panel.fa)
for sname, (dname, p1, p2) in mi.SAMPLES.items():
    dd = mi.EV / dname
    data = np.load(dd / "windowed_k25native_wcount.npy")
    contig, pos, rb = mi.layout_rows(dd / "raw.npy.bins.tsv", data.shape[0])
    names = [l.split("\t")[1].strip() for l in open(dd / "raw.npy.gametes.tsv")][1:]
    t1, t2 = fss.load_truth(p1), fss.load_truth(p2)
    chroms = [c for c in fss.AUTOSOMES if (contig == c).any()]
    off, o = {}, 0
    for c in chroms:
        off[c] = o; o += int(pos[contig == c].max()) + 1
    gpos = np.array([off[c] for c in contig]) + pos
    gb = gpos // BIN; nb = int(gb.max()) + 1
    store = {"chrom_offsets_mb": np.array([off[c] / BIN for c in chroms]), "chroms": np.array(chroms)}
    for mname, ck in models.items():
        m = GRITSCRFDiploidIndel.load_from_checkpoint(ck, map_location="cpu", strict=False).to(dev).eval()
        tern = data[:, :, :K].astype(np.float32); dist = data[:, :, K + 2:2 * K + 2].astype(np.float32)
        feats = torch.tensor(np.stack([tern, dist], -1)); count = torch.tensor(data[:, :, 2 * K + 2:3 * K + 2].astype(np.float32))
        with contextlib.redirect_stdout(io.StringIO()):
            lo, hi = infer_real_founder_pairs(m, data, K, device=dev)
        G, P, Hh = [], [], []
        eb = torch.zeros(1, K, 2)
        with torch.no_grad():
            for s in range(0, len(feats), 256):
                xb = feats[s:s + 256].to(dev)
                m(xb, None, eb.expand(xb.shape[0], -1, -1).to(dev), count[s:s + 256].to(dev))
                h = m._heads
                G.append(torch.sigmoid(h["gate_logit"]).float().cpu().numpy())
                P.append(torch.sigmoid(-h["c"]).float().cpu().numpy())
                Hh.append(torch.sigmoid(h["het_logit"]).float().cpu().numpy())
        G, P, Hh = [np.concatenate(x).ravel() for x in (G, P, Hh)]
        sw = np.zeros(lo.shape, bool); sw[:, 1:] = (lo[:, 1:] != lo[:, :-1]) | (hi[:, 1:] != hi[:, :-1])
        bed = outd / "beds" / mname / sname
        bed.mkdir(parents=True, exist_ok=True)
        for f in bed.glob("*.bed"):
            f.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            hae.write_imputed_bed(sname, lo, hi, None, names, dd / "raw.npy.bins.tsv", bed, bin_size=256)
        f1, f2 = fss.pair_from_bed(panel, bed)
        srows, smatch = wt.site_match(panel, t1, t2, f1, f2)
        n = np.bincount(gb, minlength=nb).astype(float); n[n == 0] = np.nan
        x = np.arange(nb)
        store[f"{mname}/gate"] = np.bincount(gb, G, nb) / n
        store[f"{mname}/p_switch_row"] = np.bincount(gb, P, nb) / n
        store[f"{mname}/het"] = np.bincount(gb, Hh, nb) / n
        store[f"{mname}/switches"] = np.bincount(gb, sw.ravel().astype(float), nb)
        sc = np.array([off.get(fss.AUTOSOMES[ci], 0) for ci in panel.chrom[srows]]) + panel.pos[srows]
        sb = sc // BIN; keep = sb < nb
        ne = np.bincount(sb[keep], (~smatch[keep]).astype(float), nb); ns = np.bincount(sb[keep], minlength=nb).astype(float)
        ns[ns == 0] = np.nan
        store[f"{mname}/snprc_err_pct"] = 100 * ne / ns
        print(sname, mname, "snprc", round(100 * float(1 - smatch.mean()), 3), "g_med", round(float(np.median(G)), 3),
              "p_med", float(np.median(P)), "h_mean", round(float(Hh.mean()), 3), "switches", int(sw.sum()), flush=True)
        del m; torch.cuda.empty_cache()
    np.savez_compressed(outd / f"tracks_{sname}.npz", title=f"{sname} ({dname})", **store)
print("done")
