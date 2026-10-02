import numpy as np
a = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3_regionfix_inb1.0.npy")
b = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3_regionfix_inb0.0.npy")
ra = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3_regionfix_inb1.0.refpos.npy")
rb = np.load("/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3_regionfix_inb0.0.refpos.npy")
mixed = np.concatenate([a, b], axis=0)
refpos_mixed = np.concatenate([ra, rb], axis=0)
out_path = "/workdir/zrm22/HackathonJun2026/grits_workdir/indel_baseline/data/training/maize_v3_regionfix_mixed.npy"
np.save(out_path, mixed)
np.save(out_path.replace(".npy", ".refpos.npy"), refpos_mixed)
print("wrote", out_path, mixed.shape)
