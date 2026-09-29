"""(A) checkpoint: stage-1 encoder + heads with the CRF scalars set to their simulator-derived
values (no CRF learning): l = lik_table.py log-LR tables, stay_bonus 0, c = c_head + log 2
(c_scale 1, c_offset log 2), het and affinity log-priors with weight 1."""
import json, math, sys
import torch
src, table, dst = sys.argv[1:4]
ck = torch.load(src, map_location="cpu", weights_only=False)
t = json.load(open(table))
sd = ck["state_dict"]
sd["tern_dist_loglik"] = torch.tensor(t["tern_dist_loglik"], dtype=torch.float32)
sd["tern_loglik"] = torch.tensor(t["tern_loglik"], dtype=torch.float32)
for k, v in (("stay_bonus", 0.0), ("het_w", 1.0), ("aff_w", 1.0), ("c_scale", 1.0), ("c_offset", math.log(2.0))):
    sd[k] = torch.tensor(v)
ck["optimizer_states"] = []
ck["lr_schedulers"] = []
torch.save(ck, dst)
print("wrote", dst, {k: float(sd[k]) for k in ("stay_bonus", "het_w", "aff_w", "c_scale", "c_offset")})
