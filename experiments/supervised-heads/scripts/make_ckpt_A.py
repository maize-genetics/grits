"""(A) checkpoint: stage-1 encoder + heads with the CRF scalars set to their simulator-derived
values (no CRF learning): l = lik_table.py log-LR tables, stay_bonus 0, switch probability
p_t = lambda_w/(T-1) (xo_scale 1; --placement: redistributed by the per-row switch head),
c = -logit p + log 2 (c_offset), het and affinity log-priors with weight 1.
usage: make_ckpt_A.py STAGE1_CKPT TABLE_JSON OUT [--placement]"""
import json, math, sys
import torch
src, table, dst = sys.argv[1:4]
placement = "--placement" in sys.argv[4:]
ck = torch.load(src, map_location="cpu", weights_only=False)
t = json.load(open(table))
sd = ck["state_dict"]
sd["tern_dist_loglik"] = torch.tensor(t["tern_dist_loglik"], dtype=torch.float32)
sd["tern_loglik"] = torch.tensor(t["tern_loglik"], dtype=torch.float32)
ck["hyper_parameters"]["xo_placement"] = placement
for k, v in (("stay_bonus", 0.0), ("het_w", 1.0), ("aff_w", 1.0), ("c_scale", 1.0), ("c_offset", math.log(2.0)),
             ("xo_scale", 1.0)):
    sd[k] = torch.tensor(v)
ck["optimizer_states"] = []
ck["lr_schedulers"] = []
torch.save(ck, dst)
print("wrote", dst, "placement", placement, {k: float(sd[k]) for k in ("stay_bonus", "het_w", "aff_w", "xo_scale", "c_offset")})
