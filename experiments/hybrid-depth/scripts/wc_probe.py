"""Whole-chromosome vs per-window decode on a few validation rows (diagnostic)."""
import sys, time, json
import hd_lib as H
from wholechrom_v6 import decode_wholechrom
rows = [a.split(":") for a in sys.argv[2:]]          # DS:IND:DEPTH
model = H.load_model()
out = []
for ds, ind, dep in rows:
    row = H.manifest_row("eval", ds, ind, dep); inp = H.load_input("eval", ds, ind, dep)
    t0 = time.time(); lo, hi = H.reference_decode(model, inp); t1 = time.time()
    ref = H.score(inp, lo, hi, row, f"/home/zrm22/.claude/jobs/ec1f8138/tmp/wc/{ds}_{ind}_{dep}_ref")
    lo2, hi2, info = decode_wholechrom(model, inp); t2 = time.time()
    wc = H.score(inp, lo2, hi2, row, f"/home/zrm22/.claude/jobs/ec1f8138/tmp/wc/{ds}_{ind}_{dep}_wc")
    rec = dict(row=f"{ds} {ind} {dep}x", ref_snprc=ref["snprc_error_rate"], wc_snprc=wc["snprc_error_rate"],
               ref_all=ref["error_rate"], wc_all=wc["error_rate"], ref_cov=ref["coverage"], wc_cov=wc["coverage"],
               t_ref=round(t1 - t0), t_wc=round(t2 - t1), hybrid=info["hybrid"])
    print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open(sys.argv[1], "w"), indent=1)
