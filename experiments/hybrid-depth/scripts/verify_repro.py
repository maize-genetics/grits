"""decode_custom with every knob off == infer_real_founder_pairs; score == recorded JSON."""
import json, sys, time
import numpy as np
import hd_lib as L
ds, ind, dep, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
m = L.load_model()
inp = L.load_input("eval", ds, ind, dep)
t = time.time(); lo0, hi0 = L.reference_decode(m, inp); t1 = time.time() - t
t = time.time(); lo1, hi1, info = L.decode_custom(m, inp); t2 = time.time() - t
print("ref vs custom: rows differ", int(((lo0 != lo1) | (hi0 != hi1)).sum()), "of", lo0.size, f"({t1:.0f}s/{t2:.0f}s)", info["route"])
row = L.manifest_row("eval", ds, ind, dep)
res = L.score(inp, lo1, hi1, row, f"/tmp/hd_verify_bed_{ind}_{dep}")
rec = json.load(open(f"{L.WORK}/results/{tag}/{json.load(open(f'{L.WORK}/results/{tag}/'+ [f for f in __import__('os').listdir(f'{L.WORK}/results/{tag}') if ind in f and ds in f][0]))['row_key']}.json"))
print("custom err", res["error_rate"], "snprc", res["snprc_error_rate"], "cov", res["coverage"])
print("recorded err", rec["error_rate"], "snprc", rec["snprc_error_rate"])
