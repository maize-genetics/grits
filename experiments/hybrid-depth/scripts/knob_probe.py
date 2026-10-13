"""Mechanism probe (validation rows, diagnostic only -- not a tuned setting)."""
import sys, json
import hd_lib as L
m = L.load_model()
ds, ind = sys.argv[1], sys.argv[2]
for dep in [float(x) for x in sys.argv[3].split(",")]:
    inp = L.load_input("eval", ds, ind, dep); row = L.manifest_row("eval", ds, ind, dep)
    for name, kw in [("base", {}), ("aff x16", dict(aff_mult=16.0)),
                     ("emis max", dict(emis="max")), ("emis max + aff x4", dict(emis="max", aff_mult=4.0))]:
        lo, hi, info = L.decode_custom(m, inp, **kw)
        r = L.score(inp, lo, hi, row, f"/tmp/hd_knob_{ind}_{dep}")
        homo = float((lo == hi).mean())
        print(f"{ds} {ind} {dep}x {name:<18} snprc {100*r['snprc_error_rate']:.2f}  all {100*r['error_rate']:.2f}  cov {r['coverage']:.4f}  homo_rows {homo:.3f}", flush=True)
