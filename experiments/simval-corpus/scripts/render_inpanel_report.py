#!/usr/bin/env python
"""
Render nam_inpanel_report_data.json (build_inpanel_report.py's output) into
the self-contained HTML artifact. Kept separate from data-building so the
report's design can be iterated on without re-parsing/re-running anything.

Leads with ERROR RATE (1 - concordance), not concordance: concordance
clusters so tightly near 100% (99.2-100%) that any reasonable display
precision hides real per-founder differences; the same signal sits in the
LEADING digits of the error rate instead of buried after "99.8...".

Usage:
    python render_inpanel_report.py [--data PATH] [--out PATH]
"""
import argparse
import json
from datetime import date
from pathlib import Path

RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
DEFAULT_DATA = RESULTS_DIR / "nam_inpanel_report_data.json"
DEFAULT_OUT = RESULTS_DIR / "nam_inpanel_report.html"


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%" if x is not None else "—"


def err_pct(x, digits=3):
    """Error rate display: more decimal places than a normal pct(), since
    values cluster in a narrow 0-1% band where the interesting digits are
    the ones a 1-decimal pct() would round away."""
    return f"{x * 100:.{digits}f}%" if x is not None else "—"


def fmt_int(x):
    return f"{x:,}" if x is not None else "—"


def fmt_seconds(x):
    m, s = divmod(int(x), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"


def build_bars(rows):
    """One HTML row per founder: label, error-rate bar. Bars are scaled to
    the MAX observed error rate (not a 0-100% axis) -- error rates cluster
    in a ~0-1% band, so a raw 0-100%-scaled bar would render every founder
    as an invisible sliver. The scale is stated explicitly so this reads as
    relative ranking, not an absolute percentage."""
    ok_rows = [r for r in rows if r["ok"] and r.get("error_rate") is not None]
    max_error = max((r["error_rate"] for r in ok_rows), default=0.0) or 1.0

    out = []
    for r in rows:
        if not r["ok"]:
            out.append(f'''
      <div class="bar-row bar-row--failed">
        <div class="bar-label">{r["founder"]}</div>
        <div class="bar-track"><div class="bar-failed-note">failed — {r.get("error", "see log")}</div></div>
      </div>''')
            continue
        er = r["error_rate"]
        ceil = r.get("matchable_ceiling_fraction")
        width = (er / max_error) * 100
        out.append(f'''
      <div class="bar-row" tabindex="0">
        <div class="bar-label">{r["founder"]}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{width:.2f}%"></div>
        </div>
        <div class="bar-value">{err_pct(er)}</div>
        <div class="bar-tooltip">
          <strong>{r["founder"]}</strong><br>
          error rate: {err_pct(er)}<br>
          partial error rate: {err_pct(r["partial_error_rate"])}<br>
          concordance: {pct(r["allele_GT_concordance"], 3)}<br>
          matchable ceiling: {pct(ceil) if ceil is not None else "n/a (no truth variants)"}<br>
          compared sites: {fmt_int(r["compared_sites"])}<br>
          excluded (no truth info): {fmt_int(r["excluded_no_info"])}<br>
          runtime: {fmt_seconds(r["seconds"])}
        </div>
      </div>''')
    return "\n".join(out), max_error


def build_table_rows(rows):
    out = []
    for r in rows:
        if not r["ok"]:
            out.append(f'''
        <tr class="row-failed">
          <td class="founder-cell">{r["founder"]}</td>
          <td colspan="6" class="error-cell">failed — {r.get("error", "see log")}</td>
        </tr>''')
            continue
        out.append(f'''
        <tr>
          <td class="founder-cell">{r["founder"]}</td>
          <td class="num">{err_pct(r.get("error_rate"))}</td>
          <td class="num">{err_pct(r.get("partial_error_rate"))}</td>
          <td class="num">{pct(r.get("matchable_ceiling_fraction"))}</td>
          <td class="num">{fmt_int(r["compared_sites"])}</td>
          <td class="num">{fmt_int(r["excluded_no_info"])}</td>
          <td class="num">{fmt_seconds(r["seconds"])}</td>
        </tr>''')
    return "\n".join(out)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NAM in-panel smoke test</title>
<style>
:root {{
  --bg: #f7f6f2;
  --surface: #ffffff;
  --surface-2: #eeece4;
  --ink: #1b2430;
  --ink-secondary: #4a5568;
  --ink-muted: #7a8494;
  --border: #ddd8cb;
  --accent: #0e6e73;
  --accent-ink: #ffffff;
  --accent-soft: #d8ecec;
  --good: #2f7d4f;
  --bad: #b3452f;
  --font-display: Palatino, "Palatino Linotype", "Iowan Old Style", Georgia, "Times New Roman", serif;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SF Mono", "Cascadia Code", "Consolas", "Liberation Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14181f;
    --surface: #1c222c;
    --surface-2: #242b37;
    --ink: #e8eaee;
    --ink-secondary: #aab2c0;
    --ink-muted: #7a8494;
    --border: #313a48;
    --accent: #4fb8bd;
    --accent-ink: #0b1416;
    --accent-soft: #1e3436;
    --good: #6fbf8b;
    --bad: #e18066;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14181f; --surface: #1c222c; --surface-2: #242b37; --ink: #e8eaee;
  --ink-secondary: #aab2c0; --ink-muted: #7a8494; --border: #313a48;
  --accent: #4fb8bd; --accent-ink: #0b1416; --accent-soft: #1e3436;
  --good: #6fbf8b; --bad: #e18066;
}}
:root[data-theme="light"] {{
  --bg: #f7f6f2; --surface: #ffffff; --surface-2: #eeece4; --ink: #1b2430;
  --ink-secondary: #4a5568; --ink-muted: #7a8494; --border: #ddd8cb;
  --accent: #0e6e73; --accent-ink: #ffffff; --accent-soft: #d8ecec;
  --good: #2f7d4f; --bad: #b3452f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--font-body); line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 56px 24px 96px; }}
.eyebrow {{
  font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 10px;
}}
h1 {{
  font-family: var(--font-display); font-weight: 500; font-size: 34px;
  line-height: 1.15; margin: 0 0 14px; text-wrap: balance; color: var(--ink);
}}
.lede {{
  font-size: 16px; color: var(--ink-secondary); max-width: 66ch; margin: 0 0 8px;
}}
.framing {{
  margin: 20px 0 40px; padding: 16px 20px; background: var(--surface-2);
  border-left: 3px solid var(--accent); border-radius: 4px;
  font-size: 14.5px; color: var(--ink-secondary); max-width: 68ch;
}}
.framing strong {{ color: var(--ink); }}
.framing + .framing {{ margin-top: -24px; border-left-color: var(--ink-muted); }}
.stat-row {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--border); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; margin-bottom: 48px;
}}
.stat-tile {{ background: var(--surface); padding: 18px 20px; }}
.stat-label {{
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--ink-muted); margin: 0 0 6px;
}}
.stat-value {{
  font-family: var(--font-display); font-size: 26px; color: var(--ink);
  font-variant-numeric: tabular-nums;
}}
section {{ margin-bottom: 52px; }}
h2 {{
  font-family: var(--font-display); font-size: 20px; font-weight: 500;
  margin: 0 0 4px; color: var(--ink);
}}
.section-note {{ font-size: 13.5px; color: var(--ink-muted); margin: 0 0 20px; max-width: 68ch; }}

.chart {{ display: flex; flex-direction: column; gap: 6px; }}
.bar-row {{
  position: relative; display: grid;
  grid-template-columns: 76px 1fr 64px; align-items: center; gap: 12px;
  padding: 5px 0;
}}
.bar-row--failed {{ opacity: 0.6; }}
.bar-label {{
  font-family: var(--font-mono); font-size: 12.5px; color: var(--ink-secondary);
  text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.bar-track {{
  position: relative; height: 16px; background: var(--surface-2);
  border-radius: 3px; overflow: visible;
}}
.bar-fill {{
  position: absolute; inset: 0 auto 0 0; height: 100%;
  background: var(--accent); border-radius: 3px 4px 4px 3px;
}}
.bar-value {{
  font-family: var(--font-mono); font-size: 12.5px; color: var(--ink);
  font-variant-numeric: tabular-nums;
}}
.bar-failed-note {{
  font-family: var(--font-mono); font-size: 12px; color: var(--bad); padding: 2px 0;
}}
.bar-tooltip {{
  display: none; position: absolute; left: 88px; top: -6px; transform: translateY(-100%);
  background: var(--ink); color: var(--bg); padding: 10px 12px; border-radius: 6px;
  font-family: var(--font-mono); font-size: 11.5px; line-height: 1.6; white-space: nowrap;
  z-index: 10; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}}
.bar-row:hover .bar-tooltip, .bar-row:focus .bar-tooltip {{ display: block; }}
.legend {{
  display: flex; gap: 20px; font-size: 12px; color: var(--ink-muted);
  margin-top: 14px; font-family: var(--font-mono);
}}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 2px; background: var(--accent); }}

table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
.table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--ink-muted); font-weight: 500;
  background: var(--surface-2);
}}
td {{ color: var(--ink-secondary); }}
td.founder-cell {{ font-family: var(--font-mono); color: var(--ink); font-weight: 600; }}
td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
tr:last-child td {{ border-bottom: none; }}
.row-failed td {{ color: var(--bad); }}
.error-cell {{ font-family: var(--font-mono); font-size: 12px; }}

.roadmap {{
  padding: 18px 20px; background: var(--surface); border: 1px dashed var(--border);
  border-radius: 8px; font-size: 14px; color: var(--ink-secondary); max-width: 68ch;
}}
.roadmap strong {{ color: var(--ink); }}
footer {{
  margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--border);
  font-family: var(--font-mono); font-size: 11.5px; color: var(--ink-muted);
}}
</style>
</head>
<body>
<div class="wrap">

  <p class="eyebrow">GRITS &middot; pipeline evaluation</p>
  <h1>NAM in-panel smoke test</h1>
  <p class="lede">Every NAM founder run through the full held-out-assembly evaluation pipeline
    &mdash; wgsim &rarr; ropebwt3 refmap &rarr; CRF+affinity inference &rarr; imputed VCF &rarr; compared
    against each founder&rsquo;s own truth gVCF.</p>

  <div class="framing">
    <strong>This is an in-panel baseline, not a generalization test.</strong> Every founder scored
    here is already in our ropebwt3 index (25 founders including B73), so the model can in principle
    recognize its own sequence. Low error here is an expected floor &mdash; it validates the
    pipeline wiring end-to-end at full genome scale and gives a best-case reference point. The real
    test is what happens on assemblies genuinely excluded from the index, which is tracked separately.
  </div>
  <div class="framing">
    <strong>Two comparator bugs fixed since the first pass of this report.</strong> (1) A contig-ordering
    bug silently excluded 100% of chr10 from every comparison (string vs. numeric sort mismatch). (2) Large
    structural deletions (common between NAM founders and B73, up to tens of kb) were collapsing every
    pangenome SNP site inside their span into "no info," excluding ~35% of the genome from scoring; these
    now resolve against the panel's own symbolic &lt;DEL&gt; allele. Both were verified against real data
    (including a B73-vs-itself negative control) before this run. Reported error rates are higher than
    earlier drafts of this report as a direct, expected result &mdash; more of the genome, including the
    harder structurally-divergent regions, is now honestly scored instead of silently excluded.
  </div>

  <div class="stat-row">
    <div class="stat-tile">
      <p class="stat-label">Founders tested</p>
      <p class="stat-value">{n_ok} / {n_founders}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Mean error rate</p>
      <p class="stat-value">{mean_error_rate}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Best founder</p>
      <p class="stat-value">{best_founder}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Worst founder</p>
      <p class="stat-value">{worst_founder}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Mean matchable ceiling</p>
      <p class="stat-value">{mean_ceiling}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Read depth</p>
      <p class="stat-value">{depth}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Total runtime</p>
      <p class="stat-value">{total_runtime}</p>
    </div>
  </div>

  <section>
    <h2>Error rate by founder</h2>
    <p class="section-note">Genotype error rate (1 &minus; concordance) between the imputed path and
      each founder&rsquo;s own truth gVCF, best (lowest error) first. Bars are scaled to the highest
      error rate observed among these 25 founders ({max_error_scale}), not to 0&ndash;100%
      &mdash; error rates cluster in a narrow band, so this reads as relative ranking, not an
      absolute share. Hover a row for exact figures.</p>
    <div class="chart">
{bars}
    </div>
  </section>

  <section>
    <h2>Full results</h2>
    <p class="section-note">Excluded-no-info sites (no truth-gVCF coverage at that position at all) are
      dropped from the denominator, not scored as reference &mdash; see compare_gvcf_truth.py.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Founder</th>
            <th class="num">Error rate</th>
            <th class="num">Partial error rate&nbsp;<span title="{partial_metric_note}">&dagger;</span></th>
            <th class="num">Matchable ceiling</th>
            <th class="num">Compared sites</th>
            <th class="num">Excluded (no info)</th>
            <th class="num">Runtime</th>
          </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
    <p class="section-note">&dagger; {partial_metric_note}</p>
  </section>

  <section>
    <div class="roadmap">
      <strong>Next.</strong> These 25 rows are the in-panel floor. The next run adds genuinely
      out-of-index maize assemblies (excluded from the ropebwt3 index, currently being sourced) through
      this exact same pipeline &mdash; the number that actually matters for real, untyped samples. As
      more pipeline variants get evaluated this way (different checkpoints, panel compositions, read
      depths), this report is the seed for a proper comparison view across runs.
    </div>
  </section>

  <footer>generated {generated_date} &middot; scripts/nam_inpanel_smoketest.py + rerun_comparisons_fixed.py &middot; grits_workdir</footer>

</div>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text())
    agg = data["aggregate"]
    rows = data["founders"]

    ok_rows = [r for r in rows if r["ok"] and r.get("error_rate") is not None]
    best = min(ok_rows, key=lambda r: r["error_rate"]) if ok_rows else None
    worst = max(ok_rows, key=lambda r: r["error_rate"]) if ok_rows else None

    bars_html, max_error = build_bars(rows)

    html = TEMPLATE.format(
        n_ok=agg["n_ok"],
        n_founders=agg["n_founders"],
        mean_error_rate=err_pct(agg.get("mean_error_rate")),
        best_founder=f'{best["founder"]} ({err_pct(best["error_rate"])})' if best else "—",
        worst_founder=f'{worst["founder"]} ({err_pct(worst["error_rate"])})' if worst else "—",
        mean_ceiling=pct(agg["mean_ceiling"]),
        depth=f'{agg["depth"]:,}',
        total_runtime=fmt_seconds(agg["total_seconds"]),
        max_error_scale=err_pct(max_error),
        bars=bars_html,
        table_rows=build_table_rows(rows),
        partial_metric_note=agg.get(
            "partial_metric_note",
            "Partial error rate note unavailable (regenerate report_data.json with the "
            "current build_inpanel_report.py to populate it).",
        ),
        generated_date=date.today().isoformat(),
    )
    Path(args.out).write_text(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
