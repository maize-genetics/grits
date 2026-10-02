#!/usr/bin/env python
"""
Render heldout5_report_data.json (build_heldout5_report.py's output) into a
self-contained HTML artifact -- the first genuine held-out/generalization
result, shown alongside the in-panel baseline for direct contrast.

Usage:
    python render_heldout5_report.py [--data PATH] [--out PATH]
"""
import argparse
import json
from datetime import date
from pathlib import Path

RESULTS_DIR = Path("/local/workdir/zrm22/HackathonJun2026/grits_workdir/results")
DEFAULT_DATA = RESULTS_DIR / "heldout5_report_data.json"
DEFAULT_OUT = RESULTS_DIR / "heldout5_report.html"


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%" if x is not None else "—"


def err_pct(x, digits=2):
    return f"{x * 100:.{digits}f}%" if x is not None else "—"


def fmt_int(x):
    return f"{x:,}" if x is not None else "—"


def fmt_seconds(x):
    m, s = divmod(int(x), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"


def build_bars(rows):
    ok_rows = [r for r in rows if r["ok"] and r.get("error_rate") is not None]
    max_error = max((r["error_rate"] for r in ok_rows), default=0.0) or 1.0

    out = []
    for r in rows:
        if not r["ok"]:
            out.append(f'''
      <div class="bar-row bar-row--failed">
        <div class="bar-label">{r["assembly"]}</div>
        <div class="bar-track"><div class="bar-failed-note">failed — {r.get("error", "see log")}</div></div>
      </div>''')
            continue
        er = r["error_rate"]
        ceil = r.get("matchable_ceiling_fraction")
        width = (er / max_error) * 100
        out.append(f'''
      <div class="bar-row" tabindex="0">
        <div class="bar-label">{r["assembly"]}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{width:.2f}%"></div>
        </div>
        <div class="bar-value">{err_pct(er)}</div>
        <div class="bar-tooltip">
          <strong>{r["assembly"]}</strong> — {r.get("type", "")}<br>
          error rate: {err_pct(er)}<br>
          concordance: {pct(r["allele_GT_concordance"], 3)}<br>
          matchable ceiling: {pct(ceil) if ceil is not None else "n/a"}<br>
          compared sites: {fmt_int(r["compared_sites"])}<br>
          excluded (no truth info): {fmt_int(r["excluded_no_info"])}<br>
          runtime: {fmt_seconds(r["seconds"])}<br>
          provenance: {r.get("provenance", "")}<br>
          MaizeGDB: {r.get("maizegdb_status", "")}
        </div>
      </div>''')
    return "\n".join(out), max_error


def pp(x, digits=2):
    """Percentage-point formatting for mechanism error-rate shares (already
    a fraction of compared_sites, same denominator as the overall error
    rate, so these are additive -- shown to more precision than a normal
    pct() since individual shares can be small)."""
    return f"{x * 100:.{digits}f}pp" if x is not None else "—"


def build_framing_table_rows(rows, inpanel_corrected):
    """ROUND 3: one row per sample, three units for the SAME underlying
    mismatches -- panel-site (the existing headline), whole-genome-weighted
    (upper bound; assumes the 94.6% of the genome that's never a panel site
    would score correct if it were ever checked), and event-level
    (contiguous mismatch RUNS counted once each, not once per covered base)."""
    out = []

    def one_row(label, panel_site, whole_genome, event, extra_class=""):
        return f'''
        <tr class="{extra_class}">
          <td class="founder-cell">{label}</td>
          <td class="num">{err_pct(panel_site)}</td>
          <td class="num">{err_pct(whole_genome, 3)}</td>
          <td class="num">{err_pct(event, 3)}</td>
        </tr>'''

    for r in rows:
        if not r.get("ok"):
            continue
        out.append(one_row(r["assembly"], r.get("error_rate"),
                            r.get("whole_genome_error_rate"),
                            r.get("events", {}).get("event_error_rate")))
    if inpanel_corrected:
        out.append(one_row(
            "Il14H (in-panel, no-fill-gaps — genuinely independent lineage)",
            inpanel_corrected.get("nofill_error_rate"),
            inpanel_corrected.get("nofill_whole_genome_error_rate"),
            inpanel_corrected.get("events", {}).get("event_error_rate"),
            extra_class="row-inpanel"))
    return "\n".join(out)


def build_noindel_table_rows(rows, inpanel_corrected):
    """ROUND 4: "what do the error rates look like if we ignore indels?" --
    panel-site and event-level, with-indels vs. indels-excluded (HOMREF+SNP
    truth classes only), side by side per sample."""
    out = []

    def one_row(label, panel_all, panel_noindel, event_all, event_noindel, extra_class=""):
        return f'''
        <tr class="{extra_class}">
          <td class="founder-cell">{label}</td>
          <td class="num">{err_pct(panel_all)}</td>
          <td class="num">{err_pct(panel_noindel)}</td>
          <td class="num">{err_pct(event_all, 3)}</td>
          <td class="num">{err_pct(event_noindel, 3)}</td>
        </tr>'''

    for r in rows:
        if not r.get("ok"):
            continue
        out.append(one_row(r["assembly"], r.get("error_rate"), r.get("error_rate_noindel"),
                            r.get("events", {}).get("event_error_rate"),
                            r.get("events", {}).get("event_error_rate_noindel")))
    if inpanel_corrected:
        out.append(one_row(
            "Il14H (in-panel, no-fill-gaps — genuinely independent lineage)",
            inpanel_corrected.get("nofill_error_rate"),
            inpanel_corrected.get("nofill_error_rate_noindel"),
            inpanel_corrected.get("events", {}).get("event_error_rate"),
            inpanel_corrected.get("events", {}).get("event_error_rate_noindel"),
            extra_class="row-inpanel"))
    return "\n".join(out)


STRAT_CLASSES = ["HOMREF", "SNP", "INS", "DEL"]


def build_stratification_table_rows(rows, inpanel_corrected_stratification):
    """One row per sample, one column per truth variant class, showing that
    class's OWN error rate (mismatches / that class's own compared sites) --
    not a share of total error. This is what actually answers whether SNP
    accuracy is meaningfully better than the overall rate once ref-block
    sites (which otherwise dominate and swamp the signal) are structurally
    separated out."""
    out = []

    def one_row(label, strat, extra_class=""):
        if not strat:
            return ""
        by_class = strat["by_class"]
        cells = []
        for cls in STRAT_CLASSES:
            info = by_class.get(cls)
            if not info or "error_rate" not in info:
                cells.append('<td class="num">—</td>')
            else:
                cells.append(f'<td class="num">{err_pct(info["error_rate"])} '
                              f'<span style="opacity:.55">(n={info["total"]:,})</span></td>')
        return f'''
        <tr class="{extra_class}">
          <td class="founder-cell">{label}</td>
          {"".join(cells)}
        </tr>'''

    for r in rows:
        if not r.get("ok") or "stratification" not in r:
            continue
        out.append(one_row(r["assembly"], r["stratification"]))
    out.append(one_row("Il14H (in-panel, no-fill-gaps — genuinely independent lineage)",
                        inpanel_corrected_stratification, extra_class="row-inpanel"))
    return "\n".join(out)


def build_mechanism_table_rows(rows, inpanel_mechanism, inpanel_corrected_mechanism):
    order = ["REFBLOCK_FALSE_POSITIVE", "VARIANT_REF_BIAS", "VARIANT_OTHER_WRONG", "VARIANT_UNMATCHABLE"]
    out = []

    def one_row(label, mech, extra_class=""):
        if not mech:
            return ""
        share = mech["error_rate_share"]
        total = sum(share[k] for k in order)
        cells = "".join(f'<td class="num">{pp(share[k])}</td>' for k in order)
        return f'''
        <tr class="{extra_class}">
          <td class="founder-cell">{label}</td>
          {cells}
          <td class="num"><strong>{pp(total)}</strong></td>
        </tr>'''

    for r in rows:
        if not r.get("ok") or "mechanism" not in r:
            continue
        out.append(one_row(r["assembly"], r["mechanism"]))
    out.append(one_row("Oh43 (in-panel vs. smm477 truth — TAUTOLOGICAL, see note above)",
                        inpanel_mechanism, extra_class="row-inpanel"))
    out.append(one_row("Il14H (in-panel, no-fill-gaps — genuinely independent lineage)",
                        inpanel_corrected_mechanism, extra_class="row-inpanel"))
    return "\n".join(out)


def build_table_rows(rows):
    out = []
    for r in rows:
        if not r["ok"]:
            out.append(f'''
        <tr class="row-failed">
          <td class="founder-cell">{r["assembly"]}</td>
          <td colspan="6" class="error-cell">failed — {r.get("error", "see log")}</td>
        </tr>''')
            continue
        out.append(f'''
        <tr>
          <td class="founder-cell">{r["assembly"]}</td>
          <td>{r.get("type", "")}</td>
          <td class="num">{err_pct(r.get("error_rate"))}</td>
          <td class="num">{pct(r.get("matchable_ceiling_fraction"))}</td>
          <td class="num">{fmt_int(r["compared_sites"])}</td>
          <td class="num">{fmt_seconds(r["seconds"])}</td>
          <td class="maizegdb-cell">{r.get("maizegdb_status", "")}</td>
        </tr>''')
    return "\n".join(out)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Held-out generalization test — 5 genotypes</title>
<style>
:root {{
  --bg: #f7f6f2;
  --surface: #ffffff;
  --surface-2: #eeece4;
  --ink: #1b2430;
  --ink-secondary: #4a5568;
  --ink-muted: #7a8494;
  --border: #ddd8cb;
  --accent: #a3400e;
  --accent-ink: #ffffff;
  --accent-soft: #f4ded2;
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
    --accent: #e8875e;
    --accent-ink: #1c1006;
    --accent-soft: #3a271c;
    --good: #6fbf8b;
    --bad: #e18066;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14181f; --surface: #1c222c; --surface-2: #242b37; --ink: #e8eaee;
  --ink-secondary: #aab2c0; --ink-muted: #7a8494; --border: #313a48;
  --accent: #e8875e; --accent-ink: #1c1006; --accent-soft: #3a271c;
  --good: #6fbf8b; --bad: #e18066;
}}
:root[data-theme="light"] {{
  --bg: #f7f6f2; --surface: #ffffff; --surface-2: #eeece4; --ink: #1b2430;
  --ink-secondary: #4a5568; --ink-muted: #7a8494; --border: #ddd8cb;
  --accent: #a3400e; --accent-ink: #ffffff; --accent-soft: #f4ded2;
  --good: #2f7d4f; --bad: #b3452f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--font-body); line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1000px; margin: 0 auto; padding: 56px 24px 96px; }}
.eyebrow {{
  font-family: var(--font-mono); font-size: 12px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 10px;
}}
h1 {{
  font-family: var(--font-display); font-weight: 500; font-size: 34px;
  line-height: 1.15; margin: 0 0 14px; text-wrap: balance; color: var(--ink);
}}
.lede {{
  font-size: 16px; color: var(--ink-secondary); max-width: 68ch; margin: 0 0 8px;
}}
.framing {{
  margin: 20px 0 40px; padding: 16px 20px; background: var(--surface-2);
  border-left: 3px solid var(--accent); border-radius: 4px;
  font-size: 14.5px; color: var(--ink-secondary); max-width: 70ch;
}}
.framing strong {{ color: var(--ink); }}
.stat-row {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--border); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; margin-bottom: 32px;
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
.compare-row {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
  background: var(--border); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden; margin-bottom: 48px;
}}
.compare-tile {{ background: var(--surface); padding: 22px 24px; }}
.compare-tile.is-heldout {{ background: var(--accent-soft); }}
.compare-label {{
  font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--ink-muted); margin: 0 0 8px;
}}
.compare-value {{
  font-family: var(--font-display); font-size: 40px; color: var(--ink);
  font-variant-numeric: tabular-nums; margin: 0 0 4px;
}}
.compare-note {{ font-size: 13px; color: var(--ink-secondary); }}
section {{ margin-bottom: 52px; }}
h2 {{
  font-family: var(--font-display); font-size: 20px; font-weight: 500;
  margin: 0 0 4px; color: var(--ink);
}}
.section-note {{ font-size: 13.5px; color: var(--ink-muted); margin: 0 0 20px; max-width: 70ch; }}

.chart {{ display: flex; flex-direction: column; gap: 6px; }}
.bar-row {{
  position: relative; display: grid;
  grid-template-columns: 90px 1fr 64px; align-items: center; gap: 12px;
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
  display: none; position: absolute; left: 102px; top: -6px; transform: translateY(-100%);
  background: var(--ink); color: var(--bg); padding: 10px 12px; border-radius: 6px;
  font-family: var(--font-mono); font-size: 11.5px; line-height: 1.6; white-space: nowrap;
  z-index: 10; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}}
.bar-row:hover .bar-tooltip, .bar-row:focus .bar-tooltip {{ display: block; }}

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
td.maizegdb-cell {{ font-family: var(--font-mono); font-size: 12px; }}
tr:last-child td {{ border-bottom: none; }}
.row-failed td {{ color: var(--bad); }}
.row-inpanel {{ background: var(--surface-2); }}
.row-inpanel td.founder-cell {{ color: var(--good); }}
.error-cell {{ font-family: var(--font-mono); font-size: 12px; }}

.roadmap {{
  padding: 18px 20px; background: var(--surface); border: 1px dashed var(--border);
  border-radius: 8px; font-size: 14px; color: var(--ink-secondary); max-width: 70ch;
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
  <h1>Held-out generalization test — 5 genotypes</h1>
  <p class="lede">Five maize assemblies confirmed <strong>excluded</strong> from our 25-founder
    ropebwt3 index, run through the identical simulate&rarr;refmap&rarr;CRF+affinity&rarr;compare
    pipeline as the in-panel baseline. Picked for germplasm-type diversity (NAM/tropical, Corn Belt
    dent, European Flint, CIMMYT highland tropical, sweet corn) and quality-checked against
    MaizeGDB/the primary literature before running.</p>

  <div class="framing">
    <strong>This is the real number.</strong> The in-panel smoke test (25 NAM founders, all already
    in our index) was a best-case ceiling. These 5 are genuinely novel genotypes with no representation
    in the training panel &mdash; this is what the pipeline actually does on new, untyped samples.
  </div>

  <section>
    <h2>Held-out vs. in-panel</h2>
    <p class="section-note">Same pipeline, same read depth (250,000), same panel. The only
      difference is whether the sample was ever in the index.</p>
    <div class="compare-row">
      <div class="compare-tile">
        <p class="compare-label">In-panel mean error rate (as originally reported)</p>
        <p class="compare-value">{inpanel_error}</p>
        <p class="compare-note">25 NAM founders, all already indexed &mdash; best-case ceiling</p>
      </div>
      <div class="compare-tile is-heldout">
        <p class="compare-label">Held-out mean error rate</p>
        <p class="compare-value">{heldout_error}</p>
        <p class="compare-note">{fold_change} the naive in-panel rate &mdash; see correction below</p>
      </div>
    </div>
    <div class="framing">
      <strong>Correction, retracting an earlier retraction.</strong> {denominator_note}
    </div>
    <div class="compare-row">
      <div class="compare-tile">
        <p class="compare-label">Il14H, no-fill-gaps (in-panel, genuinely independent alignment lineage)</p>
        <p class="compare-value">{corrected_inpanel_error}</p>
        <p class="compare-note">the only in-panel founder scored against a truth gVCF NOT used to
          build the panel itself &mdash; every other founder's comparison is tautological (see note above)</p>
      </div>
      <div class="compare-tile is-heldout">
        <p class="compare-label">True fold-change</p>
        <p class="compare-value">{corrected_fold_change}</p>
        <p class="compare-note">held-out mean vs. this like-for-like, non-tautological baseline</p>
      </div>
    </div>
  </section>

  <section>
    <h2>Three units, same mismatches: is 18.6% really "too high"?</h2>
    <p class="section-note">{round3_note}</p>
    <div class="framing">
      <strong>First, the direct answer to "partial or whole-block?"</strong> A real 246bp truth
      ref-block on Tx303's chr1 (<code>chr1:1398691-1398936</code>) contains 12 panel sites and
      exactly 1 real disagreement. Running the actual comparator on it:
      <code>compared_sites=12, gt_allele_mismatches=1</code> &mdash; not 246, not 12. Every
      disagreement is scored at the single position it occurs, independently. No whole-block bug
      exists. What DOES inflate the headline number is explained below.</p>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Sample</th>
            <th class="num">Panel-site error<br><span style="font-weight:400;opacity:.7">(the headline number)</span></th>
            <th class="num">Whole-genome-weighted<br><span style="font-weight:400;opacity:.7">(upper bound)</span></th>
            <th class="num">Event-level<br><span style="font-weight:400;opacity:.7">(1 vote per wrong stretch)</span></th>
          </tr>
        </thead>
        <tbody>
{framing_rows}
        </tbody>
      </table>
    </div>
    <p class="section-note" style="margin-top:16px;">
      <strong>Whole-genome-weighted</strong> re-weights the SAME mismatch count over the true
      genome size (2,131,846,805bp, chr1&ndash;10) instead of the 115,354,779 panel sites (5.41% of
      the genome) &mdash; i.e. it assumes the 94.6% of the genome that's never a panel site (because
      no founder differs from B73 there) would score correct if it were ever checked. This is an
      <strong>upper bound on correctness, not a direct measurement</strong>: held-out samples can
      carry private variants at non-panel positions that no founder path can express, and those
      would be silent, unmeasured errors here.
      <strong>Event-level</strong> collapses contiguous runs of mismatching sites into one "event"
      each (matches stay counted per-site) &mdash; because ~35% of all panel sites are synthetic
      per-base positions inside a truth deletion's span, one missed large structural variant (e.g.
      a 21kb deletion) gets counted as a mismatch at every interior site it covers, which can turn
      one wrong biological call into 1,000+ counted "errors." Neither number replaces the
      panel-site rate above &mdash; both are additional, differently-scoped views of the exact same
      underlying predictions.</p>
  </section>

  <section>
    <h2>What if we ignore indels entirely?</h2>
    <p class="section-note">Restricting to HOMREF + SNP truth sites only (dropping INS/DEL/HET_MIXED
      from both numerator and denominator), site-count-weighted for the panel-site column. This
      isolates the model's founder-path recovery from the indel-representation effects discussed
      above &mdash; and from Il14H's own perspective, removing DEL actually RAISES its error rate,
      since DEL was its easiest class (partly a representation-coarseness artifact, see the
      stratification table below).</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Sample</th>
            <th class="num">Panel-site<br><span style="font-weight:400;opacity:.7">(with indels)</span></th>
            <th class="num">Panel-site<br><span style="font-weight:400;opacity:.7">(no indels)</span></th>
            <th class="num">Event-level<br><span style="font-weight:400;opacity:.7">(with indels)</span></th>
            <th class="num">Event-level<br><span style="font-weight:400;opacity:.7">(no indels)</span></th>
          </tr>
        </thead>
        <tbody>
{noindel_rows}
        </tbody>
      </table>
    </div>
  </section>

  <div class="stat-row">
    <div class="stat-tile">
      <p class="stat-label">Assemblies tested</p>
      <p class="stat-value">{n_ok} / {n_assemblies}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Best (lowest error)</p>
      <p class="stat-value">{best_founder}</p>
    </div>
    <div class="stat-tile">
      <p class="stat-label">Worst (highest error)</p>
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
    <h2>Error rate by assembly</h2>
    <p class="section-note">Genotype error rate (1 &minus; concordance), best (lowest error) first.
      Bars scaled to the highest error rate observed here ({max_error_scale}). Hover a row for
      provenance and exact figures.</p>
    <div class="chart">
{bars}
    </div>
  </section>

  <section>
    <h2>Why: error mechanism breakdown</h2>
    <p class="section-note">Each cell is that mechanism's own share of the OVERALL error rate
      (percentage points of <code>compared_sites</code>, so a row sums to its total error rate).
      <strong>REFBLOCK_FALSE_POSITIVE</strong> &mdash; predicting a non-reference founder where the
      genotype is actually reference-identical &mdash; is consistently the largest single mechanism
      for the held-out samples. Root cause (confirmed by reading the CRF decode code): the model's
      entire state space is pairs of the 25 known founders, with no separate "reference / none of
      the above" state &mdash; a genuinely novel genotype forces it to always commit to some known
      founder, which disagrees wherever that founder's own variants don't match. Architectural, not
      a bug.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Assembly</th>
            <th class="num">Ref-block false positive</th>
            <th class="num">Variant: ref-bias</th>
            <th class="num">Variant: other wrong</th>
            <th class="num">Variant: unmatchable</th>
            <th class="num">Total error rate</th>
          </tr>
        </thead>
        <tbody>
{mechanism_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Stratified by variant class: is SNP accuracy actually better?</h2>
    <p class="section-note">Each cell is that TRUTH class's OWN error rate (mismatches / that
      class's own compared sites), not a share of anything &mdash; keying on the truth side keeps
      HOMREF (the majority of any genome) from swamping the SNP/indel signal. <code>n=</code> is
      that class's own site count. INS/DEL accuracy here means "did the model correctly identify
      an insertion/deletion is present," not "did it reconstruct the exact sequence" &mdash; the
      panel only ever encodes indels as the symbolic <code>&lt;INS&gt;</code>/<code>&lt;DEL&gt;</code>
      alleles.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Assembly</th>
            <th class="num">HOMREF</th>
            <th class="num">SNP</th>
            <th class="num">INS</th>
            <th class="num">DEL</th>
          </tr>
        </thead>
        <tbody>
{stratification_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Full results</h2>
    <p class="section-note">Matchable ceiling (panel-representability floor) is markedly lower here
      than in-panel &mdash; expected, since a genuinely new genotype carries alleles no founder in
      our 25-founder panel has.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Assembly</th>
            <th>Germplasm type</th>
            <th class="num">Error rate</th>
            <th class="num">Matchable ceiling</th>
            <th class="num">Compared sites</th>
            <th class="num">Runtime</th>
            <th>MaizeGDB</th>
          </tr>
        </thead>
        <tbody>
{table_rows}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <div class="roadmap">
      <strong>Next.</strong> The dominant remaining mechanism (REFBLOCK_FALSE_POSITIVE) traces to
      the CRF decoder having no null/reference state in its decode space &mdash; a real architectural
      property, not a quick patch, and out of scope for a fix this round. The frequency-binned
      accuracy/R2 diagnostics remain silently dead in every report this pipeline has produced so far
      &mdash; the fix (computing AC/AN in <code>BedToVcf.kt</code>) was written but never applied to
      any of the imputed VCFs this report is built from (that would need a bed-to-vcf re-run, part of
      the panel rebuild that was evaluated and skipped this round &mdash; see the correction note
      above). Available for a future pass. This is also the second real run (after the in-panel baseline)
      that this report's own earlier draft said would seed a proper cross-run comparison view &mdash;
      pipeline versions, checkpoints, panel compositions, read depths, all against each other over
      time, not just this one static pair.
    </div>
  </section>

  <footer>generated {generated_date} &middot; scripts/heldout_batch.py + build_heldout5_report.py &middot; grits_workdir</footer>

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
    rows = data["assemblies"]
    inpanel = data.get("inpanel_comparison")
    inpanel_mechanism = data.get("inpanel_mechanism")
    denominator_note = data.get("denominator_note", "")
    round3_note = data.get("round3_note", "")
    inpanel_corrected = data.get("inpanel_corrected")
    inpanel_corrected_mechanism = data.get("inpanel_corrected_mechanism")
    inpanel_corrected_stratification = data.get("inpanel_corrected_stratification")

    ok_rows = [r for r in rows if r["ok"] and r.get("error_rate") is not None]
    best = min(ok_rows, key=lambda r: r["error_rate"]) if ok_rows else None
    worst = max(ok_rows, key=lambda r: r["error_rate"]) if ok_rows else None

    bars_html, max_error = build_bars(rows)

    inpanel_error = inpanel["mean_error_rate"] if inpanel else None
    heldout_error = agg.get("mean_error_rate")
    fold_change = (f"{heldout_error / inpanel_error:.1f}×"
                    if inpanel_error else "—")

    if inpanel_corrected:
        corrected_inpanel_error = err_pct(inpanel_corrected.get("nofill_error_rate"), 2)
        corrected_fold_change = (f"{inpanel_corrected['corrected_fold_change']:.1f}×"
                                  if inpanel_corrected.get("corrected_fold_change") else "—")
    else:
        corrected_inpanel_error = corrected_fold_change = "—"

    html = TEMPLATE.format(
        n_ok=agg["n_ok"],
        n_assemblies=agg["n_assemblies"],
        best_founder=f'{best["assembly"]} ({err_pct(best["error_rate"])})' if best else "—",
        worst_founder=f'{worst["assembly"]} ({err_pct(worst["error_rate"])})' if worst else "—",
        mean_ceiling=pct(agg.get("mean_ceiling")),
        depth=f'{agg["depth"]:,}',
        total_runtime=fmt_seconds(agg["total_seconds"]),
        max_error_scale=err_pct(max_error),
        bars=bars_html,
        table_rows=build_table_rows(rows),
        mechanism_rows=build_mechanism_table_rows(rows, inpanel_mechanism, inpanel_corrected_mechanism),
        stratification_rows=build_stratification_table_rows(rows, inpanel_corrected_stratification),
        framing_rows=build_framing_table_rows(rows, inpanel_corrected),
        noindel_rows=build_noindel_table_rows(rows, inpanel_corrected),
        inpanel_error=err_pct(inpanel_error, 3) if inpanel_error is not None else "—",
        heldout_error=err_pct(heldout_error),
        fold_change=fold_change,
        denominator_note=denominator_note,
        round3_note=round3_note,
        corrected_inpanel_error=corrected_inpanel_error,
        corrected_fold_change=corrected_fold_change,
        generated_date=date.today().isoformat(),
    )
    Path(args.out).write_text(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
