# Resources the pipeline needs

Every path is read from `scripts/grits.env`, which has no built-in values: the user sets each one, by
exporting the variable or in a copy of `grits.env` pointed at by `GRITS_ENV`. If the lab already has these
resources on a shared server, ask the lab for their locations. On another machine you need all of the
"required" items. The model
is tied to one specific 25-founder panel, so the index, lift and checkpoint must all come from that panel.

## Required

| Variable | What it is | How to get it |
|---|---|---|
| `GRITS_ROPEBWT3` | ropebwt3 with the GRITS `refmap` options | Build [maize-genetics/ropebwt3-phg](https://github.com/maize-genetics/ropebwt3-phg), branch `develop` (PR #10 merged the GRITS options there): `git clone https://github.com/maize-genetics/ropebwt3-phg && cd ropebwt3-phg && git checkout develop && make`. `check_setup.sh` confirms the build has `--err-noclip`. |
| `GRITS_INDEX` | FM-index of the 25 founder assemblies (`.fmd`), with `.fmd.ssa` and `.fmd.len.gz` beside it | Ask the lab for the v2 maize index (about 8 GB). To rebuild, see below. |
| `GRITS_REF_PREFIX` | name of the reference assembly in the index (default `B73`) | For another panel, the assembly you built the lift against. |
| `GRITS_LIFT` | anchor map from each founder to the reference, `-k 50 -s 200` | Ask the lab for its copy, or build one: `ropebwt3 lift --ref-prefix=B73 -k 50 -s 200 -t 16 -o k50_s200.lift INDEX.fmd B73.fa` takes about 4 min and 10 GB RAM. It must be k50/s200: the read-placement fix relies on dense anchors. |
| `GRITS_CKPT` | the promoted model checkpoint (`v6cfullfe-stage2B/last.ckpt`, 23 MB) | Ask the lab. It is trained for exactly this panel. |
| `GRITS_PY`, `GRITS_LD` | Python with torch, pytorch_lightning, numpy, pandas | The repo's default pixi environment works: `pixi install` (its torch build supports CUDA; the `gpu` environment is not needed and currently fails to build `mamba-ssm`), then set `GRITS_PY` to its python and `GRITS_LD` to its `lib/`. |

A GPU with about 30 GB free is strongly recommended.

### The panel

The 25 assemblies, in index order (this order is what the checkpoint expects; it comes from the index):
B73, B97, CML103, CML228, CML247, CML277, CML322, CML333, CML52, CML69, HP301, Il14H, Ki11, Ki3, Ky21,
M162W, M37W, Mo18W, Ms71, NC350, NC358, Oh43, Oh7B, P39, Tzi8.

### Building an index for another species or panel (experimental)

The same recipe as below works for any species: one FASTA per assembly, at most 25 assemblies (the checkpoint's
founder count), reference first, names without `_`. Build the lift with `--ref-prefix=<reference name>` against
the reference's renamed FASTA, and export `GRITS_REF_PREFIX=<reference name>`. `docs/run-sample.md` section 2
has the generic commands. The decode pads a smaller panel with null founders. Accuracy outside maize has not
been validated.

### Rebuilding the maize index

Copying the lab's index is far easier. Only rebuild if that's impossible, and then match the lab's build
exactly; any difference in assemblies, order or naming breaks the model. The lab built it on 2026-08-07
with PHGv2 `rope-bwt-chr-index`, which does exactly these steps:

1. **Rename every contig** (scaffolds, Pt and Mt included) to `<Founder>_<contig>`, e.g. `>chr1` becomes
   `>B73_chr1`. The founder is the text before the first `_`, and the reference is selected by the `B73`
   prefix. The lab's index has 23,860 contigs.
2. **Add the founders one at a time, in the panel order above.** Each step merges one assembly into the
   growing index: about 12-15 min and 44-53 GB RAM per founder, about 6 h in total. A single build over
   all 25 at once needs far more memory.
3. **Convert, then build the suffix array and the name list.** The suffix array step takes about 45 min.

```bash
# 1. rename (assumes headers like ">chr1 ..."); one file per founder
for f in B73 B97 CML103 ... Tzi8; do sed "s/^>\([^ \t]*\).*/>${f}_\1/" /data/nam/$f.fa > ren/${f}_renamed.fa; done
# 2. incremental build, panel order
ropebwt3 build -t30 -bo pan.fmr ren/B73_renamed.fa
for f in B97 CML103 ... Tzi8; do
  ropebwt3 build -t30 -i pan.fmr -bo pan_tmp.fmr ren/${f}_renamed.fa && mv pan_tmp.fmr pan.fmr
done
# 3. static index, suffix array, names and lengths, then the lift
ropebwt3 build -i pan.fmr -do pan.fmd
ropebwt3 ssa -t30 -s8 -o pan.fmd.ssa pan.fmd
for f in B73 B97 CML103 ... Tzi8; do   # contig name and length per line, each file separately
  awk '/^>/{if(n!="")print n"\t"l; n=substr($1,2); l=0; next}{l+=length($0)}END{print n"\t"l}' ren/${f}_renamed.fa
done | gzip > pan.fmd.len.gz
ropebwt3 lift --ref-prefix=B73 -k 50 -s 200 -t16 -o pan.k50_s200.lift pan.fmd ren/B73_renamed.fa
```

Then run `check_setup.sh`, and decode one sample you know (e.g. a NAM founder's reads, which should come
out as that founder almost everywhere) before running your own.

## Optional, for the VCF step only

| Variable | What it is |
|---|---|
| `GRITS_PANEL_VCF` | 25-founder panel VCF (about 14 GB). It has one sample column per founder, with names matching the panel. |
| `GRITS_BED2VCF` | the `sample bed-to-vcf` CLI (Kotlin, in the GRITS repo under the `test_crf_relatedness` Kotlin project; build with `./gradlew installDist`). |
| `GRITS_JAVA_HOME_BIN` | optional: a directory holding a Java 21 `java`, if it is not already on PATH |
| `GRITS_HTSLIB_BIN` | optional: directories holding `bcftools`, `bgzip` and `tabix`, if they are not already on PATH |

## Code version

Get the code with `git clone https://github.com/maize-genetics/grits` and use a branch that includes this
skill.

The scripts call the repo's own model code: `infer_real_founder_pairs` with preset
`placedfix_hetdel_hs_peel` (in `src/python/crf/train_diploid_indel.py`). Run them from a checkout that
contains that preset. If the decode fails with "unknown preset", the checkout is too old.
