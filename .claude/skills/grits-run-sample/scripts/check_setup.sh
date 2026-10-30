#!/bin/bash
# check_setup.sh -- verify every resource in grits.env exists and works, before spending an hour on a run.
#   check_setup.sh [--vcf]      (--vcf also checks the optional BED->VCF resources)
# Prints one line per check; exits 1 if anything required is missing.
HERE=$(cd "$(dirname "$0")" && pwd)
source "${GRITS_ENV:-$HERE/grits.env}"
bad=0
ok()   { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; bad=1; }
need() { [ -e "$2" ] && ok "$1: $2" || fail "$1 missing: $2"; }

echo "Required:"
[ -x "$GRITS_ROPEBWT3" ] && ok "ropebwt3: $GRITS_ROPEBWT3" || fail "ropebwt3 not executable: $GRITS_ROPEBWT3"
if [ -x "$GRITS_ROPEBWT3" ]; then
  "$GRITS_ROPEBWT3" refmap 2>&1 | grep -q -- '--err-noclip' && ok "ropebwt3 has the promoted refmap options" \
    || fail "ropebwt3 is too old (no --err-noclip): build maize-genetics/ropebwt3-phg branch grits-pipeline"
fi
need "index" "$GRITS_INDEX"
need "index .ssa" "$GRITS_INDEX.ssa"
need "index .len.gz" "$GRITS_INDEX.len.gz"
need "lift (k50 s200)" "$GRITS_LIFT"
need "checkpoint" "$GRITS_CKPT"
if [ -e "$GRITS_INDEX.len.gz" ]; then
  n=$(zcat "$GRITS_INDEX.len.gz" | cut -f1 | sed 's/_.*//' | sort -u | wc -l)
  [ "$n" = 25 ] && ok "index holds 25 assemblies" || fail "index holds $n assemblies; the model needs exactly the 25-founder panel"
fi
if [ -x "$GRITS_PY" ]; then
  LD_LIBRARY_PATH=$GRITS_LD:${LD_LIBRARY_PATH:-} "$GRITS_PY" -c \
    "import torch, pytorch_lightning, numpy, pandas; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu-only')" \
    > /tmp/grits_py_check.$$ 2>&1 && ok "python: $(cat /tmp/grits_py_check.$$)" || fail "python env lacks torch/pytorch_lightning/pandas: $(tail -1 /tmp/grits_py_check.$$)"
  rm -f /tmp/grits_py_check.$$
else
  fail "python not found: $GRITS_PY"
fi
if command -v nvidia-smi >/dev/null; then
  ok "GPUs (free MiB): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
else
  echo "  note  no nvidia-smi: decode will run on CPU (slow for a whole genome)"
fi

if [ "${1:-}" = --vcf ]; then
  echo "Optional (VCF output):"
  need "panel VCF" "$GRITS_PANEL_VCF"
  [ -x "$GRITS_BED2VCF" ] && ok "bed-to-vcf CLI: $GRITS_BED2VCF" || fail "bed-to-vcf CLI not executable: $GRITS_BED2VCF"
  [ -x "$GRITS_JAVA_HOME_BIN/java" ] && ok "java: $("$GRITS_JAVA_HOME_BIN/java" -version 2>&1 | head -1)" || fail "java 21 not found in $GRITS_JAVA_HOME_BIN"
  PATH=$GRITS_HTSLIB_BIN:$PATH command -v bcftools >/dev/null && ok "bcftools" || fail "bcftools not found"
fi
[ $bad = 0 ] && echo "All required resources found." || echo "Fix the FAIL lines (see references/inputs.md)."
exit $bad
