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
    || fail "ropebwt3 is too old (no --err-noclip): build maize-genetics/ropebwt3-phg branch develop"
fi
need "index" "$GRITS_INDEX"
need "index .ssa" "$GRITS_INDEX.ssa"
need "index .len.gz" "$GRITS_INDEX.len.gz"
need "lift (k50 s200)" "$GRITS_LIFT"
need "checkpoint" "$GRITS_CKPT"
n=0
if [ -e "$GRITS_INDEX.len.gz" ]; then
  n=$(zcat "$GRITS_INDEX.len.gz" | cut -f1 | sed 's/_.*//' | sort -u | wc -l)
  zcat "$GRITS_INDEX.len.gz" | cut -f1 | grep -q "^${GRITS_REF_PREFIX}_" \
    && ok "reference assembly '$GRITS_REF_PREFIX' is in the index" \
    || fail "no contigs named ${GRITS_REF_PREFIX}_* in the index: set GRITS_REF_PREFIX to the reference assembly's name"
fi
if [ -x "$GRITS_PY" ]; then
  LD_LIBRARY_PATH=$GRITS_LD:${LD_LIBRARY_PATH:-} "$GRITS_PY" -c \
    "import torch, pytorch_lightning, numpy, pandas; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu-only')" \
    > /tmp/grits_py_check.$$ 2>&1 && ok "python: $(cat /tmp/grits_py_check.$$)" || fail "python env lacks torch/pytorch_lightning/pandas: $(tail -1 /tmp/grits_py_check.$$)"
  rm -f /tmp/grits_py_check.$$
  if [ -e "$GRITS_CKPT" ]; then
    k=$(LD_LIBRARY_PATH=$GRITS_LD:${LD_LIBRARY_PATH:-} "$GRITS_PY" -c \
      "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['hyper_parameters']['num_parents'])" \
      "$GRITS_CKPT" 2>/dev/null)
    if [ -z "$k" ]; then fail "could not read the founder count from the checkpoint"
    elif [ "$n" -gt "$k" ]; then fail "index holds $n assemblies but the checkpoint takes at most $k: rebuild it from at most $k"
    elif [ "$n" = "$k" ]; then ok "index holds $n assemblies = checkpoint founder count"
    else ok "index holds $n assemblies; decode pads them to the checkpoint's $k (EXPERIMENTAL, not validated)"; fi
  fi
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
  B=$(command -v "$GRITS_BED2VCF") && ok "bed-to-vcf CLI: $B" || fail "bed-to-vcf CLI not found: $GRITS_BED2VCF (set GRITS_BED2VCF to the sample CLI)"
  TP=${GRITS_JAVA_HOME_BIN:+$GRITS_JAVA_HOME_BIN:}${GRITS_HTSLIB_BIN:+$GRITS_HTSLIB_BIN:}$PATH
  J=$(PATH=$TP command -v java) && ok "java: $("$J" -version 2>&1 | head -1)" || fail "java 21 not found (put it on PATH or set GRITS_JAVA_HOME_BIN)"
  for t in bcftools bgzip tabix; do PATH=$TP command -v $t >/dev/null && ok "$t" || fail "$t not found (put it on PATH or set GRITS_HTSLIB_BIN)"; done
fi
[ $bad = 0 ] && echo "All required resources found." || echo "Fix the FAIL lines (see references/inputs.md)."
exit $bad
