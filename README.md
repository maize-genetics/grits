# GRITS
**G**enetic **R**ecombination **I**mputation **T**ool **S**et

## Prerequisites

* [pixi](https://pixi.sh/latest/installation/) (v0.5+)
* NVIDIA-based GPU hardware with CUDA (v12+)
  + For `gpu` environment only (_see next section_)
* Linux or macOS OS environment preferred


## Setup

**Retrieve project code**

```bash
git clone https://github.com/maize-genetics/grits.git
cd grits
```

**Option A (_preferred_) - set up virtual environment (pixi)**

```bash
# For CPU only
pixi install

# For Linux + GPU machines
pixi install --environment gpu
```

**Option B - set up virtual environment (conda)**

```bash
conda env create -f environment.yml
```

## Quick start: impute a maize sample

GRITS calls, along each chromosome, which pair of the 25 maize founders (B73 + 24 NAM) your sample's two
haplotypes look most like. Input: FASTQ reads (short reads, ~0.1x-1x). Output: one BED of founder pairs per
chromosome, and optionally a VCF. Full guide: [docs/run-sample.md](docs/run-sample.md).

**1. Build the aligner** (ropebwt3 with the GRITS `refmap` options):

```bash
git clone https://github.com/maize-genetics/ropebwt3-phg
cd ropebwt3-phg && git checkout develop && make && cd ..
```

**2. Set up GRITS** (Linux, NVIDIA GPU with ~30 GB free recommended):

```bash
git clone https://github.com/maize-genetics/grits
cd grits && pixi install                   # default env; its torch already supports CUDA GPUs
```

**3. Get the data files.** All three must come from the same 25-founder panel:
- the model checkpoint `v6cfullfe-stage2B/last.ckpt`, from the Buckler lab (it can't be rebuilt);
- the v2 25-founder FM-index (`.fmd` + `.fmd.ssa` + `.fmd.len.gz`), from the lab, or rebuilt as described in
  the full guide;
- the lift, built once from the index in about 4 min:
  `ropebwt3 lift --ref-prefix=B73 -k 50 -s 200 -t 16 -o k50_s200.lift INDEX.fmd B73.fa`

**4. Point GRITS at them and check:**

```bash
export GRITS_WORKDIR=$PWD                    # required; the variables below override the defaults under it
export GRITS_ROPEBWT3=/path/to/ropebwt3-phg/ropebwt3
export GRITS_INDEX=/path/to/maizeFastaIndex_SampleContig_v2.fmd
export GRITS_LIFT=/path/to/k50_s200.lift
export GRITS_CKPT=/path/to/v6cfullfe-stage2B/last.ckpt
export GRITS_PYTHON_ENV=$PWD/.pixi/envs/default
S=.claude/skills/grits-run-sample/scripts
$S/check_setup.sh                            # must end with "All required resources found."
```

**5. Run:**

```bash
$S/align_sample.sh out/NAME 8 NAME_R1.fq.gz NAME_R2.fq.gz        # CPU, ~3 min at 0.1x, ~20 min at 1x
source $S/grits.env
LD_LIBRARY_PATH=$GRITS_LD $GRITS_PY $S/decode_sample.py --dir out/NAME --sample NAME   # GPU, 1-2 min
$S/bed_to_vcf.sh out/NAME NAME                                     # optional VCF, ~45 min
```

Results are in `out/NAME/bed/NAME_<chrom>_imputed.bed` (`chrom start end parent1 parent2`) and
`out/NAME/decode_summary.txt`. Check its first line, e.g. `route: ... hybrid=False`, against what you know
about the sample. For expected accuracy and how to read the calls, see [docs/run-sample.md](docs/run-sample.md).
With [Claude Code](https://claude.com/claude-code), you can instead ask Claude to run GRITS on your sample;
the `grits-run-sample` skill runs these same steps.

## Run (CLI tool)

Conda:
```bash
python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```

Pixi:
```bash
pixi run -- python impute.py --input <input_file> --output <output_file> --model <imputation_method>
```

# Development (GUI app - _experimental_)

## Prerequisites

* Node.js
* Rust (with `wasm32-unknown-unknown` target for web builds)
* [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) (for web builds only)

## Tauri Desktop App

If running this for the first time:

```bash
# First install pixi
# Next, run the pixi install script
pixi install

# Initialize npm environment
npm install

# Set up wheelhouse
npm run download-python
npm run build-wheelhouse

# Run the dev container of the Tauri app
npm run tauri dev
```

For subsequent runs, just use `npm run tauri dev`.

## Static Web App (PS4G / BED Visualization)

The PS4G and BED visualization components can be built as a standalone
static web application. File parsing runs entirely client-side via Rust
compiled to WebAssembly -- no server required.

### One-time setup

```bash
# Add the WASM compilation target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install npm dependencies
npm install
```

### Development

```bash
# Start the Vite dev server in web mode (port 3001)
npm run dev:web
```

### Production build

```bash
# Build WASM + TypeScript + Vite to dist-web/
npm run build:web

# Preview the production build locally
npm run preview:web
```

The `dist-web/` directory is a fully static site that can be deployed to
GitHub Pages, Netlify, Vercel, S3, or any static file host.

### Build commands reference

| Command              | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `npm run dev`        | Vite dev server for Tauri (port 3000)                              |
| `npm run dev:web`    | Vite dev server for web (port 3001)                                |
| `npm run build`      | Production frontend build for Tauri (`dist/`)                      |
| `npm run build:web`  | Full web pipeline: WASM + TS + Vite (`dist-web/`)                  |
| `npm run build:wasm` | Build only the WASM module (`src/wasm/pkg/`)                       |
| `npm run preview:web`| Serve the `dist-web/` production build locally                     |
| `npm run tauri dev`  | Launch the full Tauri desktop app in development mode              |

# Testing

## Run python unit tests with HTML output (CLI)

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```





