# GRITS Documentation

Welcome to the GRITS documentation! This directory contains comprehensive 
guides for building, installing, and using both the command-line interface 
(CLI) and graphical user interface (GUI) applications.

## Overview

GRITS (Genetic Recombination Imputation Tool Set) is a machine learning-based haplotype imputation tool that combines 
multiple approaches including:

- **BiMamba** - State space models for sequence imputation
- **ModernBERT** - Transformer models for genetic sequence modeling
- **HMM** - Hidden Markov Model implementation with Viterbi decoding
- **K-Nearest Neighbors** - Simple baseline imputation method

The project provides two interfaces:
1. **Python CLI Tool** - Command-line interface for batch processing
2. **Tauri Desktop Application** - Cross-platform GUI with visualization 
   capabilities


## Prerequisites
* [pixi](https://pixi.sh/latest/)


## Quick Start

### For CLI Users
If you want to use the command-line interface for batch processing:

```bash
# Clone the repository
git clone https://github.com/maize-genetics/grits.git
cd grits

# Install with pixi (recommended)

## CPU only
pixi install

## GPU option
pixi install --environment gpu


# Run imputation
pixi run -- python src/impute.py --input <ps4g_input.ps4g> --output <bed_file_output.bed> --model bimamba
```

### For GUI Users
Download the latest releases found [here](https://github.com/maize-genetics/grits/releases/tag/v0.0.18-test):

* macOS (`.dmg` release)
* Windows (`.msi` release)

> [!NOTE]
> A Linux GUI release is currently not supported. Please see the next section for steps on how to
> manually build if you are interested.

### For GUI Users (development and local build)
If you want to build the desktop application:

```bash
# Clone the repository
git clone https://github.com/maize-genetics/grits.git
cd grits

# Install dependencies
npm install
pixi install

# Run in development mode
npm run tauri dev

# Or build for production
npm run tauri build
```

## Documentation Files

### [CLI Usage Guide](https://github.com/maize-genetics/grits/wiki/CLI-Usage)
Complete guide for the Python command-line interface including:
- Installation with pixi or conda
- Command-line arguments and examples
- Model-specific usage (BiMamba, ModernBERT, HMM, KNN)
- Input/output file formats (PS4G, BED)
- Troubleshooting and performance tips

### [GUI Usage Guide](./gui_usage.md)
Comprehensive guide for the desktop application including:
- Cross-platform build instructions
- Development and production workflows
- Python integration and bootstrap system
- Data visualization features
- Platform-specific troubleshooting

### [Session Handoff](./HANDOFF.md)
Current project status and research history — start here for "what's
being worked on and why." Research/evaluation work lives under
[`experiments/`](../experiments/) as self-contained directories, each
with its own `README.md` (setup, exact commands, results); the most
recent `HANDOFF.md` entry links directly to the relevant ones.

## Architecture Overview

### CLI Application
```
Python CLI (src/impute.py)
├── BiMamba Models (src/bimamba/)
├── ModernBERT Models (src/modernBERT/)
├── HMM Implementation (src/hmm/)
├── File I/O (src/ps4g_io/, src/bed_io/)
└── Data Processing (src/python/)
```

### GUI Application
```
Tauri Desktop App
├── React Frontend (src/)
│   ├── D3.js Visualizations
│   └── TypeScript Utilities
├── Rust Backend (src-tauri/)
│   ├── System Info Commands
│   └── Python Integration
└── Embedded Python Runtime
    ├── ML Models
    └── Data Processing
```

## Key Features

### Machine Learning Models
- **BiMamba**: State space models optimized for sequence data
- **ModernBERT**: Transformer architecture for genetic imputation
- **HMM**: Traditional probabilistic approach with Viterbi decoding
- **KNN**: Distance-based baseline method

### Data Processing
- **Input**: PS4G format files (custom haplotype format)
- **Output**: Extended BED format files
- **Features**: Weighted data, gamete collapsing, contiguous region handling

### GUI Capabilities
- Real-time system information display
- GPU detection and utilization
- Interactive matrix visualizations
- Embedded Python environment management
- Cross-platform distribution (macOS, Windows)


## Development Environment

### Required Tools
- **Python 3.10+** - Core ML pipeline
- **Node.js** - Frontend development (GUI only)
- **Rust** - Backend development (GUI only)
- **Pixi** - Python environment management

### GPU Support
- **CUDA 12+** - For BiMamba and ModernBERT acceleration
- **Platform**: Linux and Windows (macOS support planned)
- **Models**: BiMamba and ModernBERT support GPU acceleration

## File Formats

### PS4G Input Format
Custom haplotype format containing:
- Genetic variation data with positional information
- Separate columns for chromosome (`refContig`) and binned position (`refPosBinned`)
- Position identifiers created as `{refContig}_{refPosBinned}`
- Metadata including sample information (gamete names, indices, counts)
- Support for weighted and unweighted data

### BED Output Format
Extended BED format including:
- Chromosome identifiers (`chrom` column with decoded contig names)
- Genomic position or interval information (`pos`, or `start`/`end` for collapsed output)
- Imputed parental assignments (`parent1`, `parent2` columns)
- Optional collapsed format merging contiguous regions with identical predictions
- Compatibility with standard genomic analysis tools

## Testing

Both CLI and GUI applications include comprehensive test suites:

```bash
# Run Python tests
pixi run pytest

# Tests cover:
# - PS4G file I/O
# - PyTorch data loaders
# - Model integration
# - Array utilities
```

## Deployment

### CLI Distribution
- Source code distribution via GitHub
- Users build locally with pixi/conda
- Cross-platform compatibility (Linux, macOS, Windows)

### GUI Distribution
- Self-contained installers with embedded Python
- Platform-specific packages (.dmg for macOS, .msi for Windows)
- Ad-hoc code signing for macOS compatibility




