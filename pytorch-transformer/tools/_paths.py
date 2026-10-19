"""Path bootstrap for the tools in this directory.

The tools are analysis and reporting scripts that sit *around* the model, not
part of it. They import the implementation from the parent directory
(`config`, `model`, `train`, `dataset`, `translate`, `evaluate`), so this module
puts that parent on `sys.path` before they do, and defines every path they read
or write as an absolute path.

That means a tool can be run from any working directory:

    python tools/evaluate.py --samples 200
    python pytorch-transformer/tools/make_graphs.py

Every tool must import this module *first*, before any project import.
"""

import sys
from pathlib import Path

# .../pytorch-transformer
PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# the trained checkpoint, its tokenizers and the model card
MODEL_DIR = PROJECT_DIR / "model"
# evaluation output: metrics json + sample tables
RESULTS_DIR = PROJECT_DIR / "results" / "wordlevel"
# figures
GRAPHS_DIR = PROJECT_DIR / "graphs" / "wordlevel"
ATTENTION_DIR = GRAPHS_DIR / "attention"
# raw TensorBoard event files downloaded from Colab - the source of the curves
RUNS_DIR = PROJECT_DIR / "drive" / "transformers-from-scratch_runs" / "tmodel_en_nl"
# the plan for the not-yet-trained BPE run
BPE_DIR = PROJECT_DIR / "graphs" / "bpe"
