from pathlib import Path, PurePosixPath
import os

# Every local path below is anchored to this file, not to the working
# directory, so `python tools/evaluate.py` behaves the same from anywhere.
PROJECT_DIR = Path(__file__).resolve().parent


def _model_dir(config):
    # The trained checkpoint, its tokenizers and the model card live together in
    # model/. Colab overrides model_folder with an absolute Drive path
    # (e.g. /content/drive/MyDrive/transformers-from-scratch_weights), which is
    # used verbatim.
    return config['model_folder']


def apply_tokenizer_variant(config):
    """Redirect artifact paths to a tokenizer-variant-specific subfolder.

    Two tokenizer variants coexist without ever touching each other's files:
      - 'wordlevel' (default): model/, results/wordlevel/, runs/tmodel
      - 'bpe':                 model/bpe/, results/bpe/, runs/tmodel_bpe

    This is called automatically at the end of get_config(). On Colab, where
    model_folder and tokenizer_file are overridden with Drive paths, call this
    again *after* those overrides so the remaining paths (metrics_file,
    experiment_name) still point at the variant's own folder.

    The model_basename ('tmodel_') is intentionally unchanged — the two
    variants are distinguished by their folder, not their filename.
    """
    if config.get('tokenizer_type', 'wordlevel') == 'bpe':
        config['model_folder']    = str(PROJECT_DIR / 'model' / 'bpe')
        config['tokenizer_file']  = str(PROJECT_DIR / 'model' / 'bpe' / 'tokenizer_{0}.json')
        config['metrics_file']    = str(PROJECT_DIR / 'results' / 'bpe' / 'training_metrics.csv')
        config['experiment_name'] = str(PROJECT_DIR / 'runs' / 'tmodel_bpe')
    return config


def bind_metrics_to_model_folder(config):
    """Put the per-epoch metrics CSV next to the checkpoints.

    On Colab the checkpoints live in Google Drive, so the CSV has to live
    there too: `results/<variant>/training_metrics.csv` is inside the
    ephemeral runtime and is lost when the session ends (exactly what
    happened to the first BPE run - only 5 of 11 epochs survived).

    Only rebinds when the model folder is OUTSIDE this project, i.e. a Drive
    path. A local run keeps writing to results/<variant>/.
    """
    folder = str(config['model_folder'])
    if folder.replace('\\', '/').startswith(str(PROJECT_DIR).replace('\\', '/')):
        return config  # local run - keep the default location
    config['metrics_file'] = _join_model_path(folder, 'training_metrics.csv')
    return config


def get_config():
    config = {
        "batch_size": 8,
        "num_epochs": int(os.environ.get("NUM_EPOCHS", 1)),
        "lr": float(os.environ.get("LR", 10**-4)),
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "nl",
        "model_folder": str(PROJECT_DIR / "model"),
        "model_basename": "tmodel_",
        "preload": os.environ.get("PRELOAD", "latest"),
        "tokenizer_file": str(PROJECT_DIR / "model" / "tokenizer_{0}.json"),
        "experiment_name": str(PROJECT_DIR / "runs" / "tmodel"),
        # Durable per-epoch metrics (CSV). TensorBoard is for watching a run
        # live; this is the record that survives a disconnect, so the peak
        # epoch can always be identified after the fact.
        "metrics_file": str(PROJECT_DIR / "results" / "wordlevel" / "training_metrics.csv"),
        # seed for the train/val split so held-out scores are reproducible
        "seed": 42,
        # 'none' = constant lr (what this checkpoint was trained with)
        # 'noam' = "Attention Is All You Need" schedule (opt-in: only safe when
        #         training from scratch, it would crush the lr on a resume)
        "lr_schedule": os.environ.get("LR_SCHEDULE", "none"),
        "warmup_steps": 4000,
        # ---- knobs for the "fix the recipe" run ---------------------------
        # cosine / linear_decay: decay the lr across the whole step budget.
        # The scale is a function of the ABSOLUTE global step, so a run that
        # is resumed half way continues the same decay it would have had.
        "min_lr_scale": float(os.environ.get("MIN_LR_SCALE", 0.05)),
        # 0 = derive from num_epochs * steps_per_epoch (set in train.py)
        "total_steps": int(os.environ.get("TOTAL_STEPS", 0)),
        # regularisation
        "dropout": float(os.environ.get("DROPOUT", 0.1)),
        "label_smoothing": float(os.environ.get("LABEL_SMOOTHING", 0.1)),
        # stop after this many epochs without a validation BLEU improvement
        # (0 = never stop early)
        "early_stop_patience": int(os.environ.get("EARLY_STOP_PATIENCE", 0)),
        # keep a copy of the best checkpoint as <basename>best.pt
        "keep_best": int(os.environ.get("KEEP_BEST", 1)),
        # save weights only (no Adam moments) -> ~294 MB instead of ~679 MB,
        # so every epoch fits on disk
        "save_weights_only": int(os.environ.get("SAVE_WEIGHTS_ONLY", 0)),
        # how many validation sentences to decode+score at the end of an epoch
        "val_samples": int(os.environ.get("VAL_SAMPLES", 200)),
        # Tokenizer variant: 'wordlevel' (default, baseline) or 'bpe'.
        # Set via the TOKENIZER_TYPE environment variable — no code changes
        # needed to switch variants. Affects only artifact paths (see
        # apply_tokenizer_variant below); the model architecture is identical.
        "tokenizer_type": os.environ.get("TOKENIZER_TYPE", "wordlevel"),
        # Debug-only: set to an int (e.g. 50) to stop training after that many
        # batches. Used for a quick local smoke test; leave as None for real runs.
        "max_train_steps": None,
    }
    # Remap artifact paths to the variant's own subfolder (no-op for wordlevel)
    config = apply_tokenizer_variant(config)
    # Explicit overrides win over everything. A checkpoint folder and the
    # tokenizers that belong to it can live anywhere - e.g. the BPE run sits
    # in drive_bpe/transformers-from-scratch_weights_bpe/ - and the tools
    # need to be pointed at that pair, not at the default local folders.
    #   MODEL_FOLDER   -> config['model_folder']   (where the .pt files live)
    #   TOKENIZER_FILE -> config['tokenizer_file'] ({0} = 'en' or 'nl')
    if os.environ.get("MODEL_FOLDER"):
        config['model_folder'] = os.environ["MODEL_FOLDER"]
    if os.environ.get("TOKENIZER_FILE"):
        config['tokenizer_file'] = os.environ["TOKENIZER_FILE"]
    return config

def _join_model_path(model_folder: str, filename: str) -> str:
    # Colab Drive paths are POSIX even when tested on Windows — preserve leading /
    if model_folder.startswith("/"):
        return str(PurePosixPath(model_folder) / filename)
    return str(Path(model_folder) / filename)

def get_weights_file_path(config, epoch: str):
    model_folder = _model_dir(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return _join_model_path(model_folder, model_filename)

# Find the newest checkpoint in the model folder (model/tmodel_*.pt)
def latest_weights_file_path(config):
    model_folder = _model_dir(config)
    model_filename = f"{config['model_basename']}*"
    # Glob needs OS-correct Path; for POSIX Drive path on Linux this is fine
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def resolve_checkpoint(config, spec):
    """Turn a user-supplied checkpoint spec into a path.

    Accepts '17', 'tmodel_17', 'tmodel_17.pt', or any path containing a
    separator (used verbatim, so checkpoints can live outside model/).
    """
    spec = str(spec)
    if "/" in spec or "\\" in spec:
        return spec
    stem = spec
    if stem.endswith(".pt"):
        stem = stem[: -len(".pt")]
    base = config["model_basename"]
    if stem.startswith(base):
        stem = stem[len(base):]
    candidate = Path(_model_dir(config)) / f"{base}{stem}.pt"
    if candidate.exists():
        return str(candidate)
    # fall back to a glob so zero-padded / unpadded names both work
    hits = sorted(Path(_model_dir(config)).glob(f"{base}{stem.strip('0') or '0'}*.pt"))
    if hits:
        return str(hits[-1])
    raise FileNotFoundError(f"No checkpoint matching {spec!r} in {_model_dir(config)}")