from pathlib import Path, PurePosixPath

# Every local path below is anchored to this file, not to the working
# directory, so `python tools/evaluate.py` behaves the same from anywhere.
PROJECT_DIR = Path(__file__).resolve().parent


def _model_dir(config):
    # The trained checkpoint, its tokenizers and the model card live together in
    # model/. Colab overrides model_folder with an absolute Drive path
    # (e.g. /content/drive/MyDrive/transformers-from-scratch_weights), which is
    # used verbatim.
    return config['model_folder']


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 1,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "nl",
        "model_folder": str(PROJECT_DIR / "model"),
        "model_basename": "tmodel_",
        "preload": "latest",
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
        "lr_schedule": "none",
        "warmup_steps": 4000,
        # how many validation sentences to decode+score at the end of an epoch
        "val_samples": 200,
    }

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