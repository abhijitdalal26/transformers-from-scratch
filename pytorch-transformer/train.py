from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import (get_config, get_weights_file_path, latest_weights_file_path,
                    bind_metrics_to_model_folder)

import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
import math
import random
import re
import time
from collections import Counter
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel as ByteLevelPreTok
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import Lowercase, Sequence

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# One row per epoch, appended and flushed immediately. TensorBoard is a
# debugging aid; this file is the durable record, so a Colab disconnect cannot
# lose an epoch of metrics.
METRICS_CSV = "training_metrics.csv"
METRICS_FIELDS = ['epoch', 'global_step', 'train_loss', 'bleu4', 'bp',
                  'p1', 'p2', 'p3', 'p4', 'cer', 'wer', 'n_val_sentences',
                  'minutes']


def append_metrics_row(path, row):
    """Append one epoch's metrics to `path` as CSV, creating the header if needed.

    Opened and closed per call, so the row is on disk before the next epoch
    starts. If the process dies mid-training, everything up to the previous
    epoch is still readable with no TensorBoard and no Python.

    A row for an epoch that is already in the file is REPLACED, not
    duplicated: a resumed run rewrites the epochs it redoes, so the CSV keeps
    one row per epoch.
    """
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not new_file:
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                existing = list(csv.DictReader(f))
        except Exception:
            existing = []
        kept = [r for r in existing if str(r.get('epoch')) != str(row.get('epoch'))]
        if len(kept) != len(existing):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
                writer.writeheader()
                for r in kept:
                    writer.writerow({k: r.get(k, '') for k in METRICS_FIELDS})
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in METRICS_FIELDS})


# --- Self-contained corpus BLEU-n -------------------------------------------------
# torchmetrics' BLEUScore is unusable here: on torchmetrics 2.21 it returns 0.0 even
# when predictions are identical to the references, and it renamed
# `n_gram_order` -> `n_gram`. These helpers follow the standard BLEU definition
# (Papineni et al. 2002) with corpus-level clipped counts, uniform weights and a
# brevity penalty, and tokenize with the sacreBLEU "13a" tokenizer.

def _tok_13a(text):
    text = text.replace('<skipped>', '').replace('-\n', '').replace('\n', ' ')
    text = text.replace('&amp;', '&')
    text = ' ' + text + ' '
    text = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', text)
    text = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', text)
    text = re.sub(r'([\.,])([^0-9])', r' \1 \2', text)
    text = re.sub(r'([0-9])(-)', r'\1 \2 ', text)
    return text.split()


def corpus_bleu(preds, refs, max_n=4):
    """Corpus BLEU-n in percent, plus the individual n-gram precisions and BP."""
    if len(preds) == 0:
        return 0.0, [0.0] * max_n, 0.0
    clipped = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for pred, ref in zip(preds, refs):
        p, r = _tok_13a(pred), _tok_13a(ref)
        hyp_len += len(p)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            p_grams = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
            r_grams = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            clipped[n - 1] += sum(min(c, r_grams[g]) for g, c in p_grams.items())
            total[n - 1] += sum(p_grams.values())
    precisions = [(clipped[i] / total[i]) if total[i] else 0.0 for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    if min(precisions) == 0.0:
        bleu = 0.0
    else:
        log_mean = sum(math.log(x) for x in precisions) / max_n
        bleu = bp * math.exp(log_mean) * 100
    return bleu, precisions, bp


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    generated = []
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        logits = model.project(out[:, -1])
        # never emit padding: mask it out before taking the argmax
        logits[:, pad_idx] = float('-inf')
        next_word = torch.argmax(logits, dim=1)

        if next_word.item() == eos_idx:
            break

        generated.append(next_word.item())
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

    # return only the generated tokens, without the leading [SOS]
    return torch.tensor(generated, dtype=torch.int64)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2, val_samples=None):
    """Decode `val_samples` sentences, print `num_examples` of them, and score
    the metrics over the *whole* decoded sample.

    The previous version broke out of the loop as soon as `num_examples`
    sentences had been printed and then computed the metrics on that tiny
    sample, so the logged BLEU/CER/WER were computed on 2 sentences from a
    shuffled loader -> pure noise. Printing and scoring are now decoupled.
    """
    model.eval()

    if val_samples is None:
        val_samples = len(validation_ds)
    val_samples = min(val_samples, len(validation_ds))

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for count, batch in enumerate(validation_ds, start=1):
            if count > val_samples:
                break

            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # only the first `num_examples` are printed, but all are scored
            if count <= num_examples:
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

        print_msg('-'*console_width)

    # Metrics are computed unconditionally, so the caller can log them to the
    # crash-proof metrics file even when there is no TensorBoard writer.
    cer = float(torchmetrics.CharErrorRate()(predicted, expected))
    wer = float(torchmetrics.WordErrorRate()(predicted, expected))
    bleu, precisions, bp = corpus_bleu(predicted, expected)
    precisions = [float(p) for p in precisions]

    if writer:
        writer.add_scalar('validation cer', cer, global_step)
        writer.add_scalar('validation wer', wer, global_step)
        # BLEU-n from the self-contained implementation; torchmetrics'
        # BLEUScore returns 0.0 in this environment
        writer.add_scalar('validation BLEU', bleu, global_step)
        for n, p in enumerate(precisions, start=1):
            writer.add_scalar(f'validation BLEU p{n}', p, global_step)
        writer.add_scalar('validation BLEU bp', bp, global_step)
        writer.add_scalar('validation sample size', len(predicted), global_step)
        writer.flush()

    print_msg(f"validation on {len(predicted)} sentences: "
              f"BLEU-4 = {bleu:.2f}  BP = {bp:.3f}  "
              f"precisions = {'/'.join(f'{p*100:.1f}' for p in precisions)}  "
              f"CER = {cer:.3f}  WER = {wer:.3f}")

    return {
        'bleu4': bleu,
        'p1': precisions[0],
        'p2': precisions[1],
        'p3': precisions[2],
        'p4': precisions[3],
        'bp': bp,
        'cer': cer,
        'wer': wer,
        'n_sentences': len(predicted),
    }


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """Build (or load from disk) the tokenizer for one language.

    Two variants, selected by config['tokenizer_type'] (set via TOKENIZER_TYPE env var):

    - 'wordlevel' (default): one token per whole word. Simple, but any word not
      seen during training becomes [UNK] — this is the known limitation we are
      testing BPE against.

    - 'bpe': Byte-Pair Encoding — starts from bytes and repeatedly merges the
      most frequent adjacent pair. This means:
        * No [UNK] at inference: any byte sequence can be represented.
        * Morphological sharing: "running", "runs", "ran" share subword pieces,
          so the model learns patterns, not just memorized whole words.
        * Much smaller vocab (8k vs 17-19k) → fewer embedding params, less overfitting.
      The ByteLevel pre-tokenizer maps raw text to printable byte tokens (GPT-2
      style); the Lowercase normalizer folds case BEFORE tokenization so "The"
      and "the" share the same subword IDs (the word-level run wasted ~981 vocab
      entries on case variants alone).
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if tokenizer_path.exists():
        # Tokenizer already trained and saved — just load it (fast path)
        return Tokenizer.from_file(str(tokenizer_path))

    # --- No saved tokenizer yet: train one from the corpus ---
    variant = config.get('tokenizer_type', 'wordlevel')

    if variant == 'bpe':
        # BPE tokenizer: ByteLevel pre-tokenizer + Lowercase normalizer.
        # unk_token is still [UNK] (same special-token order as wordlevel so
        # dataset.py's token_to_id calls work unchanged), but in practice it
        # is almost never emitted because BPE can represent any byte sequence.
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = Sequence([Lowercase()])
        tokenizer.pre_tokenizer = ByteLevelPreTok(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()  # decode IDs back to readable text
        trainer = BpeTrainer(
            vocab_size=8000,               # target vocab size (includes byte alphabet + specials)
            min_frequency=2,               # ignore pairs seen only once
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            show_progress=True,
        )
    else:
        # Original word-level tokenizer (baseline — unchanged behavior)
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
        )

    # Train on all sentences for this language (generator = memory efficient)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

    # Ensure the output directory exists (e.g. model/bpe/ may not exist yet)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"  [{variant}] tokenizer for '{lang}' saved to {tokenizer_path} "
          f"(vocab size = {tokenizer.get_vocab_size()})")
    return tokenizer


def get_ds(config, seed=None):
    if seed is None:
        seed = config.get('seed', 42)
    # make the split reproducible so held-out scores are comparable across runs
    random.seed(seed)
    torch.manual_seed(seed)

    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size],
                                            generator=torch.Generator().manual_seed(seed))

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # the validation loader must NOT be shuffled, otherwise every epoch is scored
    # on a different random subset and the metrics are not comparable
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
                              d_model=config['d_model'],
                              dropout=config.get('dropout', 0.1))
    return model


def get_lr_scale(config, step):
    """Learning-rate scale factor for the current step.

    'none' -> constant LR (the schedule this checkpoint was trained with).
    'noam' -> the schedule from "Attention Is All You Need":
              d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    'cosine' -> cosine decay from 1.0 down to min_lr_scale over the whole
                step budget of the run.
    'linear_decay' -> same, but straight line instead of a cosine.

    NOTE: the noam factor is ~0.044 at peak for d_model=512, so switching it on
    for a run that is resuming at step ~74k would drop the LR by more than an
    order of magnitude and effectively freeze training. That is why it is
    opt-in and defaults to 'none'.

    The decay schedules are a function of the ABSOLUTE global step and of the
    total step budget, never of "steps since this run started". A run resumed
    at step 43,490 of 86,980 therefore continues exactly the decay it would
    have had if it had never been stopped.
    """
    schedule = config.get('lr_schedule', 'none')
    if schedule == 'none':
        return 1.0
    d_model = config['d_model']
    warmup = config.get('warmup_steps', 4000)
    step = max(step, 1)
    if schedule == 'noam':
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
    if schedule in ('cosine', 'linear_decay'):
        total = config.get('total_steps') or 0
        if not total:
            return 1.0
        min_scale = config.get('min_lr_scale', 0.05)
        progress = min(max(step / total, 0.0), 1.0)
        if schedule == 'cosine':
            return min_scale + (1.0 - min_scale) * 0.5 * (1 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * (1.0 - progress)
    return 1.0


def set_lr(optimizer, config, step):
    """Set the LR for `step` (called every batch, and once right after resume)."""
    base_lr = config['lr']
    scale = get_lr_scale(config, step)
    for group in optimizer.param_groups:
        group['lr'] = base_lr * scale
    return base_lr * scale


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # Make sure the weights folder exists
    from config import _model_dir as _md
    Path(_md(config)).mkdir(parents=True, exist_ok=True)

    # The per-epoch metrics CSV must live next to the checkpoints. On Colab
    # that is Google Drive - a CSV written into the runtime directory is lost
    # when the session ends, which is how the first BPE run lost 6 of its 11
    # epoch rows. Locally the default results/<variant>/ path is kept.
    bind_metrics_to_model_folder(config)
    print(f"Metrics CSV: {config['metrics_file']}")

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    # The step budget of the whole run. A decay schedule is a function of the
    # absolute global step against this budget, so it survives a resume.
    config['steps_per_epoch'] = len(train_dataloader)
    if not config.get('total_steps'):
        config['total_steps'] = config['num_epochs'] * config['steps_per_epoch']
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state.get('global_step', 0)
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            # a weights-only checkpoint carries no Adam moments: Adam restarts
            # from zero state, which is fine for a fresh schedule.
            print('Checkpoint has no optimizer state - Adam restarts from scratch')
    else:
        print('No model to preload, starting from scratch')

    # ignore padding using the TARGET tokenizer's [PAD] id -- the labels come
    # from the target vocabulary, so the target [PAD] is what must be ignored
    pad_idx_tgt = tokenizer_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt,
                                  label_smoothing=config.get('label_smoothing', 0.1)).to(device)

    # apply the LR for the step we are resuming at, so the schedule is
    # continuous across a restart
    cur_lr = set_lr(optimizer, config, global_step)
    print(f"LR schedule = {config.get('lr_schedule', 'none')}, current LR = {cur_lr:.3e}")

    val_samples = config.get('val_samples', 200)
    # Debug-only: if set, stop after this many batches (used for quick smoke tests)
    max_train_steps = config.get('max_train_steps', None)
    # best-checkpoint tracking / early stopping
    best_bleu = -1.0
    epochs_without_improvement = 0

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        n_batches = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss_sum += loss.item()
            n_batches += 1

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the learning rate for this step, then the weights
            set_lr(optimizer, config, global_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Debug smoke-test exit: stop after max_train_steps batches
            if max_train_steps is not None and global_step >= max_train_steps:
                print(f"\n[smoke test] stopping after {max_train_steps} steps (max_train_steps set)")
                break

        # Run validation at the end of every epoch
        val = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, val_samples=val_samples)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'global_step': global_step
        }
        if not config.get('save_weights_only'):
            # Adam's two moments triple the file size (679 MB vs 294 MB). Only
            # the last few epochs fit on disk with them.
            ckpt['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(ckpt, model_filename)

        # Keep the best checkpoint by validation BLEU. "The last epoch" is
        # usually NOT the best epoch - this run memorised from epoch 10 on.
        if config.get('keep_best') and val['bleu4'] > best_bleu:
            best_bleu = val['bleu4']
            best_filename = get_weights_file_path(config, 'best')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'global_step': global_step, 'bleu4': val['bleu4']}, best_filename)
            print(f"new best validation BLEU {best_bleu:.4f} -> saved {best_filename}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Durable per-epoch record, written *after* the checkpoint so a row only
        # exists for an epoch that actually completed and was saved.
        append_metrics_row(config['metrics_file'], {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': round(loss_sum / max(n_batches, 1), 4),
            'bleu4': round(val['bleu4'], 4),
            'bp': round(val['bp'], 4),
            'p1': round(val['p1'], 4),
            'p2': round(val['p2'], 4),
            'p3': round(val['p3'], 4),
            'p4': round(val['p4'], 4),
            'cer': round(val['cer'], 4),
            'wer': round(val['wer'], 4),
            'n_val_sentences': val['n_sentences'],
            'minutes': round((time.time() - epoch_start) / 60, 1),
        })
        print(f"epoch {epoch:02d} logged to {config['metrics_file']}")

        # Early stop: this many epochs in a row without a better validation
        # BLEU means the run is memorising, not learning.
        patience = config.get('early_stop_patience', 0)
        if patience and epochs_without_improvement >= patience:
            print(f"\n[early stop] {epochs_without_improvement} epochs without a better "
                  f"validation BLEU (best {best_bleu:.4f}) - stopping at epoch {epoch}")
            break


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)