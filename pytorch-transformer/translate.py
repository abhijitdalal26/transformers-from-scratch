import config as cfgmod
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
import torch
import sys

# The checkpoint is 838 MB, so reloading it for every sentence dominates runtime.
# Cache the last-loaded (checkpoint -> model, tokenizers) so batch translations
# only pay the load cost once.
_LOADED = {}

def _load(config, checkpoint, device):
    model_filename = checkpoint or cfgmod.latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(f"No checkpoint found in {config['model_folder']}. Please train the model first.")
    key = (str(model_filename), str(device), config["seq_len"], config["d_model"],
           config['tokenizer_file'].format(config['lang_src']),
           config['tokenizer_file'].format(config['lang_tgt']))
    if key not in _LOADED:
        tokenizer_src = Tokenizer.from_file(key[4])
        tokenizer_tgt = Tokenizer.from_file(key[5])
        model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
                                  config["seq_len"], config["seq_len"], d_model=config["d_model"]).to(device)
        state = torch.load(model_filename, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        _LOADED.clear()
        _LOADED[key] = (model, tokenizer_src, tokenizer_tgt)
    return _LOADED[key]

def translate(sentence: str, config=None, checkpoint=None):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config is None:
        config = cfgmod.get_config()
    model, tokenizer_src, tokenizer_tgt = _load(config, checkpoint, device)

    # if the sentence is a number use it as an index to the test set
    label = ""
    if type(sentence) == int or sentence.isdigit():
        id = int(sentence)
        ds = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        ds = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
        sentence = ds[id]['src_text']
        label = ds[id]["tgt_text"]
    seq_len = config['seq_len']

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)
        # Truncate to the seq_len budget (2 slots go to <s> and </s>), otherwise
        # the negative padding count below produces an over-long tensor and the
        # positional encoding fails to broadcast. Mirrors the truncation in
        # dataset.py so training and inference see the same limit.
        source_ids = source.ids[:seq_len - 2]
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source_ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source_ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        if label != "": print(f"{f'ID: ':>12}{id}") 
        print(f"{f'SOURCE: ':>12}{sentence}")
        if label != "": print(f"{f'TARGET: ':>12}{label}") 
        print(f"{f'PREDICTED: ':>12}", end='')

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())
    
if __name__ == "__main__":
    # optional --checkpoint picks a specific epoch, e.g. --checkpoint 17
    argv = sys.argv[1:]
    checkpoint = None
    if "--checkpoint" in argv:
        i = argv.index("--checkpoint")
        from config import resolve_checkpoint
        checkpoint = resolve_checkpoint(cfgmod.get_config(), argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    translate(argv[0] if argv else "I am not a very good a student.", checkpoint=checkpoint)