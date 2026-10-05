import json

def md(src): return {"cell_type": "markdown", "id": None, "metadata": {}, "source": src.strip().splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "id": None, "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip().splitlines(keepends=True)}

cells = []

# ---------------- Section 1
cells.append(md(r"""
# Make More: a character-level Transformer (GPT) from scratch

## 1. What are we building?

We are training a small **character-level, GPT-style, decoder-only Transformer** that learns to generate names similar to the names in `names.txt`.

```text
names.txt
   ↓
characters
   ↓
integer token IDs
   ↓
training pairs X, Y
   ↓
token embeddings + position embeddings
   ↓
Transformer blocks
   ↓
final LayerNorm
   ↓
linear LM head
   ↓
logits
   ↓
cross-entropy loss
   ↓
backpropagation + AdamW
   ↓
trained model
   ↓
autoregressive generation
```

### What is the task?

The model learns

\[
P(x_t \mid x_1,x_2,\ldots,x_{t-1})
\]

i.e. *given the previous characters, predict the next character*. This is a **character-level language model**.

> This is a decoder-only Transformer because its self-attention is causal: each position can only attend to itself and positions before it.
"""))

# ---------------- Section 2
cells.append(md(r"""
## 2. Imports and configuration

The usual imports. We also pick the device (GPU if Colab gives us one, else CPU).
"""))
cells.append(code(r"""
import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(3407)
print('device:', device)
"""))

cells.append(md(r"""
### ModelConfig

Same config as `makemore.py`. We use the original small defaults: `n_layer=4, n_embd=64, n_head=4`.
"""))
cells.append(code(r"""
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4
"""))

cells.append(md(r"""
- `vocab_size` = number of possible character tokens including the special token
- `block_size` = maximum sequence length
- `n_embd` = hidden/embedding dimension
- `n_head` = number of attention heads
- `n_layer` = number of Transformer blocks

Each attention head sees a slice of the embedding:

\[
d_{\text{head}} = \frac{n_{\text{embd}}}{n_{\text{head}}} = \frac{64}{4} = 16
\]
"""))

# ---------------- Section 3
cells.append(md(r"""
## 3. Dataset and character-level tokenization

Each line of `names.txt` is one name:

```text
emma
olivia
ava
liam
```

The model needs integers, so we build a **character-level** vocabulary (not BPE/subwords):

```text
stoi = string → integer
itos = integer → string
```

with `0` reserved for the special `<START>`/`<STOP>` token, and `a → 1, b → 2, ...`.

If the file is missing (e.g. fresh Colab session) we download it from Karpathy's repo.
"""))
cells.append(code(r"""
import urllib.request

input_file = 'names.txt'
if not os.path.exists(input_file):
    url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    print('downloading', url)
    urllib.request.urlretrieve(url, input_file)

with open(input_file, 'r') as f:
    data = f.read()
words = data.splitlines()
words = [w.strip() for w in words] # get rid of any leading or trailing white space
words = [w for w in words if w]    # get rid of any empty strings
chars = sorted(list(set(''.join(words)))) # all the possible characters
max_word_length = max(len(w) for w in words)
print(f"number of examples in the dataset: {len(words)}")
print(f"max word length: {max_word_length}")
print(f"number of unique characters in the vocabulary: {len(chars)}")
print("vocabulary:", ''.join(chars))
"""))

# ---------------- Section 4
cells.append(md(r"""
## 4. How X and Y are created

This is the most important training detail. For a name like `cat`:

```text
X: <START>  c   a   t
Y: c        a   t   <STOP>
```

The target is the input **shifted by one**. The model learns:

\[
P(x_t \mid x_{<t})
\]

and for a full sequence the autoregressive factorization is:

\[
\boxed{P(x_1,\ldots,x_T)=\prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1})}
\]

That product is the central training objective.
"""))
cells.append(code(r"""
class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y
"""))

cells.append(md(r"""
Padded positions use `-1`, and the loss ignores them via `ignore_index=-1`, so padding never contributes to training.

Sanity check on one example:
"""))
cells.append(code(r"""
demo = CharDataset(['cat'], chars, max_word_length)
x, y = demo[0]
print('word:', 'cat')
print('X:', x.tolist())
print('Y:', y.tolist())
print('decoded X:', repr(demo.decode([i for i in x.tolist() if i > 0])))
print('decoded Y:', repr(demo.decode([i for i in y.tolist() if i > 0])))
"""))

# ---------------- Section 5
cells.append(md(r"""
## 5. Train/test split

```text
training set → used to update weights
test set     → used to check generalization
```

We never train on the test set; it only measures whether the model learned patterns that transfer to unseen names.
"""))
cells.append(code(r"""
test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
rp = torch.randperm(len(words)).tolist()
train_words = [words[i] for i in rp[:-test_set_size]]
test_words = [words[i] for i in rp[-test_set_size:]]
train_dataset = CharDataset(train_words, chars, max_word_length)
test_dataset = CharDataset(test_words, chars, max_word_length)
print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

vocab_size = train_dataset.get_vocab_size()
block_size = train_dataset.get_output_length()
print(f"dataset determined that: vocab_size={vocab_size}, block_size={block_size}")
"""))

# ---------------- Section 6
cells.append(md(r"""
## 6. Token embeddings

A token ID is just an integer — it carries no numerical meaning. The embedding table maps it to a learned vector:

\[
E \in \mathbb{R}^{V \times d}, \qquad e_i = E[i]
\]

```text
token ID
   ↓
embedding lookup
   ↓
d-dimensional vector
```

Token vs embedding:

```text
token ID  = integer
embedding = learned vector
```

Shapes: `idx: (B, T)` → token embeddings `(B, T, C)`, where `B` = batch, `T` = sequence length, `C` = embedding dimension.
"""))
cells.append(code(r"""
wte = nn.Embedding(vocab_size, 64)
idx = torch.zeros(4, block_size, dtype=torch.long)  # pretend we have 4 sequences
tok = wte(idx)
print('idx shape:     ', idx.shape)
print('token embedding:', tok.shape, ' # (B, T, C)')
"""))

# ---------------- Section 7
cells.append(md(r"""
## 7. Positional embeddings

Attention by itself does not know *where* a token is in the sequence. So we add a learned position embedding:

```python
wpe = nn.Embedding(block_size, n_embd)
```

\[
\boxed{x_i = e_i + p_i} \qquad\text{or}\qquad \boxed{X = E_{\text{token}} + E_{\text{position}}}
\]

Shapes: token embeddings `(B, T, C)` + position embeddings `(1, T, C)` → `(B, T, C)`.
"""))
cells.append(code(r"""
wpe = nn.Embedding(block_size, 64)
pos = torch.arange(0, block_size, dtype=torch.long).unsqueeze(0)  # (1, T)
pos_emb = wpe(pos)
print('pos_emb shape:', pos_emb.shape)
x = tok + pos_emb
print('x shape:      ', x.shape, ' # (B, T, C)')
"""))

# ---------------- Section 8
cells.append(md(r"""
## 8. Causal self-attention: Q, K, V

The key question: *for the current character, which previous characters should the model pay attention to?*

For input matrix \(X\):

\[
\boxed{Q = XW_Q} \qquad \boxed{K = XW_K} \qquad \boxed{V = XW_V}
\]

- Query = what this position is looking for
- Key = what each position offers for matching
- Value = information that gets passed along

Karpathy computes all three with one linear layer and splits:

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```
"""))

cells.append(md(r"""
### Attention score formula

\[
\boxed{S = \frac{QK^\top}{\sqrt{d_k}}}
\]

For one head the shapes multiply as:

```text
Q:  (B, T, d_head)
K:  (B, T, d_head)
Kᵀ: (B, d_head, T)
        ↓
QKᵀ: (B, T, d_head) @ (B, d_head, T) = (B, T, T)
```

Every token gets a score against every other token. We divide by \(\sqrt{d_k}\) because dot products grow in magnitude as \(d_k\) increases; scaling keeps the logits at a manageable size before softmax.
"""))

cells.append(md(r"""
### Causal mask

Position \(t\) may only see positions \(\le t\):

\[
M=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
\]

```text
position 1 → can see position 1
position 2 → can see positions 1,2
position 3 → can see positions 1,2,3
position 4 → can see positions 1,2,3,4
```

The future must be hidden. Since \(e^{-\infty}=0\), masked positions get probability 0 after softmax. This mask is what makes this an **autoregressive decoder-style** Transformer rather than bidirectional encoder-style attention.
"""))
cells.append(code(r"""
mask = torch.tril(torch.ones(4, 4))
print(mask)
"""))

cells.append(md(r"""
### Softmax and the full attention formula

\[
\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

\[
\boxed{\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V}
\]

\(A=\text{softmax}(\cdot)\) holds the attention weights; \(Y=AV\) is the weighted combination of values.
"""))

cells.append(md(r"""
### Multi-head attention

With `n_embd=64, n_head=4` we get \(d_{\text{head}}=16\). Each head runs its own attention:

\[
H_i=\operatorname{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}+M\right)V_i,\qquad \boxed{H=\operatorname{Concat}(H_1,\ldots,H_h)},\qquad \boxed{Y=HW_O}
\]

```text
Q/K/V before split : (B, T, C)
Q/K/V after split  : (B, H, T, D)
attention scores   : (B, H, T, T)
attention output   : (B, H, T, D)
merged output      : (B, T, C)
```
"""))
cells.append(code(r"""
class NewGELU(nn.Module):
    '''
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    '''
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    '''
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    '''

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
"""))

cells.append(md(r"""
Sanity check — run a tiny random tensor through attention and watch the shapes:
"""))
cells.append(code(r"""
config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=4, n_head=4, n_embd=64)
attn = CausalSelfAttention(config)
x = torch.randn(2, block_size, 64)
q, k, v = attn.c_attn(x).split(64, dim=2)
print('input x      :', tuple(x.shape))
print('Q/K/V        :', tuple(q.shape))
q = q.view(2, block_size, 4, 16).transpose(1, 2)
print('Q reshaped   :', tuple(q.shape), ' # (B, H, T, D)')
print('attn scores  :', tuple((q @ q.transpose(-2,-1)).shape), ' # (B, H, T, T)')
print('attn out     :', tuple(attn(x).shape), ' # (B, T, C)')
"""))

# ---------------- Section 9 GELU
cells.append(md(r"""
## 9. GELU

The exact approximation used by Karpathy:

\[
\boxed{\operatorname{GELU}(x)=\frac{1}{2}x\left[1+\tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right]}
\]

GELU is the nonlinear activation inside the Transformer feed-forward network.
"""))

# ---------------- Section 10 Block
cells.append(md(r"""
## 10. The Transformer block

One block is pre-LayerNorm + residual attention, then pre-LayerNorm + residual MLP:

\[
\boxed{x' = x + \operatorname{MHA}(\operatorname{LN}(x))}
\]

\[
\boxed{x'' = x' + \operatorname{MLP}(\operatorname{LN}(x'))}
\]

The MLP:

\[
\boxed{\operatorname{MLP}(x) = W_2\,\operatorname{GELU}(W_1 x + b_1) + b_2}
\]

With the defaults, the MLP width is \(4d = 4(64) = 256\), i.e. `64 → 256 → 64`.

Roles:

```text
Attention → communication between token positions
MLP       → transforms each token representation independently
Residual  → preserves/adds the previous representation
LayerNorm → normalizes the representation before sublayers
```
"""))
cells.append(code(r"""
class Block(nn.Module):
    ''' an unassuming Transformer block '''

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
"""))

# ---------------- Section 11 LayerNorm
cells.append(md(r"""
## 11. LayerNorm

For a vector \(x \in \mathbb{R}^d\):

\[
\mu=\frac{1}{d}\sum_i x_i, \qquad \sigma^2=\frac{1}{d}\sum_i (x_i-\mu)^2
\]

\[
\boxed{\operatorname{LN}(x)=\gamma\,\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta}
\]

`γ` and `β` are learned; normalization happens across the feature dimension for each token.
"""))

# ---------------- Section 12 Full model
cells.append(md(r"""
## 12. The full Transformer

```text
Token IDs
   ↓
Token Embedding
   +
Position Embedding
   ↓
Transformer Block 1
   ↓
Transformer Block 2
   ↓
Transformer Block 3
   ↓
Transformer Block 4
   ↓
Final LayerNorm
   ↓
LM Head
   ↓
Logits
```
"""))
cells.append(code(r"""
class Transformer(nn.Module):
    ''' Transformer Language Model, exactly as seen in GPT-2 '''

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device_ = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device_).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
"""))

cells.append(md(r"""
- `wte` — token embedding table
- `wpe` — position embedding table
- `h` — the stack of Transformer blocks
- `ln_f` — final LayerNorm
- `lm_head` — linear projection to vocabulary logits

Sanity check on the full model:
"""))
cells.append(code(r"""
model = Transformer(config).to(device)
X_demo, Y_demo = next(iter(DataLoader(train_dataset, batch_size=4)))
X_demo, Y_demo = X_demo.to(device), Y_demo.to(device)
logits, loss = model(X_demo, Y_demo)
print('logits shape:', tuple(logits.shape), ' # (B, T, V)')
print('loss:', loss.item())
"""))

# ---------------- Section 13 logits
cells.append(md(r"""
## 13. LM head and logits

The Transformer output is still a hidden representation. The LM head converts it to vocabulary-sized logits:

\[
H \in \mathbb{R}^{B \times T \times d}, \qquad Z = HW_{\text{lm}}, \qquad W_{\text{lm}} \in \mathbb{R}^{d \times V}, \qquad Z \in \mathbb{R}^{B \times T \times V}
\]

> Each position now has one logit for every possible character.
>
> Logits vs probabilities: `logits = raw model scores`, `probabilities = softmax(logits)`.
"""))

# ---------------- Section 14 loss
cells.append(md(r"""
## 14. Cross-entropy loss

Logits become probabilities via softmax:

\[
\boxed{P(y=i)=\frac{e^{z_i}}{\sum_j e^{z_j}}}
\]

For the correct target \(y\): \(L=-\log P(y)\). Over the whole sequence:

\[
\boxed{L(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})}
\]

High probability for the correct character → low loss; low probability → high loss. This is exactly the `F.cross_entropy(...)` line in `forward`.
"""))

# ---------------- Section 15 AdamW
cells.append(md(r"""
## 15. Backpropagation and AdamW

Training wants:

\[
\boxed{\theta^*=\arg\min_\theta L(\theta)}
\]

Backpropagation computes \(\boxed{\frac{\partial L}{\partial \theta}}\), then AdamW updates the weights. Conceptually, with \(g_t=\nabla_\theta L_t\):

\[
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t, \qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\hat m_t=\frac{m_t}{1-\beta_1^t}, \qquad \hat v_t=\frac{v_t}{1-\beta_2^t}
\]

\[
\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

We do **not** implement Adam manually — `torch.optim.AdamW` does it. The equations only explain what the optimizer is doing.
"""))

# ---------------- Section 16 training loop
cells.append(md(r"""
## 16. Training loop

```text
get batch
   ↓
X, Y
   ↓
model(X, Y) → logits, loss
   ↓
zero gradients
   ↓
loss.backward()
   ↓
optimizer.step()
   ↓
repeat
```

We also keep the evaluation helpers: `model.train()` puts the model in training mode, `model.eval()` switches to evaluation mode, and we periodically evaluate so we can watch train/test loss.
"""))
cells.append(code(r"""
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    '''
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    '''
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(model, num=10):
    ''' samples from the model and pretty prints the decoded samples '''
    model.eval()
    X_init = torch.zeros(num, 1, dtype=torch.long).to(device)
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, do_sample=True).to('cpu')
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        crop_index = row.index(0) if 0 in row else len(row) # token 0 is the <STOP> token
        print(train_dataset.decode(row[:crop_index]))
    print('-'*40)
    model.train()

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

class InfiniteDataLoader:
    '''
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    '''

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
"""))

cells.append(code(r"""
max_steps = 2000
batch_size = 32
learning_rate = 5e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.99), eps=1e-8)
batch_loader = InfiniteDataLoader(train_dataset, batch_size=batch_size, num_workers=0)

for step in range(max_steps):
    t0 = time.time()

    # get the next batch, ship to device, and unpack it to input and target
    batch = batch_loader.next()
    batch = [t.to(device) for t in batch]
    X, Y = batch

    # feed into the model
    logits, loss = model(X, Y)

    # calculate the gradient, update the weights
    model.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t1 = time.time()

    # logging
    if step % 200 == 0:
        print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

    # evaluate the model
    if step > 0 and step % 1000 == 0:
        train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
        test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
        print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")
        print_samples(model, num=5)

print('training done')
"""))

cells.append(md(r"""
## 17. Evaluation: train vs test loss

- **train loss** — how well the model fits names it has seen
- **test loss** — how well it predicts names it has *never* seen

If train loss is much lower than test loss, the model is memorizing instead of generalizing.
"""))

# ---------------- Section 18 generation
cells.append(md(r"""
## 18. Generation

Generation connects training to actual use:

1. start with `<START>`
2. sample \(P(x_1 \mid \langle\text{START}\rangle)\)
3. append it, then sample \(P(x_2 \mid \langle\text{START}\rangle, x_1)\)
4. repeat:

\[
\boxed{x_t \sim P_\theta(x_t \mid x_{<t})}
\]

The feedback loop is literally `idx = torch.cat((idx, idx_next), dim=1)` — the newly generated token becomes part of the next input.

```text
training  = all known tokens processed in parallel (subject to causal mask)
generation = one token at a time
```
"""))
cells.append(code(r"""
model.eval()
print_samples(model, num=20)
"""))

# ---------------- Section 19 temperature/top-k
cells.append(md(r"""
## 19. Temperature and top-k

Temperature scales the logits before softmax:

\[
\boxed{P_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}}
\]

- lower `T` → more deterministic
- higher `T` → more random

**Top-k** keeps only the `k` highest-logit candidates, then samples from those.
"""))

# ---------------- Section 20 final picture
cells.append(md(r"""
## 20. Final end-to-end picture

```text
                    names.txt
                        │
                        ▼
                 Character IDs
                        │
                        ▼
                    X and Y
                        │
                        ▼
              Token Embeddings
                        │
                        +
                        │
             Position Embeddings
                        │
                        ▼
              ┌─────────────────┐
              │ Transformer     │
              │ Block           │
              │                 │
              │ LayerNorm       │
              │ Causal MHA      │
              │ Residual        │
              │ LayerNorm       │
              │ MLP + GELU      │
              │ Residual        │
              └─────────────────┘
                        │
                     × N layers
                        │
                        ▼
                  Final LayerNorm
                        │
                        ▼
                    LM Head
                        │
                        ▼
                     Logits
                        │
                        ▼
                 Cross Entropy
                        │
                        ▼
                   Backprop
                        │
                        ▼
                     AdamW
                        │
                        ▼
                  Updated model
                        │
                        ▼
                    Generate
                        │
                        ▼
                Next character
                        │
                        └──────→ repeat
```

### Math summary

\[
\boxed{X=E_{\text{token}}+E_{\text{position}}}
\]

\[
\boxed{Q=XW_Q,\quad K=XW_K,\quad V=XW_V}
\]

\[
\boxed{A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)}
\]

\[
\boxed{\operatorname{Attention}(X)=AV}
\]

\[
\boxed{x'=x+\operatorname{MHA}(\operatorname{LN}(x))}
\]

\[
\boxed{x''=x'+\operatorname{MLP}(\operatorname{LN}(x'))}
\]

\[
\boxed{\operatorname{MLP}(x)=W_2\operatorname{GELU}(W_1x+b_1)+b_2}
\]

\[
\boxed{Z=HW_{\text{lm}}}
\]

\[
\boxed{P(y=i)=\frac{e^{z_i}}{\sum_j e^{z_j}}}
\]

\[
\boxed{L=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})}
\]

\[
\boxed{\theta^*=\arg\min_\theta L(\theta)}
\]

\[
\boxed{x_t \sim P_\theta(x_t \mid x_{<t})}
\]
"""))

for c in cells:
    if c['cell_type'] == 'markdown':
        c.pop('id')
    else:
        import uuid
        c['id'] = uuid.uuid4().hex[:8]

nb = {
  "cells": cells,
  "metadata": {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "accelerator": "GPU",
    "language_info": {"name": "python"}
  },
  "nbformat": 4,
  "nbformat_minor": 5,
}

path = r'D:\AI-ML-stuff\LLM\neural-network-projects\karpathy\makemore_transformer.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('written,', len(cells), 'cells, all code compiles')
