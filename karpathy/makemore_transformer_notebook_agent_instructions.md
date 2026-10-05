# Agent Instructions — Build an Easy-to-Understand `makemore` Transformer Jupyter Notebook

## 0. Goal

Create a Jupyter notebook that teaches me **Karpathy's `makemore.py` Transformer implementation from the ground up**.

The notebook must keep the **same core Transformer code and mathematical behavior as Karpathy's `makemore.py`**, but reorganize it into a clean, educational flow so I can run each section and understand exactly how the pieces connect.

### Important scope restriction

**ONLY implement and explain the Transformer model.**

Do NOT include:

- Bigram
- MLP language model
- RNN
- GRU
- Bag-of-Words

Those are present in the original `makemore.py`, but they are not part of this notebook.

The notebook should focus exclusively on the code path:

```text
dataset
→ character encoding
→ X/Y training pairs
→ token embedding
→ positional embedding
→ causal self-attention
→ multi-head attention
→ Transformer block
→ stacked Transformer blocks
→ final LayerNorm
→ LM head
→ logits
→ cross-entropy loss
→ backpropagation
→ AdamW
→ autoregressive generation
```

---

# 1. Source of truth

Use Karpathy's `makemore.py` Transformer implementation as the primary source.

The relevant classes/functions are:

```text
ModelConfig
NewGELU
CausalSelfAttention
Block
Transformer
generate
print_samples
evaluate
CharDataset
create_datasets
InfiniteDataLoader
training loop
```

Preserve the original mathematical behavior and implementation wherever possible.

Do **not** replace the implementation with:

```python
torch.nn.Transformer
torch.nn.MultiheadAttention
HuggingFace Transformers
```

The educational point is to understand how the Transformer is implemented manually.

In particular, keep the explicit implementation of:

```python
Q, K, V
attention scores
causal mask
softmax
attention @ V
multi-head reshaping
output projection
```

The original code deliberately implements attention explicitly for learning. Preserve that.

---

# 2. Notebook philosophy

This is a **learning notebook**, not merely a runnable version of the script.

Every major concept should follow this pattern:

```text
Concept
↓
Small explanation
↓
Mathematical formula
↓
Code
↓
Tensor shapes / what happened
↓
Tiny sanity check when useful
```

Do not dump huge explanations before the code.

Keep explanations:

- clear
- short
- technically correct
- beginner-friendly
- directly connected to the code immediately below/above them

Avoid excessive theory that is not needed to understand this implementation.

---

# 3. Overall notebook structure

Use the following section order.

---

## Section 1 — What are we building?

Start with a short explanation.

Explain:

> We are training a small character-level, GPT-style, decoder-only Transformer that learns to generate names similar to the names in `names.txt`.

Show the high-level architecture:

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

Clearly state:

### What is the task?

The model learns:

\[
P(x_t \mid x_1,x_2,\ldots,x_{t-1})
\]

That is:

> Given the previous characters, predict the next character.

Explain that this is a **character-level language model**.

Also explicitly state:

> This is a decoder-only Transformer because its self-attention is causal: each position can only attend to itself and positions before it.

---

# 4. Section 2 — Imports and configuration

Show the required imports.

Then define the same `ModelConfig` used by the Transformer.

Explain these parameters:

```python
block_size
vocab_size
n_layer
n_embd
n_head
```

Use a small example configuration such as the original defaults:

```text
n_layer = 4
n_embd  = 64
n_head  = 4
```

Explain:

\[
d_{\text{head}} = \frac{n_{\text{embd}}}{n_{\text{head}}}
\]

For the defaults:

\[
d_{\text{head}}=\frac{64}{4}=16
\]

Make clear that:

- `vocab_size` = number of possible character tokens including the special token
- `block_size` = maximum sequence length
- `n_embd` = hidden/embedding dimension
- `n_head` = number of attention heads
- `n_layer` = number of Transformer blocks

---

# 5. Section 3 — Dataset and character-level tokenization

Load `names.txt`.

If the notebook needs to download the file, make this simple and clearly explain where it comes from.

Show an example:

```text
emma
olivia
ava
liam
```

Then build:

```python
stoi
itos
```

Explain:

```text
stoi = string → integer
itos = integer → string
```

For example:

```text
a → 1
b → 2
c → 3
...
0 → special START/STOP token
```

Do not overcomplicate tokenization.

Make clear that this is **character-level**, not BPE/subword tokenization.

---

# 6. Section 4 — How X and Y are created

This section is extremely important.

Use one concrete example, such as:

```text
cat
```

Show:

```text
X:
<START> c a t

Y:
c       a t <STOP>
```

Explain that the target is shifted by one position.

Mathematically:

\[
X_t=x_{t-1}
\]

and the model learns:

\[
P(x_t\mid x_{<t})
\]

For a sequence:

\[
x_1,x_2,\ldots,x_T
\]

the autoregressive factorization is:

\[
\boxed{
P(x_1,x_2,\ldots,x_T)
=
\prod_{t=1}^{T}
P(x_t\mid x_1,\ldots,x_{t-1})
}
\]

Explain that this is the central training objective.

Show the relevant `CharDataset.__getitem__` code from Karpathy's implementation.

Also explain why padded positions use:

```python
-1
```

and why the loss uses:

```python
ignore_index=-1
```

---

# 7. Section 5 — Train/test split

Explain briefly:

```text
training set → used to update weights
test set     → used to check generalization
```

Keep the original split logic from `makemore.py`.

Explain why we do not train on the test set.

---

# 8. Section 6 — Token embeddings

Introduce:

```python
nn.Embedding(vocab_size, n_embd)
```

Explain that token IDs are not meaningful numerical quantities.

The embedding table is:

\[
E\in\mathbb{R}^{V\times d}
\]

where:

- \(V\) = vocabulary size
- \(d\) = embedding dimension

For token \(i\):

\[
e_i=E[i]
\]

Therefore:

```text
token ID
   ↓
embedding lookup
   ↓
d-dimensional vector
```

Show the tensor shape:

```text
idx:      (B, T)
token embedding: (B, T, C)
```

where:

- \(B\) = batch size
- \(T\) = sequence length
- \(C\) = embedding dimension

---

# 9. Section 7 — Positional embeddings

Explain why the Transformer needs positional information.

Attention alone does not inherently encode sequence order.

The model uses:

```python
wpe = nn.Embedding(block_size, n_embd)
```

For position \(i\):

\[
p_i=P[i]
\]

Then:

\[
\boxed{
x_i=e_i+p_i
}
\]

or in matrix form:

\[
\boxed{
X=E_{\text{token}}+E_{\text{position}}
}
\]

Show the code:

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = tok_emb + pos_emb
```

Explain the resulting shape:

```text
(B, T, C)
```

---

# 10. Section 8 — Causal self-attention

This is the most important section of the notebook.

Start with the intuitive question:

> For the current character, which previous characters should the model pay attention to?

Then introduce Query, Key, Value.

For input matrix \(X\):

\[
\boxed{
Q=XW_Q
}
\]

\[
\boxed{
K=XW_K
}
\]

\[
\boxed{
V=XW_V
}
\]

Explain briefly:

- Query = what this position is looking for
- Key = what each position offers for matching
- Value = information that gets passed along

Show that Karpathy computes all three with one linear layer:

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
```

then splits it into:

```python
q, k, v
```

---

# 11. Attention score formula

Derive the attention score:

\[
\boxed{
S=\frac{QK^T}{\sqrt{d_k}}
}
\]

Explain why the transpose is needed.

Explain the shapes.

For one head:

```text
Q: (B, T, d_head)
K: (B, T, d_head)
Kᵀ: (B, d_head, T)

QKᵀ:
(B, T, d_head)
×
(B, d_head, T)
=
(B, T, T)
```

Interpret:

> Every token gets a score against every other token.

---

# 12. Why divide by sqrt(d_k)?

Give a short explanation.

The dot products can grow in magnitude as \(d_k\) increases.

Scaling by:

\[
\frac{1}{\sqrt{d_k}}
\]

keeps the attention logits at a more manageable scale before softmax.

Do not give a long statistical derivation.

---

# 13. Causal mask

This must be explained carefully.

Show:

\[
M=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}
\]

Explain:

```text
position 1 → can see position 1
position 2 → can see positions 1,2
position 3 → can see positions 1,2,3
position 4 → can see positions 1,2,3,4
```

The future must be hidden.

Show the code:

```python
torch.tril(...)
```

and:

```python
att = att.masked_fill(mask == 0, float('-inf'))
```

Explain why `-inf` is used.

Softmax:

\[
\operatorname{softmax}(z_i)
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Since:

\[
e^{-\infty}=0
\]

masked positions receive probability:

\[
0
\]

Explicitly connect this to decoder-only Transformers:

> The causal mask is what makes this an autoregressive decoder-style Transformer rather than bidirectional encoder-style attention.

---

# 14. Softmax attention

After masking:

\[
\boxed{
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
+M
\right)
}
\]

Then:

\[
\boxed{
Y=AV
}
\]

Therefore:

\[
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
+M
\right)V
}
\]

Explain that \(A\) contains attention weights and \(Y\) is the weighted combination of values.

---

# 15. Multi-head attention

Explain why there are multiple heads.

For:

```text
n_embd = 64
n_head = 4
```

we get:

\[
d_{\text{head}}=\frac{64}{4}=16
\]

Each head performs its own attention.

For head \(i\):

\[
Q_i=XW_Q^i
\]

\[
K_i=XW_K^i
\]

\[
V_i=XW_V^i
\]

\[
H_i=
\operatorname{softmax}
\left(
\frac{Q_iK_i^T}{\sqrt{d_k}}+M
\right)V_i
\]

Then concatenate:

\[
\boxed{
H=\operatorname{Concat}(H_1,\ldots,H_h)
}
\]

and apply output projection:

\[
\boxed{
Y=HW_O
}
\]

Show the corresponding reshaping code from `CausalSelfAttention`.

Make the tensor dimensions visible in comments.

---

# 16. Section 9 — GELU

Introduce the exact GELU approximation used by Karpathy:

\[
\boxed{
\operatorname{GELU}(x)
=
\frac{1}{2}x
\left[
1+
\tanh
\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right]
}
\]

Show the `NewGELU` class exactly or nearly exactly as in the source.

Explain only:

> GELU is the nonlinear activation used inside the Transformer feed-forward network.

Do not spend too much time on GELU theory.

---

# 17. Section 10 — Transformer block

Now explain how attention and MLP are combined.

The original code is:

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlpf(self.ln_2(x))
```

Explain that this is **pre-LayerNorm + residual connections**.

Mathematically:

\[
\boxed{
x'=
x+
\operatorname{MHA}(\operatorname{LN}(x))
}
\]

then:

\[
\boxed{
x''=
x'+
\operatorname{MLP}(\operatorname{LN}(x'))
}
\]

The MLP is:

\[
\boxed{
\operatorname{MLP}(x)
=
W_2\operatorname{GELU}(W_1x+b_1)+b_2
}
\]

For the default configuration:

```text
64 → 256 → 64
```

because:

\[
4d=4(64)=256
\]

Explain the roles:

```text
Attention → communication between token positions

MLP       → transforms each token representation independently

Residual  → preserves/adds the previous representation

LayerNorm → normalizes the representation before sublayers
```

This distinction is important.

---

# 18. Section 11 — LayerNorm

Give the mathematical formula, but keep the explanation short.

For a vector \(x\in\mathbb{R}^d\):

\[
\mu=
\frac{1}{d}
\sum_{i=1}^{d}x_i
\]

\[
\sigma^2=
\frac{1}{d}
\sum_{i=1}^{d}(x_i-\mu)^2
\]

Then:

\[
\boxed{
\operatorname{LN}(x)
=
\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
}
\]

Explain:

- \(\gamma\) is learned
- \(\beta\) is learned
- normalization happens across the feature dimension for each token

---

# 19. Section 12 — Full Transformer

Now combine everything.

The model should visually be:

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

Show the actual `Transformer` class from Karpathy's code.

Do not hide the important lines.

Especially show:

```python
wte
wpe
h
ln_f
lm_head
```

Explain what each one means.

---

# 20. Section 13 — LM head and logits

Explain that the Transformer output is still a hidden representation.

The LM head converts it to vocabulary-sized logits.

If:

\[
H\in\mathbb{R}^{B\times T\times d}
\]

then:

\[
Z=HW_{\text{lm}}
\]

where:

\[
W_{\text{lm}}\in\mathbb{R}^{d\times V}
\]

and therefore:

\[
Z\in\mathbb{R}^{B\times T\times V}
\]

Explain:

> Each position now has one logit for every possible character.

---

# 21. Section 14 — Cross-entropy loss

Explain that logits are converted conceptually to probabilities using softmax:

\[
\boxed{
P(y=i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
}
\]

For the correct target \(y\):

\[
\boxed{
L=-\log P(y)
}
\]

For the whole sequence:

\[
\boxed{
L(\theta)
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log
P_\theta(x_t\mid x_{<t})
}
\]

Explain:

- high probability for the correct character → low loss
- low probability for the correct character → high loss

Show the actual:

```python
F.cross_entropy(...)
```

line.

---

# 22. Section 15 — Backpropagation and AdamW

Explain the training update simply.

The model has parameters:

\[
\theta
\]

Training wants:

\[
\boxed{
\theta^*
=
\arg\min_\theta L(\theta)
}
\]

Backpropagation computes:

\[
\boxed{
\frac{\partial L}{\partial\theta}
}
\]

Then AdamW updates the parameters.

You may show the conceptual Adam equations:

\[
g_t=\nabla_\theta L_t
\]

\[
m_t=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
\]

\[
v_t=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
\]

with bias correction:

\[
\hat m_t=
\frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t=
\frac{v_t}{1-\beta_2^t}
\]

and the Adam-style update:

\[
\theta_t
=
\theta_{t-1}
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

Do not implement Adam manually. Use:

```python
torch.optim.AdamW
```

The equations are only there to explain what the optimizer is doing conceptually.

---

# 23. Section 16 — Training loop

Now show the actual training loop.

Make the flow very explicit:

```text
get batch
   ↓
X, Y
   ↓
model(X, Y)
   ↓
logits, loss
   ↓
zero gradients
   ↓
loss.backward()
   ↓
optimizer.step()
   ↓
repeat
```

Show the relevant code from Karpathy.

Explain only the important lines.

Also explain:

```python
model.train()
model.eval()
```

and why evaluation is periodically performed.

---

# 24. Section 17 — Evaluation

Explain:

```text
train loss
test loss
```

in simple terms.

The test set is used to check whether the model learned patterns that generalize beyond the exact training examples.

Keep the original `evaluate()` logic.

---

# 25. Section 18 — Generation

Explain generation very carefully because this connects training to actual use.

Start:

```text
<START>
```

Then:

\[
P(x_1\mid <START>)
\]

sample a character.

Now:

\[
P(x_2\mid <START>,x_1)
\]

sample again.

Continue:

\[
\boxed{
x_t\sim P_\theta(x_t\mid x_{<t})
}
\]

The newly generated token becomes part of the next input.

Show the loop from `generate()`.

Explain:

```python
idx = torch.cat((idx, idx_next), dim=1)
```

This is the autoregressive feedback loop.

---

# 26. Section 19 — Temperature and top-k

Explain briefly.

Temperature:

\[
\boxed{
P_i=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}
}
\]

Lower \(T\):

```text
more deterministic
```

Higher \(T\):

```text
more random
```

Top-k:

```text
keep only the k highest-logit candidates
then sample from those
```

Do not over-explain.

---

# 27. Section 20 — Final end-to-end picture

End the notebook with one complete diagram:

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

Then include one compact "math summary":

\[
\boxed{
X=E_{\text{token}}+E_{\text{position}}
}
\]

\[
\boxed{
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
}
\]

\[
\boxed{
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}+M
\right)
}
\]

\[
\boxed{
\operatorname{Attention}(X)=AV
}
\]

\[
\boxed{
x'=x+\operatorname{MHA}(\operatorname{LN}(x))
}
\]

\[
\boxed{
x''=x'+\operatorname{MLP}(\operatorname{LN}(x'))
}
\]

\[
\boxed{
\operatorname{MLP}(x)
=
W_2\operatorname{GELU}(W_1x+b_1)+b_2
}
\]

\[
\boxed{
Z=HW_{\text{lm}}
}
\]

\[
\boxed{
P(y=i)=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
}
\]

\[
\boxed{
L=
-\frac1T
\sum_{t=1}^{T}
\log P_\theta(x_t|x_{<t})
}
\]

\[
\boxed{
\theta^*=\arg\min_\theta L(\theta)
}
\]

\[
\boxed{
x_t\sim P_\theta(x_t|x_{<t})
}
\]

---

# 28. Code-preservation requirements

The notebook should remain recognizably Karpathy's `makemore.py`.

Do not unnecessarily rewrite the implementation into a completely different style.

Keep these classes recognizable:

```python
NewGELU
CausalSelfAttention
Block
Transformer
CharDataset
InfiniteDataLoader
```

Keep the important method names:

```python
forward()
get_block_size()
generate()
evaluate()
```

You may split large classes across multiple notebook cells if that improves readability.

You may add:

- comments
- shape-printing cells
- tiny demonstrations
- assertions
- visualizations of the causal mask
- small tensor examples

But do not change the underlying Transformer mathematics just to make the notebook look different.

---

# 29. Tensor-shape annotations are REQUIRED

The notebook should repeatedly show shapes.

Use notation:

```text
B = batch size
T = sequence length
C = embedding dimension
H = number of attention heads
D = C / H = head dimension
V = vocabulary size
```

For example:

```text
idx                  : (B, T)
token embeddings     : (B, T, C)
Q/K/V before split   : (B, T, C)
Q/K/V after split    : (B, H, T, D)
attention scores     : (B, H, T, T)
attention output     : (B, H, T, D)
merged output        : (B, T, C)
logits               : (B, T, V)
```

These shape explanations are important for understanding the code.

---

# 30. Add small sanity-check cells

Where useful, add small runnable cells such as:

### Causal mask

Print a small:

```text
4 × 4
```

causal mask.

### Attention shape

Run a tiny random tensor through attention and print:

```text
input shape
Q shape
K shape
V shape
attention score shape
output shape
```

### Dataset example

Print one word and its:

```text
X
Y
decoded X
decoded Y
```

### Model output

Show:

```text
logits.shape
loss.item()
```

The purpose is to let me **see the mathematics happening in tensors**.

---

# 31. Do not make the notebook too abstract

Avoid creating excessive helper functions that hide what is happening.

For example, do not turn:

```python
Q = ...
K = ...
V = ...
att = ...
```

into one mysterious function where I cannot see the intermediate values.

This notebook is for learning.

I should be able to read:

```text
code
↓
formula
↓
tensor shape
```

and understand the connection.

---

# 32. Do not over-explain

The notebook should NOT become a textbook.

For each concept, aim for:

```text
2–5 paragraphs maximum
+
formula(s)
+
code
+
shape explanation
```

Use longer explanation only for:

- autoregressive training
- Q/K/V
- causal masking
- multi-head attention
- Transformer block
- training vs generation

Those are the concepts that need more explanation.

---

# 33. Important conceptual distinctions to explicitly teach

The notebook MUST make these distinctions clear:

### Token vs embedding

```text
token ID = integer
embedding = learned vector
```

### Logits vs probabilities

```text
logits = raw model scores
probabilities = softmax(logits)
```

### Attention vs MLP

```text
attention = mixes information across positions
MLP = transforms each position's representation
```

### Training vs generation

```text
training = all known tokens can be processed in parallel, subject to causal mask
generation = one token is generated at a time
```

### Model parameters vs activations

```text
parameters = learned weights
activations = intermediate values produced for a particular input
```

### Decoder-only vs encoder-decoder

Explain briefly:

```text
decoder-only:
tokens → causal self-attention → next token
```

There is no encoder or cross-attention in this implementation.

---

# 34. Final notebook quality requirements

The finished notebook must:

1. Run from top to bottom.
2. Use only the Transformer model.
3. Keep Karpathy's core Transformer implementation.
4. Have a logical teaching progression.
5. Explain each major code component immediately around the code.
6. Include the mathematical formulas in proper LaTeX.
7. Include tensor shapes.
8. Include the causal-mask visualization.
9. Include a small example of X/Y creation.
10. Train a small model.
11. Generate new names at the end.
12. Make clear that this is a character-level decoder-only Transformer.
13. Avoid unnecessary libraries.
14. Avoid hidden abstractions.
15. Prefer readability over cleverness.

---

# 35. Final learning objective

By the end of the notebook, I should be able to look at Karpathy's original `makemore.py` and understand this complete chain:

```text
names
  ↓
characters
  ↓
token IDs
  ↓
X/Y next-token pairs
  ↓
token embedding
  ↓
position embedding
  ↓
Q/K/V
  ↓
scaled dot-product attention
  ↓
causal mask
  ↓
softmax
  ↓
weighted values
  ↓
multi-head attention
  ↓
output projection
  ↓
residual + LayerNorm
  ↓
MLP + GELU
  ↓
residual + LayerNorm
  ↓
repeat Transformer blocks
  ↓
final LayerNorm
  ↓
LM head
  ↓
logits
  ↓
softmax
  ↓
next-character probabilities
  ↓
cross-entropy loss
  ↓
backpropagation
  ↓
AdamW
  ↓
trained parameters
  ↓
autoregressive sampling
  ↓
new name
```

The final notebook should make this chain feel like **one connected system**, rather than a collection of unrelated PyTorch classes.
