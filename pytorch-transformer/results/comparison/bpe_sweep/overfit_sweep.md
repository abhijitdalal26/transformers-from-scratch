# Generalisation gap per checkpoint (teacher-forced loss)

- train subset = the first 200 sentences of the seeded train split (fixed, not reshuffled), val = the first 200 sentences of the seeded val split
- loss = cross-entropy in nats per token / per word, no label smoothing, padding excluded

| variant | epoch | train nats/token | val nats/token | **gap** | train nats/word | val nats/word | **gap** |
|---|---|---|---|---|---|---|---|
| bpe | 09 | 1.857 | 3.641 | **1.784** | 2.573 | 5.039 | **2.466** |
| bpe | 17 | 0.630 | 4.409 | **3.779** | 0.874 | 6.102 | **5.228** |
| bpe | 19 | 0.552 | 4.580 | **4.028** | 0.765 | 6.338 | **5.574** |
| wordlevel | 17 | 0.695 | 1.169 | **0.474** | 0.867 | 1.442 | **0.575** |
