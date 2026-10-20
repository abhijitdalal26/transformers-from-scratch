# Source ablation - is the model using the English sentence?

Teacher-forced loss on the same 200 held-out sentences, changing only the encoder input.

| variant | epoch | real source | shuffled source | blank source | **source dependence** |
|---|---|---|---|---|---|
| bpe | 09 | 3.641 | 5.804 | 6.478 | **43.8%** |
| bpe | 10 | 3.600 | 5.825 | 6.405 | **43.8%** |
| bpe | 12 | 3.734 | 6.107 | 6.626 | **43.6%** |
| bpe | 17 | 4.409 | 7.147 | 7.609 | **42.0%** |
| bpe | 19 | 4.580 | 7.424 | 7.448 | **38.5%** |
| wordlevel | 17 | 1.169 | 5.398 | 6.218 | **81.2%** |

`source dependence` = (blank - real) / blank: how much of the model's held-out prediction the English sentence is actually worth.
