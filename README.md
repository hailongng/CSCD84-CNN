# Food Image Classification with a CNN Trained from Scratch

A convolutional neural network built and trained from scratch in MATLAB to classify images across 9 visually similar food categories. The goal was not to maximise accuracy by any means available, but to understand *what actually moves the needle* when you cannot rely on pretrained weights.

Final validation accuracy: **63%**, up from a 40% baseline.

---

## Why this project is interesting

Transfer learning would have solved this task faster and better. The constraint here was to train from scratch, which turns the project into a study of architectural decisions rather than a study of fine-tuning.

The most useful outcome was not the accuracy number. It was identifying that accuracy had plateaued because of **dataset size**, not model capacity — and being able to show that with evidence rather than assert it.

---

## Dataset

<!-- TODO: fill in -->
- **Source:** [dataset name / link]
- **Classes (9):** [list them — e.g. burger, greek_salad, caprese_salad, ...]
- **Split:** [n] training / [n] validation images
- **Preprocessing:** resized to [WxH], augmented with [random reflection / translation / rotation]

Several of the classes are genuinely ambiguous - greek salad and caprese salad share most of their visual features, and the confusion matrix reflects that.

---

## Architecture

The final network:

```
input → [conv 7x7 → BN → ReLU → maxpool]
      → [conv 5x5 → BN → ReLU → maxpool]
      → [conv 3x3 → BN → ReLU → maxpool]
      → [conv 1x1 → BN → ReLU]
      → dropout
      → fullyConnected(9) → softmax
```

<!-- TODO: correct this to match your final layer list, including channel counts -->

Training configuration:

| Setting | Value |
|---|---|
| Optimiser | [adam / sgdm] |
| Initial learning rate | 1e-4 |
| L2 regularisation | [value] |
| Mini-batch size | [value] |
| Max epochs | [value] |

---

## Experiments

Each change was tested in isolation against the previous best, so the effect of any single modification is attributable.

| # | Change | Val. accuracy | Takeaway |
|---|---|---|---|
| 0 | Baseline (shallow, no normalisation) | 40% | Underfitting; too little capacity |
| 1 | Added conv blocks | [ ]% | [ ] |
| 2 | Batch normalisation | [ ]% | Stabilised training, faster convergence |
| 3 | Dropout | [ ]% | [ ] |
| 4 | L2 regularisation | [ ]% | [ ] |
| 5 | Data augmentation | [ ]% | [ ] |
| 6 | Decreasing filter sizes 7→5→3→1 | **63%** | Best result |
| 7 | Global average pooling | 60% | **Hurt.** See below |
| 8 | Learning rate 1e-4 → 1e-3 | [ ]% | [ ] |

<!-- TODO: fill in the accuracy column from your experiment log -->

### The global average pooling result

GAP was expected to help. Replacing the fully connected layer's ~200k inputs with 256 averaged feature maps cuts parameters dramatically and usually improves generalization - it is standard in ResNet and GoogLeNet.

It made things worse, dropping accuracy to 60%.

The likely reason is that GAP discards spatial information entirely, and a network this shallow, trained on this little data, has not yet learned features abstract enough to survive that. In deeper pretrained networks the final feature maps are semantically rich enough that their spatial arrangement is redundant. Here they were not.

Negative results are kept in this table deliberately.

---

## The accuracy ceiling

Across depth changes, normalization, three kinds of regularization, augmentation, and filter size tuning, validation accuracy consistently landed between 60% and 63%. Different architectural directions converged on the same number.

That pattern points at the data rather than the model. The evidence:

- **Confusion matrix:** errors concentrate in visually overlapping class pairs rather than spreading uniformly
- **t-SNE of learned features:** [describe what the embedding showed — overlapping clusters for the ambiguous classes?]
- **Convergence behaviour:** structurally different models reached the same plateau

<!-- TODO: add the figures -->
![Confusion matrix](figures/confusion_matrix.png)
![t-SNE embedding](figures/tsne.png)

The practical conclusion is that further architectural work would have had poor returns compared to acquiring more data, or using pretrained weights.

---

## Repository structure

```
├── src/
│   ├── train.m              # training script
│   ├── buildNetwork.m       # layer definitions
│   └── evaluate.m           # accuracy, confusion matrix, t-SNE
├── figures/
├── experiments/             # per-experiment logs
└── README.md
```

<!-- TODO: match this to your actual files -->

## Running it

```matlab
% Requires MATLAB R2024a+ with Deep Learning Toolbox
rng(42)
run('src/train.m')
```

Results vary by a few percentage points between seeds. `rng(42)` reproduces the reported run.

---

## What I would do differently

- Start with a fixed seed and a proper experiment log from experiment 0, not experiment 4
- Hold out a test set separately from the validation set used for model selection
- Benchmark against a pretrained SqueezeNet or ResNet-18 to quantify the transfer learning gap directly
