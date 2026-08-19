# Board-v7 normalize candidate

This branch tests one raw-model architecture change for scale robustness:

```text
HWC card pixels [N, 21, 30, 3]
        │ permute
NCHW [N, 3, 21, 30]
        │ shared 5×5 convolution, stride 1, 12 channels
        │ GELU
        │ adaptive average pool → [N, 12, 6, 8]
        ├─ four attribute specialists: 3×3 convolution, stride 1, 8 channels
        │  GELU → flatten → 24-wide projection → GELU → 3-class logits
        └─ existing relation scorer → raw triple action score
```

The incumbent used stride-2 convolutions in both the shared frontend and each
specialist. This candidate removes those two fixed decimation points and lets
adaptive pooling choose the reduction windows from the feature-map dimensions.
For the current 21×30 card raster, FrankenTorch uses the PyTorch-compatible
window rule `start=floor(i*input/output)` and `end=ceil((i+1)*input/output)`.
MPSGraph constructs the same windows explicitly from slices and means. The
specialist map and flattened feature width remain 8×6×8, so the raw action and
attribute contracts do not change.

The pool has no parameters; the candidate has 41,918 trainable parameters,
the same as the incumbent. It is an architectural hypothesis only: no guard,
abstention, symbolic solver, card metadata, or sealed board-v7 evaluation is
part of this candidate.

## Fixed-batch backend parity

Before any candidate training, both backends used initialization seed
`3935559000370003845`, data seed `99005`, one AdamW step, no augmentation, and
the same 64-triple batch. The MPSGraph post-step probe reported loss
`5.364172`; loading its checkpoint into FrankenTorch reported
`5.364171782`. The first eight shared-convolution gradients were equal within
`1.7e-7` absolute difference. This is the required forward/loss/gradient
parity gate; no 20,000-step run was started in this branch.

The focused FrankenTorch test `adaptive_frontend_emits_the_fixed_specialist_map`
checks the `[N,12,6,8]` pool output, `[N,12]` attribute-logit contract, and
parameter count. The existing board-path equivalence test also passes with the
new frontend.
