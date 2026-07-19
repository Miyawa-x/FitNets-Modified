FitNets
=======

FitNets: Hints for Thin Deep Nets

http://arxiv.org/abs/1412.6550

- To run FitNets stage-wise training:
  THEANO_FLAGS="device=gpu,floatX=float32,optimizer_including=cudnn" python fitnets_training.py fitnet_yaml regressor -he hints_epochs -lrs lr_scale
  
  - fitnet_yaml: path to the FitNet yaml file,
  - regressor: regressor type, either convolutional (conv) or   fully-connected (fc),
  - Optional argument -he hints_epochs: int - number of epochs to train the 1st stage. It is set to None by default. Leave as None when using the validation set to determine the number of epochs. Set to X when using the whole training set.
  - Optional argument -lrs lr_scale: float - learning rate scaler to be applied to the pre-trained layers at the 2nd stage.

PyTorch projected-logit migration
---------------------------------

The original code is a legacy Python 2 / pylearn2 / Theano implementation.
For Python 3 and NVIDIA GPUs, use the PyTorch migration entry point:

First train a CIFAR-100 teacher:

```
python train_teacher_torch.py \
  --dataset cifar100 \
  --download \
  --whiten \
  --output checkpoints/cifar100_teacher.pt \
  --epochs 288 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --amp
```

Teacher checkpoints include model, optimizer, scheduler, scaler, epoch, and
preprocessing metadata. Resume an interrupted run with
`--resume checkpoints/cifar100_teacher_last.pt`; `--epochs` remains the final
target epoch rather than an additional epoch count.

Then run projected-logit FitNets:

```
python train_projected_logits_torch.py \
  --dataset cifar100 \
  --download \
  --teacher-ckpt path/to/pretrained_teacher.pt \
  --output-dir runs/cifar100_projected_fitnets \
  --stage0-epochs 20 \
  --stage1-epochs 40 \
  --stage2-epochs 288 \
  --device cuda \
  --amp
```

The PyTorch flow implements the projected-logit design:

- The CIFAR student backbone mirrors the original 19-layer Maxout FitNet
  student in `yaml/cifar100_fitnet19_all.yaml`: Maxout convolutional layers,
  `h4/h10/h16` pooling, a 500-unit Maxout fully connected layer, and a final
  classifier.
- The original repo does not include the teacher YAML, only a
  `<path_teacher_pkl>` placeholder. The FitNets paper distills the Goodfellow
  Maxout-CNN teacher, so `maxout_cifar_teacher` reproduces that architecture:
  three maxout convolutional layers (96-192-192, 2 pieces, 8x8/8x8/5x5 kernels,
  4x4/4x4/2x2 pooling, max-kernel-norm 0.9/1.9365/1.9365), a 500-unit / 5-piece
  maxout fully connected layer, and a softmax. The hint index defaults to teacher
  layer `1` (the middle conv), matching the original `hints: [[10, 1]]`. The
  teacher trains with RMSprop (base lr 0.005, per-conv `W_lr_scale=0.05`, alpha
  0.9, eps 1e-5), gradient clipping, max-norm constraints, and no L2 weight
  decay. The effective convolution learning rate is 0.00025, preventing the
  unnormalized Maxout stack from saturating its max-norm bounds on the first
  updates.
- Stage 0 freezes the teacher backbone and trains `teacher_proj`, a bias-free
  `1x1` projection plus global average pooling from teacher middle features to
  class logits, using true-label CE.
- Stage 1 freezes `teacher + teacher_proj`, trains the student front and
  `student_proj`, and minimizes middle-logit KL plus true-label CE.
- Stage 2 discards `student_proj` and trains the full student with final CE +
  KD from the teacher's final logits.

`--teacher-ckpt` is required for real projected-logit distillation. To sanity
check the training loop without data downloads or a teacher checkpoint, run:

```
python train_projected_logits_torch.py \
  --dataset fake-cifar100 \
  --allow-random-teacher \
  --stage0-epochs 1 \
  --stage1-epochs 1 \
  --stage2-epochs 1
```

Useful Stage 1 controls:

- `--teacher-mid-index` and `--student-mid-index` choose the middle features.
  Defaults match the bundled FitNet-style PyTorch models.
- `--stage1-temperature` controls middle-logit KL temperature.
- `--stage1-ce-weight` keeps the student projection aligned with true labels.
- `--stage1-kd-weight` controls the teacher projected-logit KL weight.

FitNets baseline (for comparison)
---------------------------------

`train_fitnets_baseline_torch.py` implements the *original* FitNets recipe so it
can be compared head-to-head with the projected-logit variant:

- Stage 1 (hints): a convolutional regressor maps the student guided feature map
  onto the teacher hint feature map, trained with the FitNets MSE objective
  (`torch_fitnets/regressors.py`, `losses.hint_mse_loss`).
- Stage 2 (KD): the full student is trained with CE + KD from the teacher's final
  logits (identical to the projected variant's final stage).

```
python train_fitnets_baseline_torch.py \
  --dataset cifar100 --download --whiten \
  --teacher-ckpt checkpoints/cifar100_teacher.pt \
  --output-dir runs/cifar100_fitnets_baseline \
  --hint-epochs 40 --kd-epochs 288 --device cuda \
  --grad-clip 0
```

The baseline defaults reproduce the source implementation where practical: a
Maxout convolutional hint regressor, the C01B `HintCost` reduction, soft-target
cross entropy without `T^2`, teacher weight decay from 4 toward 1, and the YAML
tail initialization. Such runs are tagged `legacy_source_compatible` in
`run_config.json`; the unavailable original teacher checkpoint still prevents
claiming bit-for-bit reproduction.

For a controlled comparison that isolates only the Stage 1 guidance method,
run both baselines with the same stabilized Stage 2. The Original-controlled
side uses:

```
python train_fitnets_baseline_torch.py \
  --dataset cifar100 --download --whiten \
  --teacher-ckpt checkpoints/cifar100_teacher.pt \
  --output-dir runs/cifar100_fitnets_controlled \
  --hint-epochs 40 --kd-epochs 288 \
  --stage2-tail-init kaiming \
  --kd-loss-mode modern --kd-weight-schedule fixed \
  --device cuda
```

Relation FitNets (heterogeneous features)
-----------------------------------------

`train_relation_fitnets_torch.py` removes the learned hint regressor entirely.
Stage 1 flattens each model's native middle feature and matches normalized
pairwise distances plus centered cosine-similarity matrices across the batch.
Teacher and student feature dimensions may differ because both relation
matrices are `batch_size x batch_size`. Stage 2 is the same final CE + KD
training used by the baseline. A small parameter-free log-RMS feature-energy
loss anchors the absolute student feature scale that normalized relational
losses intentionally discard. Stage 1 feature extraction and relation losses
run in FP32 even when `--amp` is enabled; AMP remains active for Stage 2.
The optional `--stage2-tail-init kaiming` initializes only the still-untrained
tail at the stage boundary, preventing six tiny-initialized Maxout layers from
attenuating the guided feature. The same option is available in the original
FitNets baseline for controlled comparisons.

```
python train_relation_fitnets_torch.py \
  --dataset cifar100 --download --whiten \
  --teacher-ckpt checkpoints/cifar100_teacher.pt \
  --output-dir runs/cifar100_relation_fitnets \
  --relation-epochs 40 --kd-epochs 288 \
  --distance-weight 1 --similarity-weight 1 --energy-weight 0.1 \
  --stage2-tail-init kaiming \
  --device cuda --amp
```

GCN+ZCA whitening
-----------------

Pass `--whiten` to any of the three scripts to reproduce the original
FitNets/Maxout preprocessing (per-image global contrast normalization with
`scale=55`, then dataset ZCA whitening with `filter_bias=0.1`). It is computed
once per launch on CIFAR-10/CIFAR-100. The default is standard per-channel
mean/std normalization; for an internal A/B comparison either works as long as
the teacher and both students use the same setting.

One-command comparison (disconnect-safe)
----------------------------------------

`scripts/run_comparison.sh` trains one teacher and both students (FitNets
baseline + projected logits) end to end, each stage tee'd to its own log:

```
nohup bash scripts/run_comparison.sh > runs/compare.out 2>&1 &
tail -f runs/compare_cifar100/logs/*.log
```

It is configurable via environment variables (`DATASET`, `WHITEN`, `AMP`,
`TEACHER_EPOCHS`, `HINT_EPOCHS`, `KD_EPOCHS`, `TEACHER_CKPT`, ...) and skips
teacher training if the checkpoint already exists, so re-runs only redo the
students.

Install PyTorch and torchvision with the CUDA build that matches your NVIDIA
driver from https://pytorch.org/get-started/locally/.
