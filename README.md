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
  --output checkpoints/cifar100_teacher.pt \
  --epochs 288 \
  --batch-size 128 \
  --num-workers 4 \
  --device cuda \
  --amp
```

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
  `<path_teacher_pkl>` placeholder. The bundled PyTorch teacher is therefore a
  wider Maxout FitNet-style teacher with the original hint index defaulting to
  teacher layer `1`. If you have the exact original teacher architecture, add it
  to `torch_fitnets/models.py` and pass it through `--teacher-arch`.
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

Install PyTorch and torchvision with the CUDA build that matches your NVIDIA
driver from https://pytorch.org/get-started/locally/.
