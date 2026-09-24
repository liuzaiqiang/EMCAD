# Disagreement-Guided Adaptive EMCAD reference integration

This directory is outside the repository. Nothing under `SLDGroup_EMCAD` was modified.

## 0. What the implementation actually does

Main module (innovation 1):

`U_i = normalized_entropy(q_i) + lambda * normalized_JS(q_i, up(q_{i+1}))`

`alpha_i = softmax(R(F_i, U_i) / temperature)`

`Y_i = K * sum_k alpha_i,k * D_k(F_i)`

The factor `K` is important: with equal weights, `K * sum(1/K * D_k)` is exactly the
original branch sum. The implementation calculates all branches, so do not claim FLOP
reduction. It supports `equal/global/feature/disagreement` routing ablations.

Training-only module (innovation 5):

- EMA model's final head is the teacher.
- Away from GT boundaries: confidence-masked teacher-to-student KL.
- In a GT boundary band: teacher/student JS plus probability-gradient matching.
- Only the first three coarse outputs are distilled; the student's final head remains the
  deployed output. The EMA model is deleted after training and adds no inference cost.

## 1. Files to add manually

Copy these reference files after review:

```text
C:\tmp\DG_EMCAD_reference\lib\dg_emcad.py
    -> <repo>\lib\dg_emcad.py

C:\tmp\DG_EMCAD_reference\utils\dg_losses.py
    -> <repo>\utils\dg_losses.py
```

Do not replace `lib/decoders.py` or `lib/networks.py`. The new classes inherit/reuse them.

## 2. `train_synapse.py`: imports

Keep the original import and add:

```python
from lib.networks import EMCADNet
from lib.dg_emcad import DGEMCADNet
```

## 3. `train_synapse.py`: command-line switches

Add immediately after the existing `--supervision` argument:

```python
parser.add_argument("--adaptive_msdc", action="store_true", default=False)
parser.add_argument("--bp_distill", action="store_true", default=False)
parser.add_argument(
    "--dg_router_mode",
    choices=["equal", "global", "feature", "disagreement"],
    default="disagreement",
)
parser.add_argument("--dg_disagreement_lambda", type=float, default=1.0)
parser.add_argument("--dg_router_temperature", type=float, default=1.0)
parser.add_argument("--dg_router_hidden", type=int, default=32)
parser.add_argument("--dg_route_aux_weight", type=float, default=0.20)
parser.add_argument("--dg_route_reg_weight", type=float, default=0.05)
parser.add_argument("--dg_distill_weight", type=float, default=1.0)
parser.add_argument("--dg_distill_warmup", type=int, default=10)
parser.add_argument("--dg_distill_ramp", type=int, default=20)
parser.add_argument("--dg_ema_decay", type=float, default=0.999)
parser.add_argument("--dg_temperature", type=float, default=2.0)
parser.add_argument("--dg_confidence", type=float, default=0.70)
parser.add_argument("--dg_boundary_radius", type=int, default=2)
```

## 4. `train_synapse.py`: unique experiment name

After building `args.exp`, append this before constructing `snapshot_path`:

```python
dg_tag = "_baseline"
if args.adaptive_msdc:
    dg_tag = "_amsdc_{}".format(args.dg_router_mode)
if args.bp_distill:
    dg_tag += "_bpd"
args.exp += dg_tag
```

Also append the same `dg_tag` to the inner snapshot directory name. This prevents a new
run from overwriting baseline `best.pth`/`last.pth`.

## 5. `train_synapse.py`: replace only model construction

Replace the single `model = EMCADNet(...)` statement near the bottom with:

```python
common_model_kwargs = dict(
    num_classes=args.num_classes,
    kernel_sizes=args.kernel_sizes,
    expansion_factor=args.expansion_factor,
    dw_parallel=not args.no_dw_parallel,
    add=not args.concatenation,
    lgag_ks=args.lgag_ks,
    activation=args.activation_mscb,
    encoder=args.encoder,
    pretrain=not args.no_pretrain,
    pretrained_dir=args.pretrained_dir,
)

if args.adaptive_msdc:
    if args.no_dw_parallel or args.concatenation:
        raise ValueError("adaptive MSDC requires parallel branches and additive aggregation")
    model = DGEMCADNet(
        **common_model_kwargs,
        router_mode=args.dg_router_mode,
        disagreement_lambda=args.dg_disagreement_lambda,
        router_temperature=args.dg_router_temperature,
        router_hidden=args.dg_router_hidden,
    )
else:
    model = EMCADNet(**common_model_kwargs)
```

## 6. `trainer.py`: imports

Add beside the current loss imports:

```python
from utils.dg_losses import (
    BoundaryPartitionDistillationLoss,
    ModelEMA,
    linear_ramp,
    routing_prediction_loss,
    routing_regularization,
    unpack_output,
)
```

## 7. `trainer.py`: create the EMA teacher/loss

After `model.to(device)`, `ce_loss`, and `dice_loss` have been created, add:

```python
ema_teacher = ModelEMA(model, decay=args.dg_ema_decay) if args.bp_distill else None
distill_criterion = None
if args.bp_distill:
    distill_criterion = BoundaryPartitionDistillationLoss(
        temperature=args.dg_temperature,
        confidence_threshold=args.dg_confidence,
        boundary_radius=args.dg_boundary_radius,
        stable_weight=1.0,
        boundary_weight=1.0,
        gradient_weight=0.5,
    ).to(device)
```

## 8. `trainer.py`: ensure training mode is restored every epoch

The current checkout calls validation (which switches to eval) and does not restore train
mode on the next epoch. Add this as the first line inside `for epoch_num in iterator:`:

```python
model.train()
```

This is a baseline correctness repair, not an innovation. Apply it to every compared run.

## 9. `trainer.py`: replace forward/loss/update block

Replace the block starting at `P = model(image_batch, mode='train')` and ending at
`optimizer.step()` with the following. Keep the existing learning-rate and checkpoint code.

```python
# Teacher first: no-grad activations are released before the student forward, reducing peak VRAM.
teacher_final = None
if ema_teacher is not None:
    ema_teacher.module.eval()
    with torch.no_grad():
        teacher_result = ema_teacher.module(image_batch, mode="test")
        teacher_outputs, _ = unpack_output(teacher_result)
        teacher_final = teacher_outputs[-1].detach()

if args.adaptive_msdc:
    student_result = model(image_batch, mode="train", return_aux=True)
else:
    student_result = model(image_batch, mode="train")
P, adaptive_aux = unpack_output(student_result)

if epoch_num == 0 and i_batch == 0:
    out_idxs = list(np.arange(len(P)))
    if args.supervision == "mutation":
        ss = [group for group in powerset(out_idxs)]
    elif args.supervision == "deep_supervision":
        ss = [[index] for index in out_idxs]
    else:
        ss = [[-1]]
    print(ss)

seg_loss = image_batch.new_zeros(())
for group in ss:
    if not group:
        continue
    logits = sum(P[index] for index in group)
    seg_loss = seg_loss + 0.3 * ce_loss(logits, label_batch.long())
    seg_loss = seg_loss + 0.7 * dice_loss(logits, label_batch, softmax=True)

route_aux_loss = image_batch.new_zeros(())
route_reg_loss = image_batch.new_zeros(())
route_stats = {}
if adaptive_aux is not None:
    route_aux_loss = routing_prediction_loss(
        adaptive_aux,
        label_batch,
        ce_loss=ce_loss,
        dice_loss=dice_loss,
    )
    route_reg_loss, route_stats = routing_regularization(adaptive_aux)

distill_loss = image_batch.new_zeros(())
distill_stats = {}
distill_ramp = 0.0
if distill_criterion is not None:
    distill_ramp = linear_ramp(
        epoch_num,
        warmup_epochs=args.dg_distill_warmup,
        ramp_epochs=args.dg_distill_ramp,
    )
    if distill_ramp > 0.0:
        distill_loss, distill_stats = distill_criterion(P, teacher_final, label_batch)

loss = seg_loss
loss = loss + args.dg_route_aux_weight * route_aux_loss
loss = loss + args.dg_route_reg_weight * route_reg_loss
loss = loss + args.dg_distill_weight * distill_ramp * distill_loss

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
optimizer.step()
if ema_teacher is not None:
    ema_teacher.update(model)
```

Add these TensorBoard records after `iter_num += 1`:

```python
writer.add_scalar("loss/seg", seg_loss.item(), iter_num)
writer.add_scalar("loss/route_aux", route_aux_loss.item(), iter_num)
writer.add_scalar("loss/route_reg", route_reg_loss.item(), iter_num)
writer.add_scalar("loss/distill", distill_loss.item(), iter_num)
writer.add_scalar("loss/distill_ramp", distill_ramp, iter_num)
for name, value in route_stats.items():
    writer.add_scalar("router/" + name, value.item(), iter_num)
for name, value in distill_stats.items():
    writer.add_scalar("distill/" + name, value.item(), iter_num)
```

## 10. `test_synapse.py`: construct the same student architecture

Add the same `--adaptive_msdc`, router mode/lambda/temperature/hidden arguments used in
training. Import `DGEMCADNet`. Replace model construction using the same
`common_model_kwargs` block from section 5. Do not create `ModelEMA` and do not enable
`return_aux`; normal test calls still receive `[p4,p3,p2,p1]` and use `P[-1]`.

The checkpoint path must include the same `_amsdc_<mode>` tag. Prefer adding an explicit
`--checkpoint` argument rather than relying on the current long auto-generated path.

## 11. First smoke test (not a result)

Use two batches and one epoch first. On an RTX 3060 Laptop 6 GB, start with batch size 2:

```powershell
python train_synapse.py `
  --adaptive_msdc `
  --bp_distill `
  --dg_router_mode disagreement `
  --batch_size 2 `
  --max_epochs 1
```

Required checks:

- Four final logits are `[B,9,H,W]`.
- Four router maps are `[B,3,H/32..H/4,W/32..W/4]`.
- No NaN/Inf in total, route, or distillation loss.
- All three branch usages are nonzero.
- `equal` mode gives the same branch aggregation formula as original MSDC.
- Test mode loads the DG checkpoint and still evaluates only `P[-1]`.

## 12. Required ablation matrix

Use the same split, preprocessing, encoder weights, seed set, optimizer, epochs and
validation-only checkpoint selection in all rows.

```text
A  Original EMCAD                  no flags
B  Equal adaptive scaffold         --adaptive_msdc --dg_router_mode equal
C  Learned global scales           --adaptive_msdc --dg_router_mode global
D  Feature-only dynamic scales     --adaptive_msdc --dg_router_mode feature
E  Disagreement-guided scales      --adaptive_msdc --dg_router_mode disagreement
F  Boundary partition distill      --bp_distill
G  Full method                     --adaptive_msdc --dg_router_mode disagreement --bp_distill
```

Run at least seeds `1234, 2222, 3407`; five seeds are preferable for the final baseline and
full model. Never select a checkpoint or tune hyperparameters on the final test set.

## 13. What to record for the paper

- Mean/std Dice, HD95, per-class and per-case results.
- Small-organ Dice and boundary metric (HD95 or ASD), not only global mean Dice.
- Params, FLOPs and measured latency for baseline and adaptive model.
- Router usage by stage and uncertainty quantile; save `routing_weights` heatmaps.
- Correlation between uncertainty and expected kernel size.
- Training time/VRAM separately from inference time/VRAM.
- Failed runs and every changed hyperparameter.

## 14. Expected improvement (absolute Dice percentage points, not a guarantee)

Before any local baseline exists, a defensible planning range is:

```text
Innovation 1 only:  +0.15 to +0.60 Dice points
Innovation 5 only:  +0.10 to +0.45 Dice points
Combined 1 + 5:     +0.25 to +0.90 Dice points
```

Zero or negative change is possible. Do not add the two ranges: both target ambiguous
boundaries/small structures and their gains may overlap. Treat `>= +0.30` mean points over
at least three seeds, with better small-organ/boundary metrics and no large efficiency
regression, as an encouraging signal. Treat one lucky seed as exploratory only.

## 15. Important scope limitation

The design is plausible but the phrase "uncertainty-guided adaptive multi-scale" is not
novel by itself. The paper claim must be narrower: adjacent decoder-scale disagreement
drives the existing EMCAD MSDC branches, paired with GT-boundary-partitioned EMA
cross-scale self-distillation. A formal related-work search is still required before claiming
novelty.

## 16. Verification completed on 2026-08-17

Using the repository's `SLDGroup_EMCAD_env` (PyTorch 1.11.0+cu113):

- Both reference Python files passed `py_compile`.
- A 4-class adaptive MSCB passed forward, routing-weight normalization and backward.
- A complete `DGEMCADNet` with a ResNet18 encoder returned four `[1,4,64,64]` logits.
- Its four route maps were `[1,3,2,2]`, `[1,3,4,4]`, `[1,3,8,8]`,
  `[1,3,16,16]`, matching decoder scales.
- Multiclass and binary boundary-partition distillation both passed backward.
- With PVTv2-B2 and 9 classes, the reference adaptive decoder had 1,956,199
  parameters versus 1,913,515 for the local baseline: +42,684 (+2.23% decoder
  parameters). FLOPs and hardware latency were not measured.

This is a synthetic tensor test, not dataset training and not evidence of a Dice gain.
