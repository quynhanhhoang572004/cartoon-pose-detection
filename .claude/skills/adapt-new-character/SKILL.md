---
name: adapt-new-character
description: >
  Adapt this repo's cartoon CAPE pose model to a NEW character — annotate a few
  shots, fit a test-time adapter, pseudo-label the unlabelled frames,
  self-train, and measure the gain. Use this whenever the work touches a
  character the model has not been trained on: adding a character, running
  test_tta.py or generate_pseudo.py or adapt_character.py or train_pseudo.sh,
  building a support/query split, choosing TTA or pseudo-label thresholds,
  reading a PCK number, or setting up the Kaggle environment or datasets for
  any of it. Use it even when the request names only one step ("just generate
  the pseudo-labels", "why is PCK so low", "zip this for Kaggle"), because the
  steps share a split and a set of traps that are expensive to rediscover.
---

# Adapting to a new character

The model is category-agnostic: what a keypoint *means* for a character comes
from the labelled support images at inference, not from the weights. So a new
character needs a handful of labelled shots, not a retrain. Everything below
builds on that.

**The character is always a parameter.** Goofy is only the current test case for
the paper. Never create `foo_goofy.py` or a config with a character's name in
it — extend the entry points below so the next character, and the next person,
costs one command.

## The pipeline

One driver runs all of it:

```bash
cd PoseAnything_ours
python adapt_character.py \
    --name <char> --category <id> \
    --config configs/cartoon/train_merged_supcon.py \
    --checkpoint <base.pth> \
    --coco ../data/coco_<char>.json --img-dir ../data/<char>_eval/images \
    --unlabeled-dir ../data/<char>_frames \
    --labeled-coco ../data/merged/coco_train.json \
    --out-dir ../data/adapt_<char>
```

| step | what it does | output |
|---|---|---|
| 1 split | K support / rest query | `support.json`, `query.json` |
| 2 measure | PCK with and without TTA | `tta.txt`, `<char>_adapter.pth` |
| 3 quality | how good the pseudo-labels would be | `quality.txt` |
| 4 generate | pseudo-label the unlabelled frames | `pseudo_<char>.json` |
| 5 merge | real + support + pseudo | `train_<char>.json` |
| 6 print | the train and final-measure commands | — |

`--steps 4,5,6` resumes partway. Training is deliberately not run here — step 6
prints the command, and `train_pseudo.sh <char> <cat> [variant]` runs it with
the before/after measurement around it.

### Before step 1: two one-line registrations

A new character needs its name in two class maps, or its annotations are
dropped with a single line that scrolls past:

- `cvat2yolo_pose.py` → `CHARACTER_TO_CLASS` (YOLO class id)
- `merge_coco.py` → `CANONICAL` (COCO category id = YOLO class + 1)

These are data, not code — adding a row is fine. Adding a *file* per character
is not.

## Reading the numbers

Two facts about this codebase's PCK decide whether a result means anything.
Getting either wrong makes a working model look broken, which has cost a full
day before.

**Only joints visible in the query AND in every support shot are scored.**
`transformer_base_dataset._report_metric` ANDs the visibility mask over the
whole episode, and this is not gratuitous strictness: `head.py` zeroes an
invisible support keypoint's embedding and masks it out of attention, so the
coordinate the model still emits for that slot is meaningless. `test_tta.py`
reproduces this via `support_visible()` and prints the unmasked column beside
it. Report both — the masked column is the apples-to-apples CAPE number, the
unmasked one is what a user of the model actually gets.

**The `PCK` in training logs is PCK@0.2**, not @0.05 (`_report_metric` defaults
to `pck_thr=0.2`). Normalisation is `max(bbox_w, bbox_h)` everywhere.

A consequence worth knowing before designing an experiment: with 5 shots only
about a third of the 21 joints survive the AND, and the survivors are the easy
central ones. The intersection shrinks as K grows, so **a PCK-vs-K curve is not
comparable across K** unless the `n_sup/21` line is reported with it.

## Settings that are already tuned

These came out of a sweep; start here rather than re-deriving them.

| | value | why not otherwise |
|---|---|---|
| `--mode` | `norm` | `norm_head` **diverges** (0.470 → 0.075, loss rises). 5 shots cannot constrain the head |
| `--lr` | `1e-3` | `1e-4` gives +0.005 and looks like failure; `3e-3` is worse and unstable |
| `--steps` | `20` | saturates there; 40 and 80 add nothing while the support loss keeps falling |
| pseudo teacher | base + adapter | more kept labels at indistinguishable accuracy |
| filter | `--eps 0.05 --min-visible 4 --emb-thresh 0.35` | see below |

Pick filter thresholds by **coverage × accuracy** — the count of correct labels
is what self-training consumes — subject to accuracy staying well above the
model's own PCK@0.05, or the labels add nothing. On Goofy, 0.03/0.25/6 gave
0.221 × 0.611 while 0.05/0.35/4 gave 0.319 × 0.599: 41% more correct labels for
a 0.012 accuracy cost. `--min-visible` is a *frame*-level cut and bites hardest
(44 frames kept at 6, 490 at 4).

## Measuring the result

The final number must use `--trials 1 --seed 0`, which reuses the split step 1
wrote: the support shots that are in training, and the query images that are
not. Any other seed draws a different support set whose queries include images
the model trained on.

`test_tta.py` prints the no-TTA and with-TTA columns from one call, so two
calls fill the table the paper needs:

```
                  no TTA      + TTA
  base              .           .
  self-trained      .           .
```

`train_pseudo.sh` does both around the training run. Train from the **base**,
not base+adapter: the adapter already contributed by being the teacher, and
folding it into the starting weights would put TTA inside the "self-trained, no
TTA" cell where the two could no longer be separated.

Report the delta **paired** per support draw. The draw dominates the absolute
numbers (±0.025) while the effect itself is tight (±0.012), so an unpaired mean
over a few trials hides a real effect.

### Calibration from Goofy

Useful for noticing when something is wrong rather than merely disappointing.

| | PCK@0.05 | PCK@0.2 |
|---|---|---|
| characters in training, 5-shot | 0.84 – 0.94 | 0.99 – 1.00 |
| Goofy, unseen | 0.470 | 0.876 |
| Goofy + TTA | 0.541 (+0.072 ± 0.012) | 0.920 |

If a character the model *was* trained on scores ~0.3, the evaluator is wrong,
not the model. TTA gains should scale with the gap: on trained characters it
gives about +0.001 at @0.2, which is what makes this adaptation rather than
under-training. The adapter is ~0.08% of parameters, about 0.2 MB.

## Traps

- **`valid_class_ids` must include the new category.** The base config stops at
  5. Omit the new id and training completes, logs look healthy, and nothing
  about the character was ever sampled.
- **mmcv's `--cfg-options` splits argv on whitespace.** Emit
  `valid_class_ids="[1,2,3,4,5,7]"` with no spaces.
- **`data.val.*` needs overriding off-server.** The base config's `DATA_DIR` is
  an absolute server path; the eval hook dies mid-run without it.
- **Never put `\` after `!` in a notebook cell.** IPython mangles the
  continuation. Build the command as an f-string and run `!{cmd}` on one line —
  it also prints what actually ran, so an unset variable shows up immediately.
- **Query images never enter the training json.** They are the only clean
  measurement.
- **The teacher's support set must be the split's support set.** The adapter
  records which images it saw; `generate_pseudo.py --adapter` prints them.

## Kaggle

The repo's stack is from 2022 and Kaggle's image is current, so the environment
needs a separate Python 3.11 venv, and the datasets need versioning rather than
recreating. Read `references/kaggle.md` before touching either — both have
non-obvious failure modes that are quick to hit and slow to diagnose.
