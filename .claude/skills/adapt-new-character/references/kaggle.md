# Running this on Kaggle

Two things are hostile here: the Python stack (mmcv-full 1.x is from 2022,
Kaggle's image is current) and the habit of creating a new dataset per
iteration. Both are solved once below.

## Datasets: version them, do not recreate

A new dataset per round leaves the notebook pointing at a stale one and a
gallery full of near-duplicates. `kaggle datasets version` adds to an existing
dataset and keeps the same slug, so the notebook's attached input only needs
its version bumped.

```bash
# once per dataset
mkdir -p up && cp <files> up/
kaggle datasets init -p up                       # edit id: <user>/<slug>
kaggle datasets create -p up --dir-mode zip

# every time after
cp <changed files> up/
kaggle datasets version -p up -m "regenerated pseudo-labels, eps 0.05" --dir-mode zip
```

Keep the number of datasets small and stable, split by **how often they
change**, not by character:

| dataset | holds | changes |
|---|---|---|
| base training data | the merged characters, their images, `coco_train/val.json` | rarely |
| character data | each character's labelled and unlabelled frames + jsons | when a character is added |
| checkpoints | base `.pth`, adapters | per training run |

Adapters are ~0.2 MB, so a character's adapter belongs with its data rather
than in a dataset of its own.

In the notebook, resolve paths by globbing for a **filename**, never by
hard-coding a dataset slug — the slug changes when someone renames a dataset,
the filename does not:

```python
CKPT = glob.glob('/kaggle/input/**/base.pth', recursive=True)[0]
```

## Environment

Kaggle's interpreter is far newer than mmcv-full 1.x supports, so build a
separate Python 3.11 venv beside it and run everything with that interpreter.
Do not try to install into Kaggle's own Python; there is no wheel for it.

```python
import os
os.environ["UV_VENV_CLEAR"] = "1"     # otherwise uv blocks on a [y/n] with no stdin
os.environ["MPLBACKEND"]    = "Agg"   # Kaggle points MPLBACKEND at a package the venv lacks
ENV, PY = "/kaggle/temp/env", "/kaggle/temp/env/bin/python"

!pip -q install uv
if not os.path.exists(PY):
    !uv venv --python 3.11 --seed {ENV}

!uv pip install -q --python {PY} torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# numpy<2 must share ONE resolve with scipy/matplotlib, or they pull numpy 2.x
# back in and every compiled module fails with "_ARRAY_API not found".
# setuptools<70 because torch 2.0.1 imports pkg_resources, removed in 81+.
!uv pip install -q --python {PY} \
    "numpy<2" "setuptools<70" "timm==0.9.16" "opencv-python-headless<4.10" \
    "yapf==0.40.1" scipy matplotlib pillow json_tricks munkres cython wheel

!{PY} -m pip install -q mmcv-full==1.7.2 --only-binary mmcv-full \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
!{PY} -m pip install -q --no-build-isolation xtcocotools
# --no-deps skips chumpy (SMPL); its setup.py imports pip and nothing here uses it
!uv pip install -q --python {PY} --no-deps mmpose==0.29.0

# LAST, after everything else. The two `pip install` calls above resolve on
# their own and have no reason to respect numpy<2, so they pull numpy 2.x back
# in and every compiled module then dies on "_ARRAY_API not found". Re-pinning
# here only downgrades numpy -- mmcv, cv2 and xtcocotools were built against
# 1.x and need it only at runtime, so nothing is reinstalled. Bitten three
# times in this project; keep this line last.
!{PY} -m pip install -q "numpy<2"

# mmpose 0.29.0 caps mmcv at 1.7.0 but torch 2.0 only has a 1.7.2 wheel, which
# is API-compatible. Patch without importing mmpose -- the import is the assert.
import subprocess, pathlib
sp = subprocess.check_output([PY, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"]).decode().strip()
f = pathlib.Path(sp) / "mmpose" / "__init__.py"
f.write_text(f.read_text().replace("mmcv_maximum_version = '1.7.0'",
                                   "mmcv_maximum_version = '1.7.3'"))
```

Why each pin exists, since they look arbitrary and get "simplified" away:

| pin | without it |
|---|---|
| Python 3.11 venv | no mmcv-full 1.x wheel for Kaggle's interpreter; source build fails |
| `torch==2.0.1+cu118` | mmcv-full's compiled ops must match the torch they were built against |
| `mmcv-full==1.7.2` | 1.6.2 has no torch-2.0 wheel at all |
| `numpy<2` *in the same command as scipy/matplotlib* | resolved separately, numpy jumps to 2.x and breaks every compiled module |
| `setuptools<70` | `from pkg_resources import packaging` in torch 2.0.1 |
| `timm==0.9.16` | `timm.models.layers` was removed in 1.0; the Swin backbone imports it |
| `opencv-python-headless<4.10` | newer builds require numpy>=2 |

Verify before running anything long:

```python
!MPLBACKEND=Agg {PY} -c "import torch,mmcv,mmpose,timm,numpy;print(torch.__version__,torch.cuda.is_available(),mmcv.__version__,mmpose.__version__,numpy.__version__)"
!MPLBACKEND=Agg {PY} -c "from mmcv.ops import nms; print('ops OK')"
!MPLBACKEND=Agg {PY} -c "from mmpose.models import build_posenet; print('posenet OK')"
```

## Session shape

`/kaggle/temp` does not survive a session, so a session that ends takes the
venv with it — put the environment cell first in the notebook so a cold start
rebuilds it. `/kaggle/working` does persist as the notebook's output.

Training runs hours (roughly 3.5 min/epoch on the project's server GPU, 2–3×
that on a T4). Use **Save Version → Save & Run All** so it runs detached; an
interactive session that loses its connection loses the run.

`/kaggle/input` is read-only and training needs one flat `img_prefix`, so copy
the images into `/kaggle/working` first.

If the server with the working environment is available, prefer it: no session
limit, the data is already there, and it is several times faster. Kaggle is
worth this setup only when the server is not an option.
