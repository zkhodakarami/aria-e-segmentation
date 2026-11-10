# ARIA-E lesion segmentation on FLAIR MRI

3D nnU-Net that segments ARIA-E lesions (amyloid-related imaging abnormalities,
edema/effusion type) on axial T2-FLAIR. ARIA-E is the hyperintense vasogenic
edema that shows up in patients on anti-amyloid antibodies such as lecanemab and
donanemab; in the clinic it is graded on a three-step severity scale, and this
model is the segmentation half of an effort to replace that scale with an
actual volume.

The model was trained on 49 manually segmented surveillance FLAIRs from 23
patients at one center, with subject-grouped 5-fold cross-validation. Training
runs through MONAI's `nnUNetV2Runner` (MONAI 1.4.0 + nnunetv2 2.6.4). The
patient data is not distributable, but the trained weights are — see
[Use the trained model](#use-the-trained-model).

## Results

Scored out-of-fold, so every scan is segmented by a model that never saw that
patient:

| metric | value |
|---|---|
| pooled (global) Dice | 0.74 |
| per-case mean / median Dice | 0.43 / 0.50 |
| lesion-wise sensitivity / precision / F1 | 0.53 / 0.66 / 0.59 |

Detection depends almost entirely on lesion size. Sensitivity by lesion volume
(voxels at native resolution, 5.0 × 0.69 × 0.69 mm):

| < 200 | 200–1k | 1k–5k | > 5k |
|---|---|---|---|
| 0.36 | 0.57 | 0.83 | 1.00 |

Every lesion above 5k voxels (about 12 mL at this resolution) is found; roughly
half the lesions in the cohort are under 200 voxels (~0.5 mL) and those drive
every headline number down. The per-case mean Dice is dragged by low-burden
cases where a single missed small lesion scores 0.0 — on cases above that
regime the model averages 0.56. The model under-calls rather than over-calls
(voxel recall 0.68 vs precision 0.81), and about 69% of cases are multifocal,
which is why keep-largest-component postprocessing is left off: it was tested
and only deletes true lesions.

Automated volumes correlate strongly with expert manual volumes despite the
modest voxel overlap, which is the regime these numbers should be read in:
small, scattered, low-contrast lesions where boundary Dice is a harsh metric.

### What did not work (so you don't retry it)

Two retrains were run against the baseline and both were rejected on a paired
lesion-level comparison:

- **Tversky + CE loss with FN weighting** (α=0.3, β=0.7, trainer in
  `nnunet_ext/nnUNetTrainer_TverskyCE.py`): meant to boost small-lesion recall,
  it did the opposite — detected fewer lesions than the baseline (lesion F1
  0.59 → 0.56), and every lesion it lost was small, exactly the population it
  targeted. It only improved boundary Dice on lesions the baseline already
  found. Lowering the decision threshold everywhere is not the same as finding
  new small lesions.
- **Two-channel input (current + rigidly registered prior FLAIR)**: best
  precision of the three models, worst recall, lower F1. The prior channel
  taught the model to be conservative, not sensitive.

The levers that remain are more labelled small-lesion data and a configuration
that preserves in-plane resolution (the 5 mm slice thickness is likely the real
ceiling).

## Use the trained model

The 5-fold ensemble is published as a release asset in nnU-Net's exchange
format. You need a working [nnunetv2](https://github.com/MIC-DKFZ/nnUNet)
install (see [Environment](#environment)) with its three result folders set:

```bash
export nnUNet_raw=/your/nnunet/raw
export nnUNet_preprocessed=/your/nnunet/preprocessed
export nnUNet_results=/your/nnunet/results

nnUNetv2_install_pretrained_model_from_zip aria_e_nnunet_baseline.zip
```

Input scans must be axial T2-FLAIR NIfTIs named the nnU-Net way, one channel
per case: `<case>_0000.nii.gz`. Then:

```bash
nnUNetv2_predict -i <input_dir> -o <output_dir> \
  -d 501 -c 3d_fullres -tr nnUNetTrainer_500epochs -f 0 1 2 3 4
```

`scripts/predict.sh` wraps the same call (edit the env paths at the top).
Output masks are binary: 0 background, 1 lesion.

Caveats worth repeating before you trust the output: the model was trained on
one center's surveillance protocol (thick-slice axial FLAIR, 5 mm), it
under-segments small lesions, and it has seen only ARIA-E — it will happily
segment other FLAIR hyperintensities it has no business labelling. Treat it as
a research tool, not a diagnostic one.

## Train on your own data

The pipeline assumes a cohort laid out as one folder per subject containing
longitudinal FLAIRs (filenames carrying a 14-digit acquisition datetime) and
manual masks named `YYYYMMDD_seg.nii.gz`. Edit the paths at the top of the
scripts to point at your data, then:

```bash
python scripts/build_dataset.py     # pair, verify, quarantine -> nnU-Net raw + splits
python scripts/make_datalist.py     # MONAI datalist + staging tree
./launch.sh                         # plan/preprocess + train 5 folds, detached in tmux
./launch.sh --skip-prep             # training only, if preprocessing is done
```

`build_dataset.py` is where the safety is. Each mask is matched to the FLAIR of
the same acquisition date and then verified two ways before it is allowed into
training:

1. **Grid identity** — mask and image must share shape and affine.
2. **Intensity alignment** — ARIA-E is hyperintense on FLAIR, so the mean
   intensity under the mask must sit well above the brain mean
   (z = (mean-in-mask − brain mean) / brain SD > 0.8).

Pairs failing either check are quarantined with a reason and a QC figure
(`scripts/qc_quarantine.py`), not trained on. This caught 6 of 55 masks in the
original cohort — five of them had intact headers but content drawn in a
different space, the signature of a broken resample step in an upstream
registration pipeline. Without the intensity check those masks would have gone
into training sitting on normal-appearing tissue. If you train on longitudinal
clinical data with externally produced masks, run this audit;
`scripts/check_alignment.py` and `scripts/audit_data.py` are the standalone
versions.

Folds are grouped by subject: the same patient never appears in both train and
validation. nnU-Net's default random split would leak timepoints of one patient
across the split and inflate validation Dice — with ~2 scans per patient this
is not a subtle effect.

Training details, all picked by nnU-Net's planner except where noted:
`3d_fullres`, patch 28 × 256 × 256 at 5.0 × 0.69 × 0.69 mm spacing, with
anisotropy-aware kernels (in-plane-only convolutions in the early stages).
`nnUNetTrainer_500epochs` instead of the default 1000 — enough for 49 cases and
half the wall time. Five folds in parallel, one GPU each, ~6 GB VRAM and ~6 h
per fold on an RTX A6000.

Two gotchas that cost real debugging time:

- `nnUNetV2Runner` shells out to the `nnUNetv2_train` console script, so the
  environment's `bin/` must be on `PATH`. Without it every fold dies instantly
  with `nnUNetv2_train: not found` — and the runner still reports success.
  `launch.sh` handles this.
- MONAI sends the training subprocess's stdout to `/dev/null`; the real
  training curves are in
  `.../nnUNet_trained_models/.../fold_N/training_log_*.txt`.

## Environment

```
conda create -n nnunet_ariae python=3.10
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install monai==1.4.0 nnunetv2==2.6.4 nibabel scipy matplotlib
```

The torch pin matters on older drivers: with CUDA 12.2 drivers, MONAI's default
dependency resolution pulls a torch build whose `torch.cuda.is_available()`
returns False. MONAI is pinned to 1.4.0 for the same reason (1.6 requires
torch ≥ 2.8).

The Tversky trainer in `nnunet_ext/` must be importable by nnunetv2 to be used
(or even named at inference time): symlink or copy it into
`nnunetv2/training/nnUNetTrainer/` inside the installed package. A dangling
symlink there breaks *all* nnU-Net inference with a `ModuleNotFoundError`,
because nnU-Net imports every module in that folder when resolving a trainer
name.

## Layout

```
scripts/build_dataset.py      pair + verify + quarantine -> nnU-Net raw, grouped splits
scripts/make_datalist.py      MONAI datalist + staging tree
scripts/prepare.py            fingerprint / plan / preprocess only
scripts/run_pipeline.py       nnUNetV2Runner: prep -> train 5 folds in parallel
scripts/predict.sh            inference with the 5-fold ensemble
scripts/check_alignment.py    standalone mask-on-hyperintensity audit
scripts/audit_data.py         per-subject geometry / date inventory
scripts/qc_quarantine.py      QC figure for rejected pairs
scripts/make_matched_csv.py   build the (file_id, date, flair, seg) manifest
scripts/fw_download_t1.py     fetch matching T1s from a Flywheel instance
nnunet_ext/                   Tversky+CE trainer (kept as a documented negative result)
configs/input.yaml            nnUNetV2Runner input config
launch.sh                     tmux-detached entrypoint
```

## Data

The imaging data is clinical and cannot be shared. Nothing in this repository
contains patient data; the manifests, audit outputs, and QC figures the
pipeline writes stay on the machine that runs it. The released weights contain
network parameters, training-curve plots, nnU-Net's plans and dataset
fingerprint (image geometry and intensity statistics, no identifiers), and the
cross-validation postprocessing decision — no imaging data and no per-case
information.

## License

MIT — see [LICENSE](LICENSE).
