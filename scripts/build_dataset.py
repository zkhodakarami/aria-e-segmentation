"""Build nnU-Net raw dataset from the ARIA-E cohort (one folder per subject).
Pairs each *_seg.nii.gz with the FLAIR of the same acquisition date, verifies
voxel-grid identity AND intensity alignment, and QUARANTINES pairs that fail.
Emits subject-grouped 5-fold splits (no timepoint leakage across folds)."""
import os, re, json, shutil, csv
import numpy as np, nibabel as nib

ROOT = "/path/to/cohort"                                  # one folder per subject
RAW  = "/path/to/nnunet/nnUNet_raw/Dataset501_ARIAE"
PP   = "/path/to/nnunet/nnUNet_preprocessed/Dataset501_ARIAE"
AUD  = "audit"
Z_MIN = 0.8   # min hyperintensity z for a trustworthy label

os.makedirs(f"{RAW}/imagesTr", exist_ok=True)
os.makedirs(f"{RAW}/labelsTr", exist_ok=True)
os.makedirs(AUD, exist_ok=True)

def zscore(a, mask):
    brain = a > np.percentile(a, 60)
    mu, sd = a[brain].mean(), a[brain].std()
    return float((a[mask].mean() - mu) / sd) if sd > 0 else float("nan")

good, quarantine = [], []
for s in sorted(os.listdir(ROOT)):
    sd = os.path.join(ROOT, s)
    if not os.path.isdir(sd): continue
    subj = re.match(r"(\d+)", s).group(1)
    files = sorted(os.listdir(sd))
    imgs = [f for f in files if f.endswith(".nii.gz") and "_seg" not in f]
    segs = [f for f in files if f.endswith(".nii.gz") and "_seg" in f]
    for sf in segs:
        m = re.match(r"(\d{8})_seg", sf)
        if not m:
            quarantine.append(dict(subject=subj, seg=sf, image="", reason="no date in seg filename")); continue
        date = m.group(1)
        sg = nib.load(os.path.join(sd, sf))
        lab = (np.asanyarray(sg.dataobj) > 0)
        if lab.sum() == 0:
            quarantine.append(dict(subject=subj, seg=sf, image="", reason="empty label")); continue
        cands = [f for f in imgs if (re.findall(r"(\d{8})\d{6}", f) or [None])[0] == date]
        paired = None
        for f in cands:
            im = nib.load(os.path.join(sd, f))
            if im.shape == sg.shape and np.allclose(im.affine, sg.affine, atol=1e-3):
                paired = f; break
        if paired is None:
            quarantine.append(dict(subject=subj, seg=sf, image=";".join(cands),
                                   reason=f"no same-date image with matching grid ({len(cands)} date candidates)")); continue
        a = np.asanyarray(nib.load(os.path.join(sd, paired)).dataobj).astype(np.float32)
        z = zscore(a, lab)
        rec = dict(subject=subj, seg=sf, image=paired, date=date, z=round(z, 3), vox=int(lab.sum()))
        if not (z > Z_MIN):
            rec["reason"] = f"label not on hyperintense tissue (z={z:.2f}) - misregistered/corrupt"
            quarantine.append(rec)
        else:
            good.append(rec)

good.sort(key=lambda r: (r["subject"], r["date"]))
manifest = []
for i, r in enumerate(good, 1):
    cid = f"ARIAE_{i:03d}"
    src_img = os.path.join(ROOT, [d for d in os.listdir(ROOT) if d.startswith(r["subject"])][0], r["image"])
    src_seg = os.path.join(ROOT, [d for d in os.listdir(ROOT) if d.startswith(r["subject"])][0], r["seg"])
    shutil.copyfile(src_img, f"{RAW}/imagesTr/{cid}_0000.nii.gz")
    # normalize label to uint8 {0,1}
    sg = nib.load(src_seg)
    out = nib.Nifti1Image((np.asanyarray(sg.dataobj) > 0).astype(np.uint8), sg.affine, sg.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, f"{RAW}/labelsTr/{cid}.nii.gz")
    r["case_id"] = cid; manifest.append(r)

json.dump({"channel_names": {"0": "FLAIR"},
           "labels": {"background": 0, "lesion": 1},
           "numTraining": len(manifest),
           "file_ending": ".nii.gz",
           "dataset_name": "Dataset501_ARIAE",
           "description": "ARIA-E FLAIR lesion segmentation"},
          open(f"{RAW}/dataset.json", "w"), indent=2)

with open(f"{AUD}/manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["case_id","subject","date","image","seg","z","vox"]); w.writeheader()
    for r in manifest: w.writerow({k: r[k] for k in w.fieldnames})
with open(f"{AUD}/quarantined.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["subject","seg","image","reason","z","vox"], extrasaction="ignore")
    w.writeheader()
    for r in quarantine: w.writerow(r)

# ---- subject-grouped 5-fold splits (prevents same-patient leakage) ----
subs = sorted({r["subject"] for r in manifest})
rng = np.random.RandomState(42); rng.shuffle(subs)
folds = [subs[i::5] for i in range(5)]
splits = []
for k in range(5):
    val_s = set(folds[k])
    splits.append({"train": sorted(r["case_id"] for r in manifest if r["subject"] not in val_s),
                   "val":   sorted(r["case_id"] for r in manifest if r["subject"] in val_s)})
os.makedirs(PP, exist_ok=True)
json.dump(splits, open(f"{PP}/splits_final.json","w"), indent=1)

print(f"TRAINING CASES : {len(manifest)}  from {len(subs)} subjects")
print(f"QUARANTINED    : {len(quarantine)}")
for r in quarantine: print(f"   - {r['subject']}/{r['seg']:24s} {r.get('reason')}")
print("\nfold sizes (val cases / val subjects):")
for k,sp in enumerate(splits): print(f"  fold {k}: train={len(sp['train']):3d} val={len(sp['val']):3d}  subjects={sorted(folds[k])}")
print("\nlesion voxels: min", min(r['vox'] for r in manifest), "median", int(np.median([r['vox'] for r in manifest])), "max", max(r['vox'] for r in manifest))
