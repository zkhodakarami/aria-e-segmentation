"""Emit a MONAI datalist (with subject-grouped `fold` keys) + staging tree
that nnUNetV2Runner can convert into an nnU-Net raw dataset."""
import os, json, csv
RAW="/path/to/nnunet/nnUNet_raw/Dataset501_ARIAE"
STG="/path/to/nnunet/staging"
OUT="configs/datalist.json"
os.makedirs(f"{STG}/imagesTr", exist_ok=True); os.makedirs(f"{STG}/labelsTr", exist_ok=True)
rows=list(csv.DictReader(open("audit/manifest.csv")))
splits=json.load(open("/path/to/nnunet/nnUNet_preprocessed/Dataset501_ARIAE/splits_final.json"))
fold_of={}
for k,sp in enumerate(splits):
    for c in sp["val"]: fold_of[c]=k
training=[]
for r in rows:
    cid=r["case_id"]
    for sub,src,dst in [("imagesTr", f"{RAW}/imagesTr/{cid}_0000.nii.gz", f"{STG}/imagesTr/{cid}.nii.gz"),
                        ("labelsTr", f"{RAW}/labelsTr/{cid}.nii.gz",      f"{STG}/labelsTr/{cid}.nii.gz")]:
        if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
        os.symlink(src, dst)
    training.append({"image": f"imagesTr/{cid}.nii.gz", "label": f"labelsTr/{cid}.nii.gz", "fold": fold_of[cid]})
json.dump({"training": training, "testing": []}, open(OUT,"w"), indent=1)
print(f"datalist: {OUT}  cases={len(training)}")
from collections import Counter
print("cases per fold:", dict(sorted(Counter(t['fold'] for t in training).items())))
